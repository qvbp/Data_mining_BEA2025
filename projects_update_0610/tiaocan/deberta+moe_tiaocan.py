"""
DeBERTa MoE单任务学习 - 贝叶斯调参版本

使用Optuna进行超参数优化，自动搜索最佳参数组合
调参参数: num_experts, top_k, expert_mlp_layers, expert_hidden_size, 
         load_balance_weight, batch_size, lr, weight_decay, seed

运行方法:
python train_single_task_bayesian.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import optuna
import pickle
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定配置
FIXED_CONFIG = {
    'model_name': '/mnt/cfs/huangzhiwei/BAE2025/models/deberta-v3-base',
    'num_classes': 3,
    'freeze_pooler': 0,
    'main_task': 'Actionability',  # 可以改为其他任务：'Mistake_Identification', 'Mistake_Location', 'Providing_Guidance'
    'max_length': 512,
    'epochs': 40,  # 调参时减少epoch数以节省时间
    'patience': 8,
    'warmup_ratio': 0.1,
    'data_path': '/mnt/cfs/huangzhiwei/Data_mining_BAE2025/data_new/train.json',
    'val_data_path': '/mnt/cfs/huangzhiwei/Data_mining_BAE2025/data_new/val.json',
    'checkpoint_dir': '/mnt/cfs/huangzhiwei/Data_mining_BAE2025/projects_update_0610/tiaocan/single_task_bayesian_optimization_results',
    'device': device
}


class BAE2025Dataset(Dataset):
    def __init__(
            self,
            data_path,
            label_type="Actionability",
            labels={
                "Yes": 0,
                "To some extent": 1, 
                "No": 2,
            }
    ):
        self.data_path = data_path
        self.label_type = label_type
        self.labels = labels
        self._get_data()
    
    def _get_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.data = []
        for item in data:
            if 'conversation_history' in item and 'response' in item:
                sent1 = item['conversation_history']
                sent2 = item['response']
                
                # 检查item中是否直接包含我们需要的标签
                if self.label_type in item and item[self.label_type] in self.labels:
                    self.data.append(((sent1, sent2), self.labels[item[self.label_type]]))
    
    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        return self.data[idx]


class BAE2025DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=16,
        max_length=512,
        shuffle=True,
        drop_last=True,
        device=None,
        tokenizer_name='/mnt/cfs/huangzhiwei/BAE2025/models/deberta-v3-base'
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.truncation_side = 'left'  # 设置截断方向为左侧
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.drop_last = drop_last

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )

    def collate_fn(self, data):
        sents = [i[0] for i in data]
        labels = [i[1] for i in data]

        # 处理两个句子的情况
        data = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(sent[0], sent[1]) for sent in sents],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_length=True
        )
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        labels = torch.LongTensor(labels).to(self.device)

        return input_ids, attention_mask, labels

    def __iter__(self):
        for data in self.loader:
            yield data

    def __len__(self):
        return len(self.loader)


class MLPExpert(nn.Module):
    """MLP专家模型"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        current_size = input_size
        
        # 构建多层MLP
        for i in range(num_layers):
            if i == num_layers - 1:  # 最后一层
                layers.append(nn.Linear(current_size, hidden_size))
            else:
                layers.append(nn.Linear(current_size, hidden_size))
                layers.append(nn.LayerNorm(hidden_size))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                current_size = hidden_size
                
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size] 或 [batch_size, input_size]
        if len(x.shape) == 3:
            # 如果是序列输入，使用平均池化
            x = torch.mean(x, dim=1)  # [batch_size, input_size]
        return self.mlp(x)  # [batch_size, hidden_size]


class BertClassificationHead(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=3, dropout_prob=0.3):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_classes)
    
    def forward(self, features):
        # 提取 [CLS] 标记的表示
        x = features[:, 0, :]  # 使用第一个标记([CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DynamicMoERouter(nn.Module):
    """动态专家路由器，支持topK选择"""
    def __init__(self, input_size, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # 确保top_k不超过专家数量
        self.router = nn.Linear(input_size, num_experts)
        
    def forward(self, x):
        # x: [batch_size, input_size]
        batch_size = x.size(0)
        
        # 计算路由logits
        router_logits = self.router(x)  # [batch_size, num_experts]
        
        # 计算所有专家的softmax概率（用于负载均衡损失）
        all_probs = F.softmax(router_logits, dim=-1)  # [batch_size, num_experts]
        
        # 选择topK个专家
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # 对选中的专家应用softmax
        top_k_probs = F.softmax(top_k_logits, dim=-1)  # [batch_size, top_k]
        
        # 创建稀疏的路由权重矩阵
        routing_weights = torch.zeros_like(router_logits)  # [batch_size, num_experts]
        routing_weights.scatter_(1, top_k_indices, top_k_probs)
        
        return routing_weights, top_k_indices, all_probs  # [batch_size, num_experts], [batch_size, top_k], [batch_size, num_experts]


class DeBERTaMoEClassifier(nn.Module):
    def __init__(
        self, 
        pretrained_model_name, 
        num_classes=3, 
        freeze_pooler=0,
        expert_hidden_size=256,
        dropout=0.3,
        num_experts=16,  # 新增：专家数量
        top_k=4,  # 新增：每次选择的专家数量
        expert_mlp_layers=2,  # 新增：专家MLP层数
        load_balance_weight=0.01,  # 新增：负载均衡损失权重
    ):
        super().__init__()
        
        # 使用 AutoModel 加载 DeBERTa 模型
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # 获取 bert 隐藏层大小
        self.bert_hidden_size = self.bert.config.hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # 确保top_k不超过num_experts
        self.load_balance_weight = load_balance_weight  # 负载均衡损失权重
        
        # 保留原有的分类头
        self.original_classifier = BertClassificationHead(
            hidden_size=self.bert_hidden_size,
            num_classes=num_classes,
            dropout_prob=dropout
        )
        
        # 创建多个MLP专家
        self.experts = nn.ModuleList([
            MLPExpert(
                input_size=self.bert_hidden_size,
                hidden_size=expert_hidden_size,
                num_layers=expert_mlp_layers,
                dropout=dropout
            ) for _ in range(num_experts)
        ])
        
        # 创建动态路由器
        self.router = DynamicMoERouter(
            input_size=self.bert_hidden_size, 
            num_experts=num_experts,
            top_k=self.top_k
        )
        
        # 专家输出映射层
        self.expert_output_proj = nn.Linear(expert_hidden_size, num_classes)
        
        # 最终的融合层
        # 原始分类头 + MoE输出 = 2 * num_classes
        combined_dim = num_classes * 2
        self.final_classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(combined_dim // 2, num_classes)
        )
    
    def compute_load_balance_loss(self, all_probs, top_k_indices):
        """
        计算负载均衡损失
        
        Args:
            all_probs: [batch_size, num_experts] - 所有专家的softmax概率
            top_k_indices: [batch_size, top_k] - 选中的专家索引
            
        Returns:
            load_balance_loss: 标量张量
        """
        batch_size = all_probs.size(0)
        
        # 1. 计算每个专家的平均选择概率 (importance)
        importance = torch.mean(all_probs, dim=0)  # [num_experts]
        
        # 2. 计算每个专家被选中的频率 (frequency)
        # 创建one-hot编码矩阵表示哪些专家被选中
        selection_mask = torch.zeros_like(all_probs)  # [batch_size, num_experts]
        
        # 将选中的专家位置设为1
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]  # [batch_size]
            selection_mask.scatter_(1, expert_indices.unsqueeze(1), 1.0)
        
        # 计算每个专家被选中的频率
        frequency = torch.mean(selection_mask, dim=0)  # [num_experts]
        
        # 3. 计算负载均衡损失 (Switch Transformer style)
        # auxiliary_loss = α * num_experts * Σ(f_i * P_i)
        # 其中 f_i 是专家i被选择的频率，P_i 是专家i的平均重要性
        auxiliary_loss = self.num_experts * torch.sum(frequency * importance)
        
        # 4. 可选：添加方差损失来进一步鼓励均匀分布
        target_frequency = 1.0 / self.num_experts
        frequency_variance = torch.var(frequency)
        importance_variance = torch.var(importance)
        
        # 总负载均衡损失
        load_balance_loss = auxiliary_loss + frequency_variance + importance_variance
        
        return load_balance_loss
        
    def forward(self, input_ids, attention_mask, return_loss=True):
        # DeBERTa 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 获取序列隐藏状态
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        batch_size = hidden_states.size(0)
        
        # 获取原始分类头结果
        original_logits = self.original_classifier(hidden_states)  # [batch_size, num_classes]
        
        # 获取路由权重和选中的专家索引
        cls_embedding = hidden_states[:, 0]  # [batch_size, hidden_size]
        routing_weights, top_k_indices, all_probs = self.router(cls_embedding)  # 获取所有概率用于负载均衡损失
        
        # MoE前向传播
        moe_output = torch.zeros(batch_size, self.expert_output_proj.out_features, device=hidden_states.device)
        
        # 只对选中的专家进行计算
        for i in range(self.top_k):
            # 获取当前位置所有样本选中的专家索引
            expert_idx = top_k_indices[:, i]  # [batch_size]
            
            # 为每个样本计算对应专家的输出
            for batch_idx in range(batch_size):
                current_expert_idx = expert_idx[batch_idx].item()
                current_weight = routing_weights[batch_idx, current_expert_idx]
                
                # 计算专家输出
                expert_hidden = self.experts[current_expert_idx](hidden_states[batch_idx:batch_idx+1])  # [1, expert_hidden_size]
                expert_logits = self.expert_output_proj(expert_hidden)  # [1, num_classes]
                
                # 加权累加
                moe_output[batch_idx] += current_weight * expert_logits.squeeze(0)
        
        # 拼接原始分类头和MoE输出
        combined_logits = torch.cat([original_logits, moe_output], dim=1)  # [batch_size, 2*num_classes]
        
        # 通过最终分类器输出最终结果
        final_logits = self.final_classifier(combined_logits)
        
        # 计算负载均衡损失
        if return_loss and self.training:
            load_balance_loss = self.compute_load_balance_loss(all_probs, top_k_indices)
            return {
                'logits': final_logits,
                'load_balance_loss': load_balance_loss,
                'routing_weights': routing_weights,
                'expert_utilization': torch.mean(all_probs, dim=0)  # 专家平均利用率
            }
        else:
            return final_logits
    
    def get_expert_utilization(self, input_ids, attention_mask):
        """获取专家使用统计，用于分析负载均衡"""
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            cls_embedding = hidden_states[:, 0]
            routing_weights, top_k_indices, all_probs = self.router(cls_embedding)
            
            # 统计每个专家被选中的次数
            expert_counts = torch.zeros(self.num_experts)
            for i in range(self.top_k):
                expert_idx = top_k_indices[:, i]
                for idx in expert_idx:
                    expert_counts[idx.item()] += 1
            
            # 计算负载均衡损失
            load_balance_loss = self.compute_load_balance_loss(all_probs, top_k_indices)
                    
            return {
                'expert_counts': expert_counts,
                'routing_weights': routing_weights, 
                'top_k_indices': top_k_indices,
                'all_probs': all_probs,
                'load_balance_loss': load_balance_loss,
                'expert_utilization': torch.mean(all_probs, dim=0)  # 专家平均利用率
            }


def train_single_trial(params):
    """单次训练函数，返回最佳验证F1分数"""
    
    # 设置随机种子
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 加载数据集
    train_dataset = BAE2025Dataset(
        FIXED_CONFIG['data_path'], 
        label_type=FIXED_CONFIG['main_task']
    )
    val_dataset = BAE2025Dataset(
        FIXED_CONFIG['val_data_path'], 
        label_type=FIXED_CONFIG['main_task']
    )

    # 创建数据加载器
    train_dataloader = BAE2025DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        max_length=FIXED_CONFIG['max_length'],
        shuffle=True,
        drop_last=True,
        device=FIXED_CONFIG['device'],
        tokenizer_name=FIXED_CONFIG['model_name']
    )

    val_dataloader = BAE2025DataLoader(
        dataset=val_dataset,
        batch_size=params['batch_size'],
        max_length=FIXED_CONFIG['max_length'],
        shuffle=False,
        drop_last=False,
        device=FIXED_CONFIG['device'],
        tokenizer_name=FIXED_CONFIG['model_name']
    )
    
    # 创建模型
    model = DeBERTaMoEClassifier(
        pretrained_model_name=FIXED_CONFIG['model_name'],
        num_classes=FIXED_CONFIG['num_classes'],
        freeze_pooler=FIXED_CONFIG['freeze_pooler'],
        expert_hidden_size=params['expert_hidden_size'],
        dropout=params['dropout'],
        num_experts=params['num_experts'],
        top_k=params['top_k'],
        expert_mlp_layers=params['expert_mlp_layers'],
        load_balance_weight=params['load_balance_weight']
    ).to(FIXED_CONFIG['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # 训练循环
    for epoch in range(FIXED_CONFIG['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels_list = []
        
        for input_ids, attention_mask, labels in train_dataloader:
            optimizer.zero_grad()
            
            # 前向传播 - 获取MoE输出
            outputs = model(input_ids, attention_mask, return_loss=True)
            
            if isinstance(outputs, dict):
                # 训练模式，包含负载均衡损失
                logits = outputs['logits']
                load_balance_loss = outputs['load_balance_loss']
            else:
                # 推理模式，只有logits
                logits = outputs
                load_balance_loss = torch.tensor(0.0, device=logits.device)
            
            # 计算分类损失
            labels = labels.long()
            classification_loss = criterion(logits, labels)
            
            # 总损失 = 分类损失 + 负载均衡损失
            total_loss = classification_loss + model.load_balance_weight * load_balance_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            preds = logits.argmax(dim=1)
            
            # 收集预测结果和真实标签，用于计算F1
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
            
            train_loss += classification_loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in val_dataloader:
                # 确保labels是长整型
                labels = labels.long()
                
                # 前向传播 - 推理模式
                logits = model(input_ids, attention_mask, return_loss=False)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                
                # 收集预测结果和真实标签，用于计算F1和混淆矩阵
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_acc = accuracy_score(val_labels_list, val_preds)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FIXED_CONFIG['patience']:
                break

        model.train()
    
    return {
        'best_f1': best_val_f1,
        'best_acc': best_val_acc,
        'best_epoch': best_epoch,
        'params': params
    }


def objective(trial):
    """Optuna目标函数"""
    
    # 定义搜索空间
    params = {
        'num_experts': trial.suggest_categorical('num_experts', [4, 8, 16, 32, 64]),
        'expert_mlp_layers': trial.suggest_int('expert_mlp_layers', 1, 4),
        'expert_hidden_size': trial.suggest_categorical('expert_hidden_size', [128, 256, 512, 768, 1024]),
        'load_balance_weight': trial.suggest_float('load_balance_weight', 0.001, 0.1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
        'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'seed': trial.suggest_int('seed', 42, 1000),
        'dropout': trial.suggest_float('dropout', 0.1, 0.4, log=True)
    }
    
    # top_k需要根据num_experts动态设置
    max_top_k = min(8, params['num_experts'])
    params['top_k'] = trial.suggest_int('top_k', 1, max_top_k)
    
    try:
        result = train_single_trial(params)
        
        # 记录中间结果
        trial.set_user_attr('best_acc', result['best_acc'])
        trial.set_user_attr('best_epoch', result['best_epoch'])
        trial.set_user_attr('params', result['params'])
        
        return result['best_f1']  # 优化目标为F1分数
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0  # 失败时返回最低分数


def run_bayesian_optimization():
    """运行贝叶斯优化"""
    
    # 创建结果保存目录
    os.makedirs(FIXED_CONFIG['checkpoint_dir'], exist_ok=True)
    
    # 创建study
    study = optuna.create_study(
        direction='maximize',  # 最大化F1分数
        study_name='single_task_deberta_moe_optimization',
        storage=f'sqlite:///{FIXED_CONFIG["checkpoint_dir"]}/optuna_study.db',
        load_if_exists=True
    )
    
    print("🚀 开始单任务DeBERTa MoE贝叶斯超参数优化...")
    print(f"任务: {FIXED_CONFIG['main_task']}")
    print(f"目标: 最大化验证集F1分数")
    print(f"总试验次数: 1000")
    print(f"结果保存至: {FIXED_CONFIG['checkpoint_dir']}")
    
    # 运行优化
    study.optimize(objective, n_trials=1000, timeout=None)
    
    # 获取最佳结果
    best_trial = study.best_trial
    best_params = best_trial.params
    best_f1 = best_trial.value
    best_acc = best_trial.user_attrs.get('best_acc', 0.0)
    best_epoch = best_trial.user_attrs.get('best_epoch', 0)
    
    # 保存结果
    results = {
        'task': FIXED_CONFIG['main_task'],
        'best_params': best_params,
        'best_f1': best_f1,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'best_trial_number': best_trial.number,
        'optimization_history': [
            {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'user_attrs': trial.user_attrs
            }
            for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存为JSON和pickle
    with open(f'{FIXED_CONFIG["checkpoint_dir"]}/best_results_{FIXED_CONFIG["main_task"]}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(f'{FIXED_CONFIG["checkpoint_dir"]}/best_results_{FIXED_CONFIG["main_task"]}.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 打印结果
    print("\n🎉 优化完成!")
    print("=" * 70)
    print(f"任务: {FIXED_CONFIG['main_task']}")
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"最佳准确率: {best_acc:.4f}")
    print(f"最佳epoch: {best_epoch}")
    print(f"最佳参数组合:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    # 保存最佳参数为单独文件便于使用
    with open(f'{FIXED_CONFIG["checkpoint_dir"]}/best_hyperparams_{FIXED_CONFIG["main_task"]}.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"详细结果已保存至: {FIXED_CONFIG['checkpoint_dir']}/best_results_{FIXED_CONFIG['main_task']}.json")
    print(f"最佳参数已保存至: {FIXED_CONFIG['checkpoint_dir']}/best_hyperparams_{FIXED_CONFIG['main_task']}.json")
    
    return study, results


if __name__ == '__main__':
    print(f"使用设备: {device}")
    print(f"优化任务: {FIXED_CONFIG['main_task']}")
    print("=" * 70)
    print("贝叶斯优化搜索空间:")
    print("  num_experts: [4, 8, 16, 32, 64]")
    print("  top_k: [1, min(8, num_experts)]")
    print("  expert_mlp_layers: [1, 2, 3, 4]")
    print("  expert_hidden_size: [128, 256, 512, 768, 1024]")
    print("  load_balance_weight: [0.001, 0.1] (log scale)")
    print("  batch_size: [4, 8, 16, 32]")
    print("  lr: [1e-6, 1e-3] (log scale)")
    print("  weight_decay: [1e-5, 1e-1] (log scale)")
    print("  seed: [42, 1000]")
    print("  dropout: [0.1, 0.4]")
    print("=" * 70)
    print()
    print("💡 提示: 如需切换任务，请修改FIXED_CONFIG中的'main_task'")
    print("   可选任务: 'Actionability', 'Mistake_Identification', 'Mistake_Location', 'Providing_Guidance'")
    print()
    
    # 运行优化
    study, results = run_bayesian_optimization()