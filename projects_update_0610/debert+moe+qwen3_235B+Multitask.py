import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AdamW, AutoConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BAE2025Dataset(Dataset):
    def __init__(
            self,
            data_path,
            label_type="Actionability",  # 根据需要可以是 "Mistake_Identification", "Mistake_Location", "Providing_Guidance", "Actionability"
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
        print("当前使用的 tokenizer 类型：", type(self.tokenizer))
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.drop_last = drop_last

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
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
        self.top_k = top_k
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
            top_k=top_k
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


def argparser():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='DeBERTa MoE Training')
    
    # 模型相关参数
    parser.add_argument('--model_name', type=str, 
                       default='/mnt/cfs/huangzhiwei/BAE2025/models/deberta-v3-base',
                       help='预训练模型路径')
    parser.add_argument('--num_classes', type=int, default=3, help='分类类别数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--freeze_pooler', type=int, default=0, help='是否冻结pooler层')
    
    # MoE相关参数
    parser.add_argument('--num_experts', type=int, default=16, help='专家数量')
    parser.add_argument('--top_k', type=int, default=4, help='选择的专家数量')
    parser.add_argument('--expert_mlp_layers', type=int, default=2, help='专家MLP层数')
    parser.add_argument('--expert_hidden_size', type=int, default=768, help='专家隐藏层大小')
    parser.add_argument('--load_balance_weight', type=float, default=0.01, help='负载均衡损失权重')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--lr', type=float, default=3e-5, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--patience', type=int, default=6, help='早停耐心值')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='预热比例')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, default='../data_new/all.json', help='训练数据路径')
    parser.add_argument('--val_data_path', type=str, default='../data_new/val.json', help='验证数据路径')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default=None, help='设备')
    parser.add_argument('--name', type=str, default=None, help='实验名称')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_track4', help='检查点保存目录')
    
    return parser.parse_args()





def train(configs):
    # 设置随机种子
    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(configs.checkpoint_dir, configs.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 为保存混淆矩阵创建目录 - 分别为训练集和验证集创建
    train_plot_dir = os.path.join(checkpoint_dir, 'plots', 'train')
    val_plot_dir = os.path.join(checkpoint_dir, 'plots', 'val')
    os.makedirs(train_plot_dir, exist_ok=True)
    os.makedirs(val_plot_dir, exist_ok=True)
    
    # 加载数据集
    train_dataset = BAE2025Dataset(configs.data_path)
    val_dataset = BAE2025Dataset(configs.val_data_path)    

    # 创建数据加载器
    train_dataloader = BAE2025DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size,
        max_length=configs.max_length,
        shuffle=True,
        drop_last=True,
        device=configs.device,
        tokenizer_name=configs.model_name
    )

    val_dataloader = BAE2025DataLoader(
        dataset=val_dataset,
        batch_size=configs.batch_size,
        max_length=configs.max_length,
        shuffle=False,
        drop_last=False,
        device=configs.device,
        tokenizer_name=configs.model_name
    )
    
    # 创建MoE模型
    model = DeBERTaMoEClassifier(
        pretrained_model_name=configs.model_name,
        num_classes=configs.num_classes,
        freeze_pooler=configs.freeze_pooler,
        expert_hidden_size=configs.expert_hidden_size,
        dropout=configs.dropout,
        num_experts=configs.num_experts,
        top_k=configs.top_k,
        expert_mlp_layers=configs.expert_mlp_layers,
        load_balance_weight=configs.load_balance_weight
    ).to(configs.device)

    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configs.lr,
        weight_decay=configs.weight_decay
    )
    
    # 初始化最佳验证损失和早停计数器
    best_val_acc = 0.0
    best_val_f1 = 0.0  # 添加F1分数作为评估指标
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 初始化最佳指标记录
    best_metrics = {
        'epoch': 0,
        'val_f1': 0.0,
        'val_acc': 0.0,
        'val_loss': float('inf')
    }

    # 定义类别名称
    class_names = ['Yes', 'To some extent', 'No']
    
    print(f"开始训练MoE模型...")
    print(f"专家数量: {configs.num_experts}, Top-K: {configs.top_k}")
    print(f"负载均衡权重: {configs.load_balance_weight}")
    
    # 训练循环
    for epoch in range(configs.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_load_balance_loss = 0.0
        train_acc = 0.0
        train_preds = []
        train_labels_list = []
        
        with tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f'Epoch {epoch + 1}/{configs.epochs}',
            unit='batch',
            ncols=120
        ) as pbar:
            for input_ids, attention_mask, labels in pbar:
                optimizer.zero_grad()
                
                # 前向传播 - 获取MoE输出
                outputs = model(input_ids, attention_mask, return_loss=True)
                
                if isinstance(outputs, dict):
                    # 训练模式，包含负载均衡损失
                    logits = outputs['logits']
                    load_balance_loss = outputs['load_balance_loss']
                    expert_utilization = outputs['expert_utilization']
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
                accuracy = (preds == labels).float().mean()
                accuracy_all = (preds == labels).float().sum()
                
                # 收集预测结果和真实标签，用于计算F1
                train_preds.extend(preds.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())
                
                train_loss += classification_loss.item()
                train_load_balance_loss += load_balance_loss.item()
                train_acc += accuracy_all.item()
                
                # 更新进度条显示
                pbar.set_postfix(
                    cls_loss=f'{classification_loss.item():.3f}',
                    lb_loss=f'{load_balance_loss.item():.4f}',
                    acc=f'{accuracy.item():.3f}'
                )
        
        train_loss = train_loss / len(train_dataloader)
        train_load_balance_loss = train_load_balance_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataset)
        
        # 计算训练集的F1分数 - 使用macro平均以处理多分类
        train_f1 = f1_score(train_labels_list, train_preds, average='macro')
        
        print(f'Training - Classification Loss: {train_loss:.4f}, Load Balance Loss: {train_load_balance_loss:.4f}')
        print(f'Training - Accuracy: {train_acc:.4f}, F1 Score: {train_f1:.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0.0
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
                accuracy = (preds == labels).float().sum()
                val_corrects += accuracy
                
                # 收集预测结果和真实标签，用于计算F1和混淆矩阵
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_dataloader)
        val_acc = val_corrects.double() / len(val_dataset)
        
        # 计算验证集的F1分数
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        
        print('Validation - Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}'.format(val_loss, val_acc, val_f1))

        # 分析专家使用情况（可选，仅在某些epoch执行以节省时间）
        if epoch % 5 == 0:  # 每5个epoch分析一次
            # 使用验证集的一个batch来分析专家使用
            sample_input_ids, sample_attention_mask, _ = next(iter(val_dataloader))
            expert_stats = model.get_expert_utilization(sample_input_ids, sample_attention_mask)
            expert_counts = expert_stats['expert_counts']
            expert_utilization = expert_stats['expert_utilization']
            
            print(f"专家使用统计 (Epoch {epoch+1}):")
            print(f"  专家选择次数: {expert_counts.numpy()}")
            print(f"  专家平均利用率: {expert_utilization.cpu().numpy()}")
            print(f"  利用率标准差: {torch.std(expert_utilization).item():.4f}")
        
        # 检查是否保存模型并判断是否需要早停
        # 使用F1分数作为主要指标
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            
            # 保存最佳模型
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(checkpoint_dir, 'best_model_f1.pt'))
            print(f'🎉 New best model saved! F1: {best_val_f1:.4f}, Acc: {best_val_acc:.4f}')
            
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= configs.patience:
                print(f'🛑 Early stopping triggered after {epoch+1} epochs.')
                break

        model.train()

    print("训练完成!")
    print(f"最佳结果 - F1: {best_val_f1:.4f}, Acc: {best_val_acc:.4f}, Loss: {best_val_loss:.4f}")


# 主函数
if __name__ == '__main__':
    # 使用argparse解析命令行参数
    configs = argparser()
    
    # 设置实验名称
    if configs.name is None:
        configs.exp_name = \
            f'{os.path.basename(configs.model_name)}' + \
            f'{"_fp" if configs.freeze_pooler else ""}' + \
            f'_moe{configs.num_experts}k{configs.top_k}' + \
            f'_b{configs.batch_size}_e{configs.epochs}' + \
            f'_len{configs.max_length}_lr{configs.lr}'
    else:
        configs.exp_name = configs.name
    
    # 设置设备
    if configs.device is None:
        configs.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    print(f"使用设备: {configs.device}")
    print(f"实验名称: {configs.exp_name}")
    
    # 调用训练函数
    train(configs)