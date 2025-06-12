"""
DeBERTa MoE多任务学习 - 贝叶斯调参版本

使用Optuna进行超参数优化，自动搜索最佳参数组合
调参参数: num_experts, top_k, expert_mlp_layers, expert_hidden_size, 
         load_balance_weight, batch_size, lr, weight_decay, seed

运行方法:
python train_bayesian.py
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
    'multi_task': True,
    'main_task': 'Providing_Guidance',
    'max_length': 512,
    'epochs': 40,  # 调参时减少epoch数以节省时间
    'patience': 8,
    'warmup_ratio': 0.1,
    'data_path': '/mnt/cfs/huangzhiwei/Data_mining_BAE2025/data_new/train.json',
    'val_data_path': '/mnt/cfs/huangzhiwei/Data_mining_BAE2025/data_new/val.json',
    'checkpoint_dir': '/mnt/cfs/huangzhiwei/Data_mining_BAE2025/projects_update_0610/tiaocan/bayesian_optimization_results_moe+Multitask_x',
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
            },
            multi_task=False
    ):
        self.data_path = data_path
        self.label_type = label_type
        self.labels = labels
        self.multi_task = multi_task
        self.task_names = ["Actionability", "Mistake_Identification", "Mistake_Location", "Providing_Guidance"]
        self._get_data()
    
    def _get_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.data = []
        for item in data:
            if 'conversation_history' in item and 'response' in item:
                sent1 = item['conversation_history']
                sent2 = item['response']
                
                if self.multi_task:
                    all_labels = {}
                    valid_sample = True
                    
                    for task_name in self.task_names:
                        if (task_name in item and 
                            item[task_name] in self.labels):
                            all_labels[task_name] = self.labels[item[task_name]]
                        else:
                            valid_sample = False
                            break
                    
                    if valid_sample:
                        self.data.append(((sent1, sent2), all_labels))
                else:
                    if (self.label_type in item and 
                        item[self.label_type] in self.labels):
                        self.data.append(((sent1, sent2), self.labels[item[self.label_type]]))
    
    def __len__(self):
        return len(self.data)
    
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
        tokenizer_name='/mnt/cfs/huangzhiwei/BAE2025/models/deberta-v3-base',
        multi_task=False
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.truncation_side = 'left'
        self.multi_task = multi_task
        self.task_names = ["Actionability", "Mistake_Identification", "Mistake_Location", "Providing_Guidance"]
        
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
        
        if self.multi_task:
            batch_labels = {}
            for task_name in self.task_names:
                task_labels = [label_dict[task_name] for label_dict in labels]
                batch_labels[task_name] = torch.LongTensor(task_labels).to(self.device)
            return input_ids, attention_mask, batch_labels
        else:
            labels = torch.LongTensor(labels).to(self.device)
            return input_ids, attention_mask, labels

    def __iter__(self):
        for data in self.loader:
            yield data

    def __len__(self):
        return len(self.loader)


class MLPExpert(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(nn.Linear(current_size, hidden_size))
            else:
                layers.append(nn.Linear(current_size, hidden_size))
                layers.append(nn.LayerNorm(hidden_size))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                current_size = hidden_size
                
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.mean(x, dim=1)
        return self.mlp(x)


class BertClassificationHead(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=3, dropout_prob=0.3):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_classes)
    
    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DynamicMoERouter(nn.Module):
    def __init__(self, input_size, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(input_size, num_experts)
        
    def forward(self, x):
        batch_size = x.size(0)
        router_logits = self.router(x)
        all_probs = F.softmax(router_logits, dim=-1)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        routing_weights = torch.zeros_like(router_logits)
        routing_weights.scatter_(1, top_k_indices, top_k_probs)
        return routing_weights, top_k_indices, all_probs


class DeBERTaMoEClassifier(nn.Module):
    def __init__(
        self, 
        pretrained_model_name, 
        num_classes=3, 
        freeze_pooler=0,
        expert_hidden_size=256,
        dropout=0.3,
        num_experts=16,
        top_k=4,
        expert_mlp_layers=2,
        load_balance_weight=0.01,
        multi_task=False,
        task_names=None,
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # 确保top_k不超过num_experts
        self.load_balance_weight = load_balance_weight
        self.multi_task = multi_task
        
        if task_names is None:
            self.task_names = ["Actionability", "Mistake_Identification", "Mistake_Location", "Providing_Guidance"]
        else:
            self.task_names = task_names
        
        self.original_classifier = BertClassificationHead(
            hidden_size=self.bert_hidden_size,
            num_classes=num_classes,
            dropout_prob=dropout
        )
        
        self.experts = nn.ModuleList([
            MLPExpert(
                input_size=self.bert_hidden_size,
                hidden_size=expert_hidden_size,
                num_layers=expert_mlp_layers,
                dropout=dropout
            ) for _ in range(num_experts)
        ])
        
        self.router = DynamicMoERouter(
            input_size=self.bert_hidden_size, 
            num_experts=num_experts,
            top_k=self.top_k
        )
        
        self.expert_output_proj = nn.Linear(expert_hidden_size, num_classes)
        
        if self.multi_task:
            self.task_classifiers = nn.ModuleDict({
                task_name: nn.Sequential(
                    nn.Linear(num_classes * 2, num_classes * 2 // 2),
                    nn.LayerNorm(num_classes * 2 // 2),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(num_classes * 2 // 2, num_classes)
                ) for task_name in self.task_names
            })
            
            self.task_log_vars = nn.Parameter(torch.zeros(len(self.task_names)))
        else:
            combined_dim = num_classes * 2
            self.final_classifier = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.LayerNorm(combined_dim // 2),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(combined_dim // 2, num_classes)
            )
    
    def compute_load_balance_loss(self, all_probs, top_k_indices):
        batch_size = all_probs.size(0)
        importance = torch.mean(all_probs, dim=0)
        selection_mask = torch.zeros_like(all_probs)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            selection_mask.scatter_(1, expert_indices.unsqueeze(1), 1.0)
        
        frequency = torch.mean(selection_mask, dim=0)
        auxiliary_loss = self.num_experts * torch.sum(frequency * importance)
        frequency_variance = torch.var(frequency)
        importance_variance = torch.var(importance)
        load_balance_loss = auxiliary_loss + frequency_variance + importance_variance
        return load_balance_loss
    
    def compute_multi_task_loss(self, task_logits, task_labels):
        task_losses = {}
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        for i, task_name in enumerate(self.task_names):
            if task_name in task_logits and task_name in task_labels:
                task_loss = criterion(task_logits[task_name], task_labels[task_name])
                task_losses[task_name] = task_loss
                precision = torch.exp(-self.task_log_vars[i])
                weighted_loss = precision * task_loss + self.task_log_vars[i]
                total_loss += weighted_loss
        
        return total_loss, task_losses
        
    def forward(self, input_ids, attention_mask, return_loss=True):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        batch_size = hidden_states.size(0)
        
        original_logits = self.original_classifier(hidden_states)
        cls_embedding = hidden_states[:, 0]
        routing_weights, top_k_indices, all_probs = self.router(cls_embedding)
        
        moe_output = torch.zeros(batch_size, self.expert_output_proj.out_features, device=hidden_states.device)
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            for batch_idx in range(batch_size):
                current_expert_idx = expert_idx[batch_idx].item()
                current_weight = routing_weights[batch_idx, current_expert_idx]
                expert_hidden = self.experts[current_expert_idx](hidden_states[batch_idx:batch_idx+1])
                expert_logits = self.expert_output_proj(expert_hidden)
                moe_output[batch_idx] += current_weight * expert_logits.squeeze(0)
        
        combined_logits = torch.cat([original_logits, moe_output], dim=1)
        
        if self.multi_task:
            task_logits = {}
            for task_name in self.task_names:
                task_logits[task_name] = self.task_classifiers[task_name](combined_logits)
            
            if return_loss and self.training:
                load_balance_loss = self.compute_load_balance_loss(all_probs, top_k_indices)
                return {
                    'task_logits': task_logits,
                    'load_balance_loss': load_balance_loss,
                    'routing_weights': routing_weights,
                    'expert_utilization': torch.mean(all_probs, dim=0),
                    'task_weights': torch.exp(-self.task_log_vars)
                }
            else:
                return task_logits
        else:
            final_logits = self.final_classifier(combined_logits)
            if return_loss and self.training:
                load_balance_loss = self.compute_load_balance_loss(all_probs, top_k_indices)
                return {
                    'logits': final_logits,
                    'load_balance_loss': load_balance_loss,
                    'routing_weights': routing_weights,
                    'expert_utilization': torch.mean(all_probs, dim=0)
                }
            else:
                return final_logits


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
        label_type=FIXED_CONFIG['main_task'],
        multi_task=FIXED_CONFIG['multi_task']
    )
    val_dataset = BAE2025Dataset(
        FIXED_CONFIG['val_data_path'], 
        label_type=FIXED_CONFIG['main_task'],
        multi_task=False
    )

    # 创建数据加载器
    train_dataloader = BAE2025DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        max_length=FIXED_CONFIG['max_length'],
        shuffle=True,
        drop_last=True,
        device=FIXED_CONFIG['device'],
        tokenizer_name=FIXED_CONFIG['model_name'],
        multi_task=FIXED_CONFIG['multi_task']
    )

    val_dataloader = BAE2025DataLoader(
        dataset=val_dataset,
        batch_size=params['batch_size'],
        max_length=FIXED_CONFIG['max_length'],
        shuffle=False,
        drop_last=False,
        device=FIXED_CONFIG['device'],
        tokenizer_name=FIXED_CONFIG['model_name'],
        multi_task=False
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
        load_balance_weight=params['load_balance_weight'],
        multi_task=FIXED_CONFIG['multi_task']
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
    task_names = ["Actionability", "Mistake_Identification", "Mistake_Location", "Providing_Guidance"]
    
    # 训练循环
    for epoch in range(FIXED_CONFIG['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels_list = []
        
        for batch_data in train_dataloader:
            optimizer.zero_grad()
            
            if FIXED_CONFIG['multi_task']:
                input_ids, attention_mask, labels_dict = batch_data
                outputs = model(input_ids, attention_mask, return_loss=True)
                task_logits = outputs['task_logits']
                load_balance_loss = outputs['load_balance_loss']
                
                multi_task_loss, individual_task_losses = model.compute_multi_task_loss(task_logits, labels_dict)
                total_loss = multi_task_loss + model.load_balance_weight * load_balance_loss
                
                # 收集主任务预测结果
                main_task_pred = task_logits[FIXED_CONFIG['main_task']].argmax(dim=1)
                main_task_label = labels_dict[FIXED_CONFIG['main_task']].long()
                train_preds.extend(main_task_pred.cpu().numpy())
                train_labels_list.extend(main_task_label.cpu().numpy())
                
                train_loss += multi_task_loss.item()
            else:
                input_ids, attention_mask, labels = batch_data
                outputs = model(input_ids, attention_mask, return_loss=True)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                    load_balance_loss = outputs['load_balance_loss']
                else:
                    logits = outputs
                    load_balance_loss = torch.tensor(0.0, device=logits.device)
                
                labels = labels.long()
                classification_loss = criterion(logits, labels)
                total_loss = classification_loss + model.load_balance_weight * load_balance_loss
                
                preds = logits.argmax(dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())
                train_loss += classification_loss.item()
            
            total_loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in val_dataloader:
                labels = labels.long()
                
                if FIXED_CONFIG['multi_task']:
                    task_logits = model(input_ids, attention_mask, return_loss=False)
                    logits = task_logits[FIXED_CONFIG['main_task']]
                else:
                    logits = model(input_ids, attention_mask, return_loss=False)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                
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
        'load_balance_weight': trial.suggest_float('load_balance_weight', 0.0001, 0.1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
        'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'seed': trial.suggest_categorical('seed', [42, 3407, 2025]),
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
        study_name='deberta_moe_optimization',
        storage=f'sqlite:///{FIXED_CONFIG["checkpoint_dir"]}/optuna_study.db',
        load_if_exists=True
    )
    
    print("🚀 开始贝叶斯超参数优化...")
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
    with open(f'{FIXED_CONFIG["checkpoint_dir"]}/best_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(f'{FIXED_CONFIG["checkpoint_dir"]}/best_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 打印结果
    print("\n🎉 优化完成!")
    print("=" * 60)
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"最佳准确率: {best_acc:.4f}")
    print(f"最佳epoch: {best_epoch}")
    print(f"最佳参数组合:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # 保存最佳参数为单独文件便于使用
    with open(f'{FIXED_CONFIG["checkpoint_dir"]}/best_hyperparams.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"详细结果已保存至: {FIXED_CONFIG['checkpoint_dir']}/best_results.json")
    print(f"最佳参数已保存至: {FIXED_CONFIG['checkpoint_dir']}/best_hyperparams.json")
    
    return study, results


if __name__ == '__main__':
    print(f"使用设备: {device}")
    print("=" * 60)
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
    print("=" * 60)
    
    # 运行优化
    study, results = run_bayesian_optimization()