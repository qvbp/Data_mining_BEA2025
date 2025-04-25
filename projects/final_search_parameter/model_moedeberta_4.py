import torch
import torch.nn as nn
import wandb
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.metrics import f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置全局变量，便于配置和修改
MODEL_PATHS = {
    'deberta': '/mnt/cfs/huangzhiwei/BAE2025/models/deberta-v3-base'
}

class BAE2025Dataset(Dataset):
    def __init__(
            self,
            data_path,
            label_type="Actionability",  # 只关注 "Actionability" 和 "Providing_Guidance"
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
            
            # 检查item中是否包含我们需要的标签
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
        tokenizer_name=MODEL_PATHS['deberta']
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.truncation_side = 'left'  # 从句子开头截断，保留句子结尾
        print(f"使用的tokenizer: {tokenizer_name}")
        
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


# 专家模型定义
class ExpertLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        output, (hidden, _) = self.lstm(x)
        return hidden[-1]  # [batch_size, hidden_size]


class ExpertBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,  # 双向LSTM，每个方向隐藏层大小减半
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        output, (hidden, _) = self.bilstm(x)
        # 拼接最后一层的正向和反向隐藏状态
        hidden_forward = hidden[-2]  # [batch_size, hidden_size//2]
        hidden_backward = hidden[-1]  # [batch_size, hidden_size//2]
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)  # [batch_size, hidden_size]
        return hidden_concat


class ExpertRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        _, hidden = self.rnn(x)
        return hidden[-1]  # [batch_size, hidden_size]


class ExpertGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        _, hidden = self.gru(x)
        return hidden[-1]  # [batch_size, hidden_size]


class ExpertLinear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, x):
        # 使用平均池化压缩序列信息
        pooled = torch.mean(x, dim=1)  # [batch_size, input_size]
        return self.linear(pooled)  # [batch_size, hidden_size]


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


class MoERouter(nn.Module):
    """专家路由器，学习为每个样本分配专家权重"""
    def __init__(self, input_size, num_experts):
        super().__init__()
        self.router = nn.Linear(input_size, num_experts)
        
    def forward(self, x):
        # 计算每个专家的权重
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1)
        return router_probs  # [batch_size, num_experts]


class DeBERTaMoEClassifier(nn.Module):
    def __init__(
        self, 
        pretrained_model_name, 
        num_classes=3, 
        freeze_layers=0,
        expert_hidden_size=256,
        dropout=0.3,
        num_rnn_layers=1
    ):
        super().__init__()
        
        # 使用 AutoModel 加载预训练模型
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # 获取 bert 隐藏层大小
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # 保留原有的分类头
        self.original_classifier = BertClassificationHead(
            hidden_size=self.bert_hidden_size,
            num_classes=num_classes,
            dropout_prob=dropout
        )
        
        # 创建多个专家模型
        self.experts = nn.ModuleDict({
            'lstm': ExpertLSTM(
                input_size=self.bert_hidden_size, 
                hidden_size=expert_hidden_size,
                num_layers=num_rnn_layers,
                dropout=dropout
            ),
            'bilstm': ExpertBiLSTM(
                input_size=self.bert_hidden_size, 
                hidden_size=expert_hidden_size,
                num_layers=num_rnn_layers,
                dropout=dropout
            ),
            'rnn': ExpertRNN(
                input_size=self.bert_hidden_size, 
                hidden_size=expert_hidden_size,
                num_layers=num_rnn_layers,
                dropout=dropout
            ),
            'gru': ExpertGRU(
                input_size=self.bert_hidden_size, 
                hidden_size=expert_hidden_size,
                num_layers=num_rnn_layers,
                dropout=dropout
            ),
            'linear': ExpertLinear(
                input_size=self.bert_hidden_size, 
                hidden_size=expert_hidden_size
            ),
        })
        
        # 创建路由器 (使用[CLS]标记表示作为路由的输入)
        self.router = MoERouter(self.bert_hidden_size, len(self.experts))
        
        # 各专家模型的输出映射层
        self.expert_outputs = nn.ModuleDict({
            expert_name: nn.Linear(expert_hidden_size, num_classes)
            for expert_name in self.experts.keys()
        })
        
        # 最终的融合层
        combined_dim = num_classes * (1 + len(self.experts))
        self.final_classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(combined_dim // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 获取序列隐藏状态
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 获取原始分类头结果
        original_logits = self.original_classifier(hidden_states)  # [batch_size, num_classes]
        
        # 获取路由权重
        cls_embedding = hidden_states[:, 0]  # [batch_size, hidden_size]
        routing_weights = self.router(cls_embedding)  # [batch_size, num_experts]
        
        # 获取各专家结果
        expert_outputs = {}
        for expert_name, expert in self.experts.items():
            # 获取专家输出
            expert_hidden = expert(hidden_states)  # [batch_size, expert_hidden_size]
            # 映射到类别空间
            expert_logits = self.expert_outputs[expert_name](expert_hidden)  # [batch_size, num_classes]
            # 存储结果
            expert_outputs[expert_name] = expert_logits
        
        # 拼接所有结果
        expert_logits_list = [original_logits]  # 包含原始分类头
        expert_names = list(self.experts.keys())
        
        for expert_name in expert_names:
            expert_logits_list.append(expert_outputs[expert_name])
        
        # 拼接所有结果 [batch_size, (1+num_experts)*num_classes]
        combined_logits = torch.cat(expert_logits_list, dim=1)
        
        # 通过最终分类器输出最终结果
        final_logits = self.final_classifier(combined_logits)
        
        return final_logits


def train_model(configs):
    # 设置随机种子以确保结果可复现
    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 初始化wandb - 只在非sweep模式下初始化
    if configs.use_wandb and not configs.is_sweep:
        wandb.init(
            project="BAE2025_deberta_track4",
            name=configs.exp_name,
            config={
                "model_name": configs.model_name,
                "num_classes": configs.num_classes,
                "dropout": configs.dropout,
                "batch_size": configs.batch_size,
                "max_length": configs.max_length,
                "lr": configs.lr,
                "epochs": configs.epochs,
                "seed": configs.seed,
                "expert_hidden_size": configs.expert_hidden_size,
                "num_rnn_layers": configs.num_rnn_layers,
                "weight_decay": configs.weight_decay,
                "label_type": configs.label_type  # 只关注这两个标签类型
            }
        )
    
    # 加载数据集
    train_dataset = BAE2025Dataset(configs.data_path, label_type=configs.label_type)
    val_dataset = BAE2025Dataset(configs.val_data_path, label_type=configs.label_type)    

    print(f"训练集数量: {len(train_dataset)}, 验证集数量: {len(val_dataset)}")
    print(f"标签类型: {configs.label_type}, 模型: {configs.model_name}")
    
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
    
    # 创建模型
    model = DeBERTaMoEClassifier(
        pretrained_model_name=configs.model_name,
        num_classes=configs.num_classes,
        freeze_layers=0,  # 不冻结任何层
        num_rnn_layers=configs.num_rnn_layers,
        expert_hidden_size=configs.expert_hidden_size,
        dropout=configs.dropout
    ).to(configs.device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器，添加weight_decay
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configs.lr,
        weight_decay=configs.weight_decay
    )
    
    # 不使用学习率调度策略
    
    # 初始化最佳指标
    best_metrics = {
        'epoch': 0,
        'val_f1': 0.0,
        'val_acc': 0.0,
        'val_loss': float('inf'),
        'params': {  # 保存参数配置
            'model_name': configs.model_name,
            'dropout': configs.dropout,
            'batch_size': configs.batch_size,
            'lr': configs.lr,
            'expert_hidden_size': configs.expert_hidden_size,
            'num_rnn_layers': configs.num_rnn_layers,
            'weight_decay': configs.weight_decay,
            'label_type': configs.label_type
        }
    }
    
    # 早停计数器
    patience_counter = 0
    
    # 定义类别名称
    class_names = list(train_dataset.get_labels().keys())
    
    # 训练循环
    for epoch in range(configs.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_preds = []
        train_labels_list = []
        
        with tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f'Epoch {epoch + 1}/{configs.epochs}',
            unit='batch',
            ncols=100
        ) as pbar:
            for input_ids, attention_mask, labels in pbar:
                optimizer.zero_grad()
                
                # 前向传播
                logits = model(input_ids, attention_mask)
                
                # 计算损失 - 确保labels是长整型
                labels = labels.long()
                loss = criterion(logits, labels)
                # 反向传播
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                preds = logits.argmax(dim=1)
                accuracy = (preds == labels).float().mean()
                accuracy_all = (preds == labels).float().sum()
                
                # 收集预测结果和真实标签，用于计算F1
                train_preds.extend(preds.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())
                
                train_loss += loss.item()
                train_acc += accuracy_all.item()
                
                pbar.set_postfix(
                    loss=f'{loss.item():.3f}',
                    accuracy=f'{accuracy.item():.3f}'
                )
        
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataset)
        
        # 计算训练集的F1分数
        train_f1 = f1_score(train_labels_list, train_preds, average='macro')
        
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.4f}')
        print(f'Training F1 Score: {train_f1:.4f}')
        
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
                
                # 前向传播
                logits = model(input_ids, attention_mask)
                
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
        
        print('Validation Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(val_loss, val_acc, val_f1))
        
        # 生成分类报告
        cls_report = classification_report(
            val_labels_list, 
            val_preds, 
            target_names=class_names, 
            digits=4
        )
        print(f"分类报告:\n{cls_report}")

        # 更新最佳指标
        if val_f1 > best_metrics['val_f1']:
            best_metrics.update({
                'epoch': epoch + 1,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_loss': val_loss
            })
            
            print(f'新的最佳性能: F1: {val_f1:.4f}, Acc: {val_acc:.4f}, Epoch: {epoch+1}')
            
            # 重置早停计数器
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'早停计数器: {patience_counter}/{configs.patience}')
            if patience_counter >= configs.patience:
                print(f'触发早停，在第{epoch+1}轮停止训练')
                break

        # 记录当前epoch的指标到wandb
        if configs.use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/f1": train_f1,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1": val_f1,
                "learning_rate": configs.lr,
                "epoch": epoch + 1
            })
    
    # 训练结束，记录最佳指标
    if configs.use_wandb:
        wandb.log({
            "best/epoch": best_metrics['epoch'],
            "best/val_f1": best_metrics['val_f1'],
            "best/val_acc": best_metrics['val_acc'],
            "best/val_loss": best_metrics['val_loss']
        })
        # 只在非sweep模式下结束wandb
        if not configs.is_sweep:
            wandb.finish()
    
    # 打印最终最佳结果
    print(f"训练结束! 最佳性能参数:")
    print(f"Model: {configs.model_name}")
    print(f"Label Type: {configs.label_type}")
    print(f"最佳Epoch: {best_metrics['epoch']}")
    print(f"F1: {best_metrics['val_f1']:.4f}")
    print(f"Accuracy: {best_metrics['val_acc']:.4f}")
    print(f"其他参数: dropout={configs.dropout}, lr={configs.lr}, batch_size={configs.batch_size}")
    print(f"expert_hidden_size={configs.expert_hidden_size}, num_rnn_layers={configs.num_rnn_layers}")
    print(f"weight_decay={configs.weight_decay}")
    
    # 将最佳结果添加到全局结果字典中
    model_type = 'deberta' if 'deberta' in configs.model_name.lower() else 'roberta'
    if model_type not in global_best_results:
        global_best_results[model_type] = {}
    
    if configs.label_type not in global_best_results[model_type]:
        global_best_results[model_type][configs.label_type] = {
            'best_epoch': best_metrics['epoch'],
            'best_val_f1': best_metrics['val_f1'],
            'params': {
                'dropout': configs.dropout,
                'batch_size': configs.batch_size,
                'lr': configs.lr,
                'expert_hidden_size': configs.expert_hidden_size,
                'num_rnn_layers': configs.num_rnn_layers,
                'weight_decay': configs.weight_decay
            }
        }
    elif best_metrics['val_f1'] > global_best_results[model_type][configs.label_type]['best_val_f1']:
        global_best_results[model_type][configs.label_type] = {
            'best_epoch': best_metrics['epoch'],
            'best_val_f1': best_metrics['val_f1'],
            'params': {
                'dropout': configs.dropout,
                'batch_size': configs.batch_size,
                'lr': configs.lr,
                'expert_hidden_size': configs.expert_hidden_size,
                'num_rnn_layers': configs.num_rnn_layers,
                'weight_decay': configs.weight_decay
            }
        }
    
    return best_metrics


def sweep_train():
    """用于wandb超参数搜索的训练函数"""
    # 初始化wandb - 在sweep模式下只需要在这里初始化一次
    wandb.init(project="BAE2025_deberta_track4")
    
    # 从wandb配置中获取超参数
    config = wandb.config
    
    # 创建配置对象
    class Args:
        def __init__(self):
            # 基本配置
            self.model_name = config.model_name
            self.num_classes = 3
            self.label_type = config.label_type
            
            # 模型配置
            self.dropout = config.dropout
            self.expert_hidden_size = config.expert_hidden_size
            self.num_rnn_layers = config.num_rnn_layers
            
            # 训练配置
            self.batch_size = config.batch_size
            self.max_length = 512
            self.lr = config.lr
            self.weight_decay = config.weight_decay
            self.epochs = 50  # 设置最大轮数
            self.patience = config.patience  # 早停轮数
            
            # 其他配置
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.seed = 42
            self.data_path = '../data_new/train.json'
            self.val_data_path = '../data_new/val.json'
            self.use_wandb = True
            self.is_sweep = True  # 标记这是sweep模式
            self.exp_name = f'sweep_{wandb.run.id}'
    
    configs = Args()
    
    # 调用训练函数
    metrics = train_model(configs)
    
    # 记录最佳结果和参数
    best_results = {
        'best_epoch': metrics['epoch'],
        'best_val_f1': metrics['val_f1'],
        'best_val_acc': metrics['val_acc'],
        'model_name': configs.model_name,
        'label_type': configs.label_type,
        'dropout': configs.dropout,
        'batch_size': configs.batch_size,
        'lr': configs.lr,
        'expert_hidden_size': configs.expert_hidden_size,
        'num_rnn_layers': configs.num_rnn_layers,
        'weight_decay': configs.weight_decay
    }
    
    # 记录到wandb
    # wandb.log(best_results)
    
    return metrics


# 定义全局变量保存最佳结果
global_best_results = {}

def main():
    import os
    # os.environ['WANDB_AGENT_DISABLE_FLAPPING'] = 'true'  # 禁用 flapping detection
    """主函数，定义和启动wandb超参数搜索"""
    # 初始化全局变量
    global global_best_results
    global_best_results = {}
    
    # 你的wandb key
    # wandb.login(key='17cd1653d2b9e39af387d772100677df5279d1cb')
    
    # 定义sweep配置
    sweep_config = {
        'method': 'bayes',  # 使用贝叶斯优化
        'metric': {
            'name': 'best/val_f1',
            'goal': 'maximize'   
        },
        'parameters': {
            'model_name': {
                'values': [
                    '/mnt/cfs/huangzhiwei/BAE2025/models/deberta-v3-base',
                ]
            },
            'label_type': {
                'values': [
                    'Actionability'
                ]
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]
            },
            'batch_size': {
                'values': [8, 16, 32]
            },
            'lr': {
                'values': [1e-5, 2e-5, 3e-5, 5e-5]
            },
            'expert_hidden_size': {
                'values': [256, 512, 768]
            },
            'num_rnn_layers': {
                'values': [1, 2, 3]
            },
            'weight_decay': {
                'values': [0.0, 0.01, 0.1]
            },
            'patience': {
                'values': [6]
            }
        }
    }

    # 创建sweep
    sweep_id = wandb.sweep(sweep_config, project="BAE2025_deberta_track4")
    
    # 运行sweep
    wandb.agent(sweep_id, function=sweep_train, count=30)  # 调整count以控制实验次数
    
    # 打印和保存结果
    print("全局最佳结果:")
    for model_type in global_best_results:
        print(f"\n{model_type.upper()}:")
        for label_type in global_best_results[model_type]:
            print(f"  {label_type}:")
            print(f"    最佳epoch: {global_best_results[model_type][label_type]['best_epoch']}")
            print(f"    最佳F1: {global_best_results[model_type][label_type]['best_val_f1']:.4f}")
            print(f"    最佳参数: {global_best_results[model_type][label_type]['params']}")
    
    # 保存到JSON文件
    import json
    with open('/mnt/cfs/huangzhiwei/BAE2025/projects/tiaocan/best_model_params_deberta_track4.json', 'w') as f:
        json.dump(global_best_results, f, indent=2)

if __name__ == '__main__':
    # 运行参数搜索
    main()
    
    # 或者运行单个实验
    # run_single_experiment()