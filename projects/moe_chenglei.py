import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict

# 配置参数
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理类
class TutorDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.texts = []
        self.labels = {
            'mistake_id': [],
            'mistake_loc': [],
            'guidance': [],
            'action': [],
            'tutor_id': []
        }
        self.tutor_ids = {}
        self.label_map = {'No': 0, 'To some extent': 1, 'Yes': 2}
        
        # 处理数据
        for item in data:
            for tutor, resp in item['tutor_responses'].items():
                text = f"{item['conversation_history']} [SEP] {resp['response']}"
                self.texts.append(text)
                
                # 处理三分类标签
                self.labels['mistake_id'].append(self.label_map[resp['annotation']['Mistake_Identification']])
                self.labels['mistake_loc'].append(self.label_map[resp['annotation']['Mistake_Location']])
                self.labels['guidance'].append(self.label_map[resp['annotation']['Providing_Guidance']])
                self.labels['action'].append(self.label_map[resp['annotation']['Actionability']])
                
                # 处理导师ID
                if tutor not in self.tutor_ids:
                    self.tutor_ids[tutor] = len(self.tutor_ids)
                self.labels['tutor_id'].append(self.tutor_ids[tutor])
        
        self.tokenizer = tokenizer
        # self._calculate_class_weights()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        #  inputs = self.tokenizer(
        #     text=stu_ans,
        #     text_pair=true_ans,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=512,
        #     add_special_tokens=True,
        # ).to(self.device)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'mistake_id': torch.tensor(self.labels['mistake_id'][idx], dtype=torch.long),
            'mistake_loc': torch.tensor(self.labels['mistake_loc'][idx], dtype=torch.long),
            'guidance': torch.tensor(self.labels['guidance'][idx], dtype=torch.long),
            'action': torch.tensor(self.labels['action'][idx], dtype=torch.long),
            'tutor_id': torch.tensor(self.labels['tutor_id'][idx], dtype=torch.long)
        }

    # def _calculate_class_weights(self):
    #     """计算类别权重用于处理不平衡数据"""
    #     self.class_weights = {}
    #     for task in ['mistake_id', 'mistake_loc', 'guidance', 'action', 'tutor_id']:
    #         counts = np.bincount(self.labels[task])
    #         weights = 1. / (counts + 1e-6)  # 添加小值防止除零
    #         weights = weights / weights.sum() * len(weights)  # 归一化
    #         self.class_weights[task] = torch.FloatTensor(weights)

# # 多任务模型
# class TutorModel(torch.nn.Module):
#     def __init__(self, num_tutors):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained('deberta-v3-base')
#         self.dropout = torch.nn.Dropout(0.1)
        
        
#         self.mistake_id = torch.nn.Linear(768, 2)
#         self.mistake_loc = torch.nn.Linear(768, 2)
#         self.guidance = torch.nn.Linear(768, 2)
#         self.action = torch.nn.Linear(768, 2)
        
#         # 导师分类层
#         self.tutor_id = torch.nn.Linear(768, 9)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.last_hidden_state[:, -1, :]
#         pooled_output = self.dropout(pooled_output)
        
#         return {
#             'mistake_id': self.mistake_id(pooled_output),
#             'mistake_loc': self.mistake_loc(pooled_output),
#             'guidance': self.guidance(pooled_output),
#             'action': self.action(pooled_output),
#             'tutor_id': self.tutor_id(pooled_output)
#         }


# class TutorModel(torch.nn.Module):
#     def __init__(self, num_tutors):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained('deberta-v3-base')
        
#         # 共享的特征提取层
#         self.shared_dropout = torch.nn.Dropout(0.2)
#         self.shared_layer_norm = torch.nn.LayerNorm(768)
        
#         # 为每个任务创建独立的特征提取层
#         self.mistake_id_head = self._create_task_head(768, 2)
#         self.mistake_loc_head = self._create_task_head(768, 2)
#         self.guidance_head = self._create_task_head(768, 2)
#         self.action_head = self._create_task_head(768, 2)
#         self.tutor_id_head = self._create_task_head(768, num_tutors)
        
#     def _create_task_head(self, input_dim, output_dim):
#         return torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 384),
#             torch.nn.GELU(),
#             torch.nn.LayerNorm(384),
#             torch.nn.Dropout(0.1),
#             torch.nn.Linear(384, output_dim)
#         )
    
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids, attention_mask=attention_mask)
        
#         # 获取[CLS]位置的隐藏状态
#         pooled_output = outputs.last_hidden_state[:, 0, :]
        
#         # 共享处理
#         pooled_output = self.shared_layer_norm(pooled_output)
#         pooled_output = self.shared_dropout(pooled_output)
        
#         # 各任务独立处理
#         return {
#             'mistake_id': self.mistake_id_head(pooled_output),
#             'mistake_loc': self.mistake_loc_head(pooled_output),
#             'guidance': self.guidance_head(pooled_output),
#             'action': self.action_head(pooled_output),
#             'tutor_id': self.tutor_id_head(pooled_output)
#         }
class TutorModel(torch.nn.Module):
    def __init__(self, num_tutors, num_experts=4, expert_capacity=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained('/mnt/cfs/huangzhiwei/BAE2025/models/deberta-v3-base')
        
        # MoE层参数
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # 共享的MoE层
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(768, expert_capacity),
                torch.nn.GELU(),
                torch.nn.LayerNorm(expert_capacity),
                torch.nn.Linear(expert_capacity, 768)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = torch.nn.Linear(768, num_experts)
        
        # 任务特定头部
        self.mistake_id = torch.nn.Linear(768, 3)
        self.mistake_loc = torch.nn.Linear(768, 3)
        self.guidance = torch.nn.Linear(768, 3)
        self.action = torch.nn.Linear(768, 3)
        self.tutor_id = torch.nn.Linear(768, num_tutors)
        
        # 共享的dropout和归一化
        self.dropout = torch.nn.Dropout(0.1)
        self.layer_norm = torch.nn.LayerNorm(768)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 使用[CLS] token
        
        # MoE处理
        gate_scores = torch.softmax(self.gate(pooled_output), dim=-1)
        expert_outputs = []
        
        for expert in self.experts:
            expert_outputs.append(expert(pooled_output))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, hidden_size]
        moe_output = torch.sum(gate_scores.unsqueeze(-1) * expert_outputs, dim=1)
        
        # 共享处理
        moe_output = self.layer_norm(moe_output)
        moe_output = self.dropout(moe_output)
        
        # 任务特定输出
        return {
            'mistake_id': self.mistake_id(moe_output),
            'mistake_loc': self.mistake_loc(moe_output),
            'guidance': self.guidance(moe_output),
            'action': self.action(moe_output),
            'tutor_id': self.tutor_id(moe_output)
        }
# 训练函数
def train_epoch(model, dataloader, optimizer, criterion, class_weights=None):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        inputs = {
            'input_ids': batch['input_ids'].to(DEVICE),
            'attention_mask': batch['attention_mask'].to(DEVICE)
        }
        
        outputs = model(**inputs)
        
        # 计算多任务损失
        losses = []
        for task in ['mistake_id', 'mistake_loc', 'guidance', 'action']:
            # weight = class_weights[task].to(DEVICE)
            loss = criterion[task](
                outputs[task], 
                batch[task].to(DEVICE)
            )
            
            losses.append(loss)
        
        # 导师分类损失
        tutor_loss = criterion['tutor_id'](
            outputs['tutor_id'], 
            batch['tutor_id'].to(DEVICE)
        )
        #losses.append(tutor_loss)
        
        total_loss = sum(losses)/4.0
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    return total_loss.item()

# 评估函数
def evaluate(model, dataloader):
    model.eval()
    predictions = defaultdict(list)
    true_labels = defaultdict(list)
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE)
            }
            
            outputs = model(**inputs)
            
            # 收集预测结果
            for task in ['mistake_id', 'mistake_loc', 'guidance', 'action']:
                preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                predictions[task].extend(preds)
                true_labels[task].extend(batch[task].numpy())
            
            # 导师身份预测
            tutor_preds = torch.argmax(outputs['tutor_id'], dim=1).cpu().numpy()
            predictions['tutor_id'].extend(tutor_preds)
            true_labels['tutor_id'].extend(batch['tutor_id'].numpy())
    
    # 计算指标
    metrics = {}
    for task in predictions:
        if task == 'tutor_id':
            metrics[task] = {
                'acc': accuracy_score(true_labels[task], predictions[task]),
                'f1': f1_score(true_labels[task], predictions[task], 
                              average='weighted', zero_division=0),
               
            }
        else:
            metrics[task] = {
                'acc': accuracy_score(true_labels[task], predictions[task]),
                'f1': f1_score(true_labels[task], predictions[task], 
                              average='weighted', zero_division=0),
              
            }
    
    return metrics, predictions, true_labels

# 主函数
def main():
    # 加载数据
    with open('/mnt/cfs/huangzhiwei/BAE2025/data/mrbench_v3_devset.json') as f:
        data = json.load(f)
    
    # 划分数据集
    # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    # print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    # print(type(train_data))

    # import sys
    # sys.exit(0)
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/mnt/cfs/huangzhiwei/BAE2025/models/deberta-v3-base')
    
    # # 创建数据集
    # with open('/mnt/cfs/huangzhiwei/BAE2025/data/train.json', 'r', encoding='utf-8') as f:
    #     train_data = json.load(f)
    # with open('/mnt/cfs/huangzhiwei/BAE2025/data/valid.json', 'r', encoding='utf-8') as f:
    #     test_data = json.load(f)
    train_dataset = TutorDataset(data, tokenizer)
    test_dataset = TutorDataset(data, tokenizer)
    
    # 标签映射
    id2label = {0: 'No', 1: 'To some extent', 2: 'Yes'}
    print(len(train_dataset.tutor_ids))
    # 创建模型
    model = TutorModel(num_tutors=len(train_dataset.tutor_ids)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 损失函数
    criterion = {
        'mistake_id': torch.nn.CrossEntropyLoss(),
        'mistake_loc': torch.nn.CrossEntropyLoss(),
        'guidance': torch.nn.CrossEntropyLoss(),
        'action': torch.nn.CrossEntropyLoss(),
        'tutor_id': torch.nn.CrossEntropyLoss()
    }
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False)
    
    # 训练循环
    for epoch in range(EPOCHS):
        loss = train_epoch(
            model, train_loader, optimizer, criterion,
            # train_dataset.class_weights
        )
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}')
        
        # 每个epoch后评估
        metrics, _, _ = evaluate(model, test_loader)
        print(f"\nValidation Metrics (Epoch {epoch+1}):")
        for task, scores in metrics.items():
            print(f"{task.upper()} - ACC: {scores['acc']:.4f}, F1: {scores['f1']:.4f}")

        if epoch >=4:
            # 保存模型
            torch.save(model.state_dict(), f'/mnt/cfs/huangzhiwei/BAE2025/projects/predict/chenlei_moe/tutor_model_{epoch+1}.pt')
            print("\nModel saved to tutor_model.pt")

    
    # 最终评估
    print("\nFinal Evaluation Results:")
    metrics, predictions, true_labels = evaluate(model, test_loader)
    for task, scores in metrics.items():
        print(f"\n{task.upper()} Performance:")
        print(f"Accuracy: {scores['acc']:.4f}")
        print(f"F1 Score: {scores['f1']:.4f}")
        print("\nClassification Report:")
    
    
    # # 保存模型
    # torch.save(model.state_dict(), '/mnt/cfs/huangzhiwei/BAE2025/projects/predict/tutor_model.pt')
    # print("\nModel saved to tutor_model.pt")

if __name__ == '__main__':
    main()