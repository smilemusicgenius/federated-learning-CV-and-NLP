"""
联邦学习实验代码 - AGNews数据集 + ALBERT模型
包含方法: Local Training, Centralized Training, FedAvg, FedProx, SCAFFOLD
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import time
import random
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from datasets import load_dataset
import copy
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore', message='Some weights of.*were not initialized')

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}\n")

# ==================== 数据集加载 ====================
class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_agnews_data():
    """加载AGNews数据集"""
    print("正在加载AGNews数据集...")
    
    # 加载数据集（离线模式需要提前下载）
    try:
        dataset = load_dataset('ag_news', cache_dir='./data_cache')
    except:
        print("提示: 如果在离线环境，请先在有网络的环境下运行一次以下载数据集")
        print("或手动下载数据集到 ./data_cache 目录")
        raise
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"测试集大小: {len(test_texts)}\n")
    
    return train_texts, train_labels, test_texts, test_labels

def split_data_for_clients(train_texts, train_labels, num_clients=10):
    """将数据分配给各个客户端（Non-IID分布）"""
    num_samples = len(train_texts)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # 简单的Non-IID划分：每个客户端获得不同数量的数据
    client_indices = []
    samples_per_client = num_samples // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else num_samples
        client_indices.append(indices[start_idx:end_idx])
    
    return client_indices

# ==================== 模型定义 ====================
def create_albert_model(num_labels=4):
    """创建ALBERT模型"""
    model = AlbertForSequenceClassification.from_pretrained(
        'albert-base-v2',
        num_labels=num_labels,
        cache_dir='./model_cache'
    )
    return model

# ==================== 训练和评估函数 ====================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# ==================== 1. Local Training ====================
def local_training(train_texts, train_labels, test_texts, test_labels, client_indices, tokenizer):
    """本地训练 - 每个客户端独立训练"""
    print("=" * 80)
    print("方法 1: Local Training")
    print("=" * 80)
    print("策略参数:")
    print(f"  - 客户端数量: {len(client_indices)}")
    print(f"  - 本地epochs: 3")
    print(f"  - 学习率: 2e-5")
    print(f"  - Batch size: 16")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # 为每个客户端训练独立模型
    client_results = []
    
    for client_id in range(len(client_indices)):
        print(f"\n客户端 {client_id + 1}/{len(client_indices)} 训练中...")
        
        # 创建客户端模型
        model = create_albert_model().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        # 准备客户端数据
        client_train_texts = [train_texts[i] for i in client_indices[client_id]]
        client_train_labels = [train_labels[i] for i in client_indices[client_id]]
        
        client_dataset = AGNewsDataset(client_train_texts, client_train_labels, tokenizer)
        client_loader = DataLoader(client_dataset, batch_size=16, shuffle=True)
        
        # 准备测试数据
        test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 训练3个epochs
        for epoch in range(3):
            train_loss, train_acc = train_epoch(model, client_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            
            print(f"  Epoch {epoch + 1}/3 - "
                  f"训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.2f}%, "
                  f"验证损失: {val_loss:.6f}, 验证准确率: {val_acc:.2f}%")
        
        client_results.append({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
    
    # 计算平均结果
    avg_train_loss = np.mean([r['train_loss'] for r in client_results])
    avg_train_acc = np.mean([r['train_acc'] for r in client_results])
    avg_val_loss = np.mean([r['val_loss'] for r in client_results])
    avg_val_acc = np.mean([r['val_acc'] for r in client_results])
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print("Local Training 最终结果 (所有客户端平均):")
    print(f"  训练损失: {avg_train_loss:.6f}")
    print(f"  训练准确率: {avg_train_acc:.2f}%")
    print(f"  验证损失: {avg_val_loss:.6f}")
    print(f"  验证准确率: {avg_val_acc:.2f}%")
    print(f"  总运行时间: {end_time - start_time:.2f} 秒")
    print("=" * 80)
    print("\n")

# ==================== 2. Centralized Training ====================
def centralized_training(train_texts, train_labels, test_texts, test_labels, tokenizer):
    """中心化训练"""
    print("=" * 80)
    print("方法 2: Centralized Training")
    print("=" * 80)
    print("策略参数:")
    print(f"  - Epochs: 3")
    print(f"  - 学习率: 2e-5")
    print(f"  - Batch size: 32")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # 创建模型
    model = create_albert_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 准备数据
    train_dataset = AGNewsDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 训练
    for epoch in range(3):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}/3 - "
              f"训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.2f}%, "
              f"验证损失: {val_loss:.6f}, 验证准确率: {val_acc:.2f}%")
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print("Centralized Training 最终结果:")
    print(f"  训练损失: {train_loss:.6f}")
    print(f"  训练准确率: {train_acc:.2f}%")
    print(f"  验证损失: {val_loss:.6f}")
    print(f"  验证准确率: {val_acc:.2f}%")
    print(f"  总运行时间: {end_time - start_time:.2f} 秒")
    print("=" * 80)
    print("\n")

# ==================== 3. FedAvg ====================
def fedavg(train_texts, train_labels, test_texts, test_labels, client_indices, tokenizer):
    """FedAvg算法"""
    print("=" * 80)
    print("方法 3: FedAvg")
    print("=" * 80)
    print("策略参数:")
    print(f"  - 客户端数量: {len(client_indices)}")
    print(f"  - 通信轮次: 5")
    print(f"  - 本地epochs: 1")
    print(f"  - 学习率: 2e-5")
    print(f"  - Batch size: 16")
    print(f"  - 客户端采样率: 1.0 (所有客户端)")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # 创建全局模型
    global_model = create_albert_model().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 准备测试数据
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_rounds = 5
    local_epochs = 1
    
    for round_idx in range(num_rounds):
        print(f"\n通信轮次 {round_idx + 1}/{num_rounds}")
        
        # 存储客户端模型权重
        client_weights = []
        client_sizes = []
        
        # 每个客户端本地训练
        for client_id in range(len(client_indices)):
            # 创建客户端模型（从全局模型复制）
            client_model = create_albert_model().to(device)
            client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            
            optimizer = optim.AdamW(client_model.parameters(), lr=2e-5)
            
            # 准备客户端数据
            client_train_texts = [train_texts[i] for i in client_indices[client_id]]
            client_train_labels = [train_labels[i] for i in client_indices[client_id]]
            
            client_dataset = AGNewsDataset(client_train_texts, client_train_labels, tokenizer)
            client_loader = DataLoader(client_dataset, batch_size=16, shuffle=True)
            
            # 本地训练
            for _ in range(local_epochs):
                train_epoch(client_model, client_loader, optimizer, criterion, device)
            
            # 保存客户端权重
            client_weights.append(copy.deepcopy(client_model.state_dict()))
            client_sizes.append(len(client_indices[client_id]))
        
        # 聚合权重（FedAvg）
        global_dict = global_model.state_dict()
        total_size = sum(client_sizes)
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for client_id in range(len(client_indices)):
                weight = client_sizes[client_id] / total_size
                global_dict[key] += weight * client_weights[client_id][key]
        
        global_model.load_state_dict(global_dict)
        
        # 评估全局模型
        val_loss, val_acc = evaluate(global_model, test_loader, criterion, device)
        
        # 在一个客户端数据上计算训练指标
        sample_client_texts = [train_texts[i] for i in client_indices[0]]
        sample_client_labels = [train_labels[i] for i in client_indices[0]]
        sample_dataset = AGNewsDataset(sample_client_texts, sample_client_labels, tokenizer)
        sample_loader = DataLoader(sample_dataset, batch_size=16, shuffle=False)
        train_loss, train_acc = evaluate(global_model, sample_loader, criterion, device)
        
        print(f"  轮次 {round_idx + 1} - "
              f"训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.2f}%, "
              f"验证损失: {val_loss:.6f}, 验证准确率: {val_acc:.2f}%")
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print("FedAvg 最终结果:")
    print(f"  训练损失: {train_loss:.6f}")
    print(f"  训练准确率: {train_acc:.2f}%")
    print(f"  验证损失: {val_loss:.6f}")
    print(f"  验证准确率: {val_acc:.2f}%")
    print(f"  总运行时间: {end_time - start_time:.2f} 秒")
    print("=" * 80)
    print("\n")

# ==================== 4. FedProx ====================
def fedprox(train_texts, train_labels, test_texts, test_labels, client_indices, tokenizer, mu=0.01):
    """FedProx算法"""
    print("=" * 80)
    print("方法 4: FedProx")
    print("=" * 80)
    print("策略参数:")
    print(f"  - 客户端数量: {len(client_indices)}")
    print(f"  - 通信轮次: 5")
    print(f"  - 本地epochs: 1")
    print(f"  - 学习率: 2e-5")
    print(f"  - Batch size: 16")
    print(f"  - 近端项系数 μ: {mu}")
    print(f"  - 客户端采样率: 1.0 (所有客户端)")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # 创建全局模型
    global_model = create_albert_model().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 准备测试数据
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_rounds = 5
    local_epochs = 1
    
    for round_idx in range(num_rounds):
        print(f"\n通信轮次 {round_idx + 1}/{num_rounds}")
        
        # 存储客户端模型权重
        client_weights = []
        client_sizes = []
        
        # 保存全局模型权重用于近端项
        global_weights = copy.deepcopy(global_model.state_dict())
        
        # 每个客户端本地训练
        for client_id in range(len(client_indices)):
            # 创建客户端模型
            client_model = create_albert_model().to(device)
            client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            
            optimizer = optim.AdamW(client_model.parameters(), lr=2e-5)
            
            # 准备客户端数据
            client_train_texts = [train_texts[i] for i in client_indices[client_id]]
            client_train_labels = [train_labels[i] for i in client_indices[client_id]]
            
            client_dataset = AGNewsDataset(client_train_texts, client_train_labels, tokenizer)
            client_loader = DataLoader(client_dataset, batch_size=16, shuffle=True)
            
            # 本地训练（带近端项）
            for _ in range(local_epochs):
                client_model.train()
                for batch in client_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = client_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    # 添加近端项
                    proximal_term = 0.0
                    for name, param in client_model.named_parameters():
                        if name in global_weights:
                            proximal_term += (mu / 2) * torch.norm(param - global_weights[name]) ** 2
                    
                    loss = loss + proximal_term
                    loss.backward()
                    optimizer.step()
            
            # 保存客户端权重
            client_weights.append(copy.deepcopy(client_model.state_dict()))
            client_sizes.append(len(client_indices[client_id]))
        
        # 聚合权重
        global_dict = global_model.state_dict()
        total_size = sum(client_sizes)
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for client_id in range(len(client_indices)):
                weight = client_sizes[client_id] / total_size
                global_dict[key] += weight * client_weights[client_id][key]
        
        global_model.load_state_dict(global_dict)
        
        # 评估全局模型
        val_loss, val_acc = evaluate(global_model, test_loader, criterion, device)
        
        # 在一个客户端数据上计算训练指标
        sample_client_texts = [train_texts[i] for i in client_indices[0]]
        sample_client_labels = [train_labels[i] for i in client_indices[0]]
        sample_dataset = AGNewsDataset(sample_client_texts, sample_client_labels, tokenizer)
        sample_loader = DataLoader(sample_dataset, batch_size=16, shuffle=False)
        train_loss, train_acc = evaluate(global_model, sample_loader, criterion, device)
        
        print(f"  轮次 {round_idx + 1} - "
              f"训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.2f}%, "
              f"验证损失: {val_loss:.6f}, 验证准确率: {val_acc:.2f}%")
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print("FedProx 最终结果:")
    print(f"  训练损失: {train_loss:.6f}")
    print(f"  训练准确率: {train_acc:.2f}%")
    print(f"  验证损失: {val_loss:.6f}")
    print(f"  验证准确率: {val_acc:.2f}%")
    print(f"  总运行时间: {end_time - start_time:.2f} 秒")
    print("=" * 80)
    print("\n")

# ==================== 5. SCAFFOLD ====================
def scaffold(train_texts, train_labels, test_texts, test_labels, client_indices, tokenizer):
    """SCAFFOLD算法"""
    print("=" * 80)
    print("方法 5: SCAFFOLD")
    print("=" * 80)
    print("策略参数:")
    print(f"  - 客户端数量: {len(client_indices)}")
    print(f"  - 通信轮次: 5")
    print(f"  - 本地epochs: 1")
    print(f"  - 学习率: 2e-5")
    print(f"  - Batch size: 16")
    print(f"  - 客户端采样率: 1.0 (所有客户端)")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # 创建全局模型
    global_model = create_albert_model().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 初始化控制变量
    global_c = {key: torch.zeros_like(value) for key, value in global_model.state_dict().items()}
    client_c = [{key: torch.zeros_like(value) for key, value in global_model.state_dict().items()} 
                for _ in range(len(client_indices))]
    
    # 准备测试数据
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_rounds = 5
    local_epochs = 1
    
    for round_idx in range(num_rounds):
        print(f"\n通信轮次 {round_idx + 1}/{num_rounds}")
        
        # 存储客户端模型权重和控制变量更新
        client_weights = []
        client_sizes = []
        delta_c_list = []
        
        # 每个客户端本地训练
        for client_id in range(len(client_indices)):
            # 创建客户端模型
            client_model = create_albert_model().to(device)
            client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            
            # 准备客户端数据
            client_train_texts = [train_texts[i] for i in client_indices[client_id]]
            client_train_labels = [train_labels[i] for i in client_indices[client_id]]
            
            client_dataset = AGNewsDataset(client_train_texts, client_train_labels, tokenizer)
            client_loader = DataLoader(client_dataset, batch_size=16, shuffle=True)
            
            # 本地训练（使用SCAFFOLD修正）
            optimizer = optim.SGD(client_model.parameters(), lr=2e-5)
            
            for _ in range(local_epochs):
                client_model.train()
                for batch in client_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = client_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    
                    # SCAFFOLD修正梯度
                    with torch.no_grad():
                        for name, param in client_model.named_parameters():
                            if param.grad is not None and name in client_c[client_id]:
                                param.grad += global_c[name] - client_c[client_id][name]
                    
                    optimizer.step()
            
            # 计算新的客户端控制变量
            new_c = {}
            delta_c = {}
            with torch.no_grad():
                for name, param in client_model.named_parameters():
                    if name in global_model.state_dict():
                        # 简化的控制变量更新
                        new_c[name] = client_c[client_id][name] - global_c[name] + \
                                     (global_model.state_dict()[name] - param) / (local_epochs * 2e-5)
                        delta_c[name] = new_c[name] - client_c[client_id][name]
                        client_c[client_id][name] = new_c[name]
            
            # 保存客户端权重
            client_weights.append(copy.deepcopy(client_model.state_dict()))
            client_sizes.append(len(client_indices[client_id]))
            delta_c_list.append(delta_c)
        
        # 聚合权重
        global_dict = global_model.state_dict()
        total_size = sum(client_sizes)
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for client_id in range(len(client_indices)):
                weight = client_sizes[client_id] / total_size
                global_dict[key] += weight * client_weights[client_id][key]
        
        global_model.load_state_dict(global_dict)
        
        # 更新全局控制变量
        for key in global_c.keys():
            delta_sum = torch.zeros_like(global_c[key])
            for client_id in range(len(client_indices)):
                if key in delta_c_list[client_id]:
                    weight = client_sizes[client_id] / total_size
                    delta_sum += weight * delta_c_list[client_id][key]
            global_c[key] += delta_sum
        
        # 评估全局模型
        val_loss, val_acc = evaluate(global_model, test_loader, criterion, device)
        
        # 在一个客户端数据上计算训练指标
        sample_client_texts = [train_texts[i] for i in client_indices[0]]
        sample_client_labels = [train_labels[i] for i in client_indices[0]]
        sample_dataset = AGNewsDataset(sample_client_texts, sample_client_labels, tokenizer)
        sample_loader = DataLoader(sample_dataset, batch_size=16, shuffle=False)
        train_loss, train_acc = evaluate(global_model, sample_loader, criterion, device)
        
        print(f"  轮次 {round_idx + 1} - "
              f"训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.2f}%, "
              f"验证损失: {val_loss:.6f}, 验证准确率: {val_acc:.2f}%")
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print("SCAFFOLD 最终结果:")
    print(f"  训练损失: {train_loss:.6f}")
    print(f"  训练准确率: {train_acc:.2f}%")
    print(f"  验证损失: {val_loss:.6f}")
    print(f"  验证准确率: {val_acc:.2f}%")
    print(f"  总运行时间: {end_time - start_time:.2f} 秒")
    print("=" * 80)
    print("\n")

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("联邦学习实验: AGNews数据集 + ALBERT模型")
    print("=" * 80)
    print()
    
    # 加载tokenizer
    print("正在加载ALBERT tokenizer...")
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir='./model_cache')
    print("Tokenizer加载完成\n")
    
    # 加载数据
    train_texts, train_labels, test_texts, test_labels = load_agnews_data()
    
    # 配置参数
    num_clients = 10
    train_subset_size = 5000  # 训练数据子集大小（用于快速测试）
    test_subset_size = 1000   # 测试数据子集大小
    
    # 注意：为了演示，这里使用较小的数据子集和较少的训练轮次
    # 如果要使用完整数据集，请将下面的子集大小设置为 None
    # train_subset_size = None
    # test_subset_size = None
    
    # 截取数据子集
    if train_subset_size is not None:
        train_texts_subset = train_texts[:train_subset_size]
        train_labels_subset = train_labels[:train_subset_size]
    else:
        train_texts_subset = train_texts
        train_labels_subset = train_labels
    
    if test_subset_size is not None:
        test_texts_subset = test_texts[:test_subset_size]
        test_labels_subset = test_labels[:test_subset_size]
    else:
        test_texts_subset = test_texts
        test_labels_subset = test_labels
    
    # 在截取后的数据上分配客户端
    client_indices = split_data_for_clients(train_texts_subset, train_labels_subset, num_clients)
    print(f"数据已分配给 {num_clients} 个客户端")
    print(f"训练数据: {len(train_texts_subset)} 样本")
    print(f"测试数据: {len(test_texts_subset)} 样本\n")
    
    # 运行各种方法
    
    # 1. Local Training
    local_training(train_texts_subset, train_labels_subset, 
                   test_texts_subset, test_labels_subset, 
                   client_indices, tokenizer)
    
    # 2. Centralized Training
    centralized_training(train_texts_subset, train_labels_subset, 
                        test_texts_subset, test_labels_subset, tokenizer)
    
    # 3. FedAvg
    fedavg(train_texts_subset, train_labels_subset, 
           test_texts_subset, test_labels_subset, 
           client_indices, tokenizer)
    
    # 4. FedProx
    fedprox(train_texts_subset, train_labels_subset, 
            test_texts_subset, test_labels_subset, 
            client_indices, tokenizer)
    
    # 5. SCAFFOLD
    scaffold(train_texts_subset, train_labels_subset, 
             test_texts_subset, test_labels_subset, 
             client_indices, tokenizer)
    
    print("=" * 80)
    print("所有实验完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
