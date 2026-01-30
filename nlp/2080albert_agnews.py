"""
联邦学习实验 - 完全离线版本
100%离线运行，不会尝试任何网络连接
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import random
import copy
import os

# ============ 强制离线模式 ============
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# ============ 配置您的路径（修改这里！）============
BASE_DIR = '/data/us-xc/jxj/FL/nlp'
MODEL_CACHE_DIR = os.path.join(BASE_DIR, 'model_cache')
DATA_CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')

print("=" * 80)
print("完全离线模式")
print("=" * 80)
print(f"基础目录: {BASE_DIR}")
print(f"模型缓存: {MODEL_CACHE_DIR}")
print(f"数据缓存: {DATA_CACHE_DIR}")
print("=" * 80)
print()

# 导入transformers和datasets（在设置离线模式后）
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from datasets import load_dataset

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

# ==================== 数据集类 ====================
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

# ==================== 数据加载 ====================
def load_agnews_data():
    """从本地缓存加载AGNews数据集（完全离线）"""
    print("正在从本地缓存加载AGNews数据集...")
    print(f"数据缓存目录: {DATA_CACHE_DIR}")
    
    # 检查缓存是否存在
    if not os.path.exists(DATA_CACHE_DIR):
        raise FileNotFoundError(f"数据缓存目录不存在: {DATA_CACHE_DIR}")
    
    try:
        # 强制离线加载
        dataset = load_dataset('ag_news', cache_dir=DATA_CACHE_DIR, download_mode='reuse_cache_if_exists')
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        print("\n请确认:")
        print(f"1. 数据缓存目录存在: {DATA_CACHE_DIR}")
        print(f"2. 目录中有AGNews数据集文件")
        raise
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    print(f"✓ 数据加载成功")
    print(f"  训练集: {len(train_texts)} 样本")
    print(f"  测试集: {len(test_texts)} 样本")
    print()
    
    return train_texts, train_labels, test_texts, test_labels

def split_data_for_clients(train_texts, train_labels, num_clients=10):
    """将数据分配给客户端"""
    num_samples = len(train_texts)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    client_indices = []
    samples_per_client = num_samples // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else num_samples
        client_indices.append(indices[start_idx:end_idx])
    
    return client_indices

# ==================== 模型加载 ====================
def create_albert_model(num_labels=4):
    """从本地缓存加载ALBERT模型（完全离线）"""
    
    # 检查缓存是否存在
    if not os.path.exists(MODEL_CACHE_DIR):
        raise FileNotFoundError(f"模型缓存目录不存在: {MODEL_CACHE_DIR}")
    
    # 强制离线加载
    model = AlbertForSequenceClassification.from_pretrained(
        'albert-base-v2',
        num_labels=num_labels,
        cache_dir=MODEL_CACHE_DIR,
        local_files_only=True  # 关键：只使用本地文件
    )
    return model

# ==================== 训练和评估 ====================
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
    """本地训练"""
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
    
    client_results = []
    
    for client_id in range(len(client_indices)):
        print(f"\n客户端 {client_id + 1}/{len(client_indices)} 训练中...")
        
        model = create_albert_model().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        client_train_texts = [train_texts[i] for i in client_indices[client_id]]
        client_train_labels = [train_labels[i] for i in client_indices[client_id]]
        
        client_dataset = AGNewsDataset(client_train_texts, client_train_labels, tokenizer)
        client_loader = DataLoader(client_dataset, batch_size=16, shuffle=True)
        
        test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
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
    
    model = create_albert_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = AGNewsDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
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
    print(f"  - 客户端采样率: 1.0")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    global_model = create_albert_model().to(device)
    criterion = nn.CrossEntropyLoss()
    
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_rounds = 5
    local_epochs = 1
    
    for round_idx in range(num_rounds):
        print(f"\n通信轮次 {round_idx + 1}/{num_rounds}")
        
        client_weights = []
        client_sizes = []
        
        for client_id in range(len(client_indices)):
            client_model = create_albert_model().to(device)
            client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            
            optimizer = optim.AdamW(client_model.parameters(), lr=2e-5)
            
            client_train_texts = [train_texts[i] for i in client_indices[client_id]]
            client_train_labels = [train_labels[i] for i in client_indices[client_id]]
            
            client_dataset = AGNewsDataset(client_train_texts, client_train_labels, tokenizer)
            client_loader = DataLoader(client_dataset, batch_size=16, shuffle=True)
            
            for _ in range(local_epochs):
                train_epoch(client_model, client_loader, optimizer, criterion, device)
            
            client_weights.append(copy.deepcopy(client_model.state_dict()))
            client_sizes.append(len(client_indices[client_id]))
        
        global_dict = global_model.state_dict()
        total_size = sum(client_sizes)
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for client_id in range(len(client_indices)):
                weight = client_sizes[client_id] / total_size
                global_dict[key] += weight * client_weights[client_id][key]
        
        global_model.load_state_dict(global_dict)
        
        val_loss, val_acc = evaluate(global_model, test_loader, criterion, device)
        
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
    print(f"  - 客户端采样率: 1.0")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    global_model = create_albert_model().to(device)
    criterion = nn.CrossEntropyLoss()
    
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_rounds = 5
    local_epochs = 1
    
    for round_idx in range(num_rounds):
        print(f"\n通信轮次 {round_idx + 1}/{num_rounds}")
        
        client_weights = []
        client_sizes = []
        
        global_weights = copy.deepcopy(global_model.state_dict())
        
        for client_id in range(len(client_indices)):
            client_model = create_albert_model().to(device)
            client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            
            optimizer = optim.AdamW(client_model.parameters(), lr=2e-5)
            
            client_train_texts = [train_texts[i] for i in client_indices[client_id]]
            client_train_labels = [train_labels[i] for i in client_indices[client_id]]
            
            client_dataset = AGNewsDataset(client_train_texts, client_train_labels, tokenizer)
            client_loader = DataLoader(client_dataset, batch_size=16, shuffle=True)
            
            for _ in range(local_epochs):
                client_model.train()
                for batch in client_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = client_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    proximal_term = 0.0
                    for name, param in client_model.named_parameters():
                        if name in global_weights:
                            proximal_term += (mu / 2) * torch.norm(param - global_weights[name]) ** 2
                    
                    loss = loss + proximal_term
                    loss.backward()
                    optimizer.step()
            
            client_weights.append(copy.deepcopy(client_model.state_dict()))
            client_sizes.append(len(client_indices[client_id]))
        
        global_dict = global_model.state_dict()
        total_size = sum(client_sizes)
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for client_id in range(len(client_indices)):
                weight = client_sizes[client_id] / total_size
                global_dict[key] += weight * client_weights[client_id][key]
        
        global_model.load_state_dict(global_dict)
        
        val_loss, val_acc = evaluate(global_model, test_loader, criterion, device)
        
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
    print(f"  - 客户端采样率: 1.0")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    global_model = create_albert_model().to(device)
    criterion = nn.CrossEntropyLoss()
    
    global_c = {key: torch.zeros_like(value) for key, value in global_model.state_dict().items()}
    client_c = [{key: torch.zeros_like(value) for key, value in global_model.state_dict().items()} 
                for _ in range(len(client_indices))]
    
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_rounds = 5
    local_epochs = 1
    
    for round_idx in range(num_rounds):
        print(f"\n通信轮次 {round_idx + 1}/{num_rounds}")
        
        client_weights = []
        client_sizes = []
        delta_c_list = []
        
        for client_id in range(len(client_indices)):
            client_model = create_albert_model().to(device)
            client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            
            client_train_texts = [train_texts[i] for i in client_indices[client_id]]
            client_train_labels = [train_labels[i] for i in client_indices[client_id]]
            
            client_dataset = AGNewsDataset(client_train_texts, client_train_labels, tokenizer)
            client_loader = DataLoader(client_dataset, batch_size=16, shuffle=True)
            
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
                    
                    with torch.no_grad():
                        for name, param in client_model.named_parameters():
                            if param.grad is not None and name in client_c[client_id]:
                                param.grad += global_c[name] - client_c[client_id][name]
                    
                    optimizer.step()
            
            new_c = {}
            delta_c = {}
            with torch.no_grad():
                for name, param in client_model.named_parameters():
                    if name in global_model.state_dict():
                        new_c[name] = client_c[client_id][name] - global_c[name] + \
                                     (global_model.state_dict()[name] - param) / (local_epochs * 2e-5)
                        delta_c[name] = new_c[name] - client_c[client_id][name]
                        client_c[client_id][name] = new_c[name]
            
            client_weights.append(copy.deepcopy(client_model.state_dict()))
            client_sizes.append(len(client_indices[client_id]))
            delta_c_list.append(delta_c)
        
        global_dict = global_model.state_dict()
        total_size = sum(client_sizes)
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for client_id in range(len(client_indices)):
                weight = client_sizes[client_id] / total_size
                global_dict[key] += weight * client_weights[client_id][key]
        
        global_model.load_state_dict(global_dict)
        
        for key in global_c.keys():
            delta_sum = torch.zeros_like(global_c[key])
            for client_id in range(len(client_indices)):
                if key in delta_c_list[client_id]:
                    weight = client_sizes[client_id] / total_size
                    delta_sum += weight * delta_c_list[client_id][key]
            global_c[key] += delta_sum
        
        val_loss, val_acc = evaluate(global_model, test_loader, criterion, device)
        
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
    
    # 加载tokenizer（完全离线）
    print("正在加载ALBERT tokenizer...")
    print(f"模型缓存目录: {MODEL_CACHE_DIR}")
    tokenizer = AlbertTokenizer.from_pretrained(
        'albert-base-v2', 
        cache_dir=MODEL_CACHE_DIR,
        local_files_only=True  # 只使用本地文件
    )
    print("✓ Tokenizer加载完成\n")
    
    # 加载数据
    train_texts, train_labels, test_texts, test_labels = load_agnews_data()
    
    # 配置参数
    num_clients = 10
    train_subset_size = 5000  # 可修改为 None 使用全部数据
    test_subset_size = 1000   # 可修改为 None 使用全部数据
    
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
    
    # 分配客户端数据
    client_indices = split_data_for_clients(train_texts_subset, train_labels_subset, num_clients)
    print(f"数据已分配给 {num_clients} 个客户端")
    print(f"训练数据: {len(train_texts_subset)} 样本")
    print(f"测试数据: {len(test_texts_subset)} 样本\n")
    
    # 运行各种方法
    local_training(train_texts_subset, train_labels_subset, 
                   test_texts_subset, test_labels_subset, 
                   client_indices, tokenizer)
    
    centralized_training(train_texts_subset, train_labels_subset, 
                        test_texts_subset, test_labels_subset, tokenizer)
    
    fedavg(train_texts_subset, train_labels_subset, 
           test_texts_subset, test_labels_subset, 
           client_indices, tokenizer)
    
    fedprox(train_texts_subset, train_labels_subset, 
            test_texts_subset, test_labels_subset, 
            client_indices, tokenizer)
    
    scaffold(train_texts_subset, train_labels_subset, 
             test_texts_subset, test_labels_subset, 
             client_indices, tokenizer)
    
    print("=" * 80)
    print("所有实验完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
