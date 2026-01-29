"""
=================================================================================
联邦学习 ALBERT 微调 - 完全离线版本 v2
=================================================================================

目录结构要求（与代码文件放在同一目录）：
├── albert_AGnews_offline.py  (本文件)
├── data/
│   └── ag_news/
│       └── default/0.0.0/xxx/
│           ├── ag_news-train.arrow
│           ├── ag_news-test.arrow
│           └── dataset_info.json
└── model/
    └── models--albert-base-v2/
        └── snapshots/xxx/
            ├── config.json
            ├── model.safetensors
            ├── spiece.model
            ├── tokenizer_config.json
            └── tokenizer.json

使用方法:
  python albert_AGnews_offline.py                          # 批量运行所有方法
  python albert_AGnews_offline.py --single --method fedavg # 只运行FedAvg
  python albert_AGnews_offline.py --num_rounds 30          # 自定义参数

=================================================================================
"""

import os
import sys

# =================================================================================
# 关键：在导入任何库之前，强制设置完全离线模式
# =================================================================================
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['NO_INTERNET'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''

# 修复 Python 3.12 多进程兼容性问题
import platform
if sys.version_info >= (3, 12):
    os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['HF_DATASETS_DISABLE_PROGRESS_BAR'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    if platform.system() != 'Windows':
        os.environ['PYTORCH_DATALOADER_MULTIPROCESSING_CONTEXT'] = 'fork'
    
    import warnings
    warnings.filterwarnings('ignore', category=ResourceWarning)
    warnings.filterwarnings('ignore', message='.*ResourceTracker.*')

# =================================================================================
# 导入其他库
# =================================================================================

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import random
import copy
import time
from datetime import timedelta
from collections import defaultdict

from transformers import AlbertModel, AlbertTokenizer
from torch.utils.data import Dataset as TorchDataset, DataLoader

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

print(f"★ 离线模式已启用")
print(f"★ 脚本目录: {SCRIPT_DIR}")
print(f"★ 数据目录: {DATA_DIR}")
print(f"★ 模型目录: {MODEL_DIR}")


# =================================================================================
# 第一部分: 模型定义
# =================================================================================

class ALBERTClassifier(nn.Module):
    """基于ALBERT的文本分类器"""
    
    _model_load_printed = {}
    
    def __init__(self, model_name='albert-base-v2', num_classes=4, dropout=0.1, cache_dir=None):
        super(ALBERTClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        if cache_dir is None:
            cache_dir = MODEL_DIR
        
        cache_key = f"{model_name}_{cache_dir}"
        if cache_key not in ALBERTClassifier._model_load_printed:
            print(f"从本地缓存加载模型: {cache_dir}")
            ALBERTClassifier._model_load_printed[cache_key] = True
        
        self.albert = self._load_albert_model(model_name, cache_dir)
        
        self.hidden_size = self.albert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def _load_albert_model(self, model_name, cache_dir):
        """尝试多种方式加载ALBERT模型"""
        
        # 方法1: 直接从缓存加载
        try:
            return AlbertModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        except Exception as e:
            print(f"方法1失败: {e}")
        
        # 方法2: 从 snapshots 目录加载
        model_cache_name = f"models--{model_name.replace('/', '--')}"
        model_cache_path = os.path.join(cache_dir, model_cache_name)
        
        if os.path.exists(model_cache_path):
            snapshots_dir = os.path.join(model_cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                    print(f"尝试从快照加载: {snapshot_path}")
                    try:
                        return AlbertModel.from_pretrained(snapshot_path, local_files_only=True)
                    except Exception as e:
                        print(f"方法2失败: {e}")
        
        # 方法3: 搜索包含 config.json 的目录
        for root, dirs, files in os.walk(cache_dir):
            if 'config.json' in files and ('model.safetensors' in files or 'pytorch_model.bin' in files):
                print(f"尝试从目录加载: {root}")
                try:
                    return AlbertModel.from_pretrained(root, local_files_only=True)
                except Exception as e:
                    print(f"方法3失败: {e}")
        
        raise RuntimeError(f"无法加载模型。请确保 {cache_dir} 目录包含完整的 {model_name} 模型文件")
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        
        return logits
    
    def clone(self):
        clone_model = ALBERTClassifier(model_name=self.model_name, num_classes=self.num_classes)
        clone_model.load_state_dict(self.state_dict())
        return clone_model


# =================================================================================
# 第二部分: 数据处理
# =================================================================================

class AGNewsDataset(TorchDataset):
    """AG News 数据集类"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


def find_arrow_directory(base_dir):
    """查找包含 arrow 文件的目录"""
    for root, dirs, files in os.walk(base_dir):
        arrow_files = [f for f in files if f.endswith('.arrow')]
        if len(arrow_files) >= 2:
            return root
    return None


def load_ag_news_offline(data_dir=None):
    """从本地加载 AG News 数据集（完全离线）"""
    print("\n正在从本地加载 AG News 数据集...")
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    ag_news_dir = os.path.join(data_dir, "ag_news")
    
    if not os.path.exists(ag_news_dir):
        raise RuntimeError(f"数据集目录不存在: {ag_news_dir}")
    
    arrow_dir = find_arrow_directory(ag_news_dir)
    
    if not arrow_dir:
        raise RuntimeError(f"未找到 arrow 文件。请确保 {ag_news_dir} 包含数据集文件")
    
    print(f"✓ 找到数据目录: {arrow_dir}")
    
    train_file = None
    test_file = None
    
    for f in os.listdir(arrow_dir):
        if f.endswith('.arrow'):
            full_path = os.path.join(arrow_dir, f)
            if 'train' in f.lower():
                train_file = full_path
            elif 'test' in f.lower():
                test_file = full_path
    
    if not train_file or not test_file:
        raise RuntimeError(f"未找到完整的数据集文件（需要train和test）")
    
    print(f"✓ 训练集文件: {train_file}")
    print(f"✓ 测试集文件: {test_file}")
    
    # 使用 pyarrow streaming 格式读取
    import pyarrow as pa
    
    def read_arrow_streaming(file_path):
        """读取 arrow streaming 格式文件"""
        with open(file_path, 'rb') as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()
        return table.to_pandas()
    
    print("正在读取训练集...")
    train_df = read_arrow_streaming(train_file)
    print("正在读取测试集...")
    test_df = read_arrow_streaming(test_file)
    
    train_data = [{'text': row['text'], 'label': int(row['label'])} for _, row in train_df.iterrows()]
    test_data = [{'text': row['text'], 'label': int(row['label'])} for _, row in test_df.iterrows()]
    
    print(f"✓ 离线加载成功")
    print(f"  - 训练集: {len(train_data)} 样本")
    print(f"  - 测试集: {len(test_data)} 样本")
    
    return train_data, test_data


def create_iid_data(data, num_clients):
    """创建IID数据分布"""
    data_copy = data.copy()
    np.random.shuffle(data_copy)
    
    client_data = []
    samples_per_client = len(data_copy) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(data_copy)
        client_data.append(data_copy[start_idx:end_idx])
    
    return client_data


def create_non_iid_data(data, num_clients, alpha=0.5):
    """使用Dirichlet分布创建Non-IID数据"""
    label_to_indices = defaultdict(list)
    for idx, item in enumerate(data):
        label_to_indices[item['label']].append(idx)
    
    client_indices = [[] for _ in range(num_clients)]
    
    for label, indices in label_to_indices.items():
        indices_copy = indices.copy()
        np.random.shuffle(indices_copy)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(indices_copy)).astype(int)[:-1]
        
        split_indices = np.split(indices_copy, proportions)
        
        for client_id, split in enumerate(split_indices):
            client_indices[client_id].extend(split)
    
    client_data = []
    for indices in client_indices:
        np.random.shuffle(indices)
        client_data.append([data[i] for i in indices])
    
    return client_data


def create_federated_data(data, num_clients=10, distribution='non-iid', alpha=0.5):
    """创建联邦数据分布"""
    if distribution == 'iid':
        return create_iid_data(data, num_clients)
    else:
        return create_non_iid_data(data, num_clients, alpha)


def load_tokenizer_offline(model_name='albert-base-v2', cache_dir=None):
    """从本地加载tokenizer"""
    if cache_dir is None:
        cache_dir = MODEL_DIR
    
    # 方法1
    try:
        return AlbertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    except Exception as e:
        print(f"tokenizer方法1失败: {e}")
    
    # 方法2: snapshots
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = os.path.join(cache_dir, model_cache_name)
    
    if os.path.exists(model_cache_path):
        snapshots_dir = os.path.join(model_cache_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                print(f"尝试从快照加载tokenizer: {snapshot_path}")
                try:
                    return AlbertTokenizer.from_pretrained(snapshot_path, local_files_only=True)
                except Exception as e:
                    print(f"tokenizer方法2失败: {e}")
    
    # 方法3
    for root, dirs, files in os.walk(cache_dir):
        if 'tokenizer_config.json' in files or 'spiece.model' in files:
            print(f"尝试从目录加载tokenizer: {root}")
            try:
                return AlbertTokenizer.from_pretrained(root, local_files_only=True)
            except Exception as e:
                print(f"tokenizer方法3失败: {e}")
    
    raise RuntimeError(f"无法加载tokenizer。请确保模型已正确下载到 {cache_dir}")


def create_dataloaders(client_data_list, batch_size, model_name, max_length=128, cache_dir=None):
    """创建数据加载器"""
    if cache_dir is None:
        cache_dir = MODEL_DIR
    
    print(f"从本地缓存加载tokenizer: {cache_dir}")
    tokenizer = load_tokenizer_offline(model_name, cache_dir)
    
    num_workers = 0 if platform.system() == 'Windows' or sys.version_info >= (3, 12) else 2
    
    dataloaders = []
    for client_data in client_data_list:
        dataset = AGNewsDataset(client_data, tokenizer, max_length)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False, persistent_workers=False
        )
        dataloaders.append(dataloader)
    
    return dataloaders


# =================================================================================
# 第三部分: 工具函数
# =================================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, logits = model(input_ids, attention_mask, labels)
            
            total_loss += loss.item()
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        loss, logits = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


# =================================================================================
# 第四部分: 联邦学习算法
# =================================================================================

def local_training(train_loaders, test_loader, model_template, args):
    print("\n" + "="*80)
    print("【开始训练：本地训练 (Local Training)】")
    print("="*80)
    
    start_time = time.time()
    results = {'train_loss': [], 'test_accuracy': [], 'test_loss': []}
    
    client_models = [model_template.clone().to(args.device) for _ in range(args.num_clients)]
    
    for client_id, (model, train_loader) in enumerate(zip(client_models, train_loaders)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.local_epochs):
            train_epoch(model, train_loader, optimizer, args.device)
    
    test_accuracies, test_losses = [], []
    
    for client_id, model in enumerate(client_models):
        test_loss, test_acc = evaluate_model(model, test_loader, args.device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        print(f"[Local][Client {client_id:02d}] TestLoss {test_loss:.6f} TestAcc {test_acc*100:.2f}%")
    
    results['test_accuracy'].append(np.mean(test_accuracies))
    results['test_loss'].append(np.mean(test_losses))
    results['training_time'] = time.time() - start_time
    
    print(f"\n【本地训练完成】耗时: {results['training_time']:.2f}秒, 平均准确率: {np.mean(test_accuracies)*100:.2f}%")
    return results


def centralized_training(train_loader, test_loader, model, args):
    print("\n" + "="*80)
    print("【开始训练：集中式学习 (Centralized)】")
    print("="*80)
    
    start_time = time.time()
    results = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.num_rounds):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, args.device)
        test_loss, test_acc = evaluate_model(model, test_loader, args.device)
        
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        
        print(f"[Centralized][Epoch {epoch+1:03d}] TrainLoss {train_loss:.6f} TrainAcc {train_acc*100:.2f}% | TestLoss {test_loss:.6f} TestAcc {test_acc*100:.2f}%")
    
    results['training_time'] = time.time() - start_time
    print(f"\n【集中式训练完成】耗时: {results['training_time']:.2f}秒")
    return results


def fedavg_training(train_loaders, test_loader, global_model, args):
    print("\n" + "="*80)
    print("【开始训练：联邦平均 (FedAvg)】")
    print("="*80)
    
    start_time = time.time()
    results = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': [], 'communication_rounds': []}
    client_data_sizes = [len(loader.dataset) for loader in train_loaders]
    
    for round_idx in range(args.num_rounds):
        selected_clients = np.random.choice(args.num_clients, args.clients_per_round, replace=False)
        client_weights, round_losses, round_accuracies = [], [], []
        
        for client_id in selected_clients:
            local_model = global_model.clone().to(args.device)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            total_loss, total_correct, total_samples = 0, 0, 0
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train_epoch(local_model, train_loaders[client_id], optimizer, args.device)
                total_loss += train_loss
                total_correct += train_acc * len(train_loaders[client_id].dataset)
                total_samples += len(train_loaders[client_id].dataset)
            
            round_losses.append(total_loss / args.local_epochs)
            round_accuracies.append(total_correct / (total_samples * args.local_epochs))
            client_weights.append({'state_dict': copy.deepcopy(local_model.state_dict()), 'data_size': client_data_sizes[client_id]})
        
        aggregated_state_dict = {}
        selected_data_size = sum([cw['data_size'] for cw in client_weights])
        for key in global_model.state_dict().keys():
            aggregated_state_dict[key] = sum([cw['state_dict'][key] * (cw['data_size'] / selected_data_size) for cw in client_weights])
        global_model.load_state_dict(aggregated_state_dict)
        
        avg_loss, avg_acc = np.mean(round_losses), np.mean(round_accuracies)
        test_loss, test_acc = evaluate_model(global_model, test_loader, args.device)
        
        results['train_loss'].append(avg_loss)
        results['train_accuracy'].append(avg_acc)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        results['communication_rounds'].append(round_idx + 1)
        
        print(f"[FedAvg][Round {round_idx+1:03d}] AvgClientLoss {avg_loss:.6f} AvgClientAcc {avg_acc*100:.2f}% | TestLoss {test_loss:.6f} TestAcc {test_acc*100:.2f}%")
    
    results['training_time'] = time.time() - start_time
    print(f"\n【FedAvg训练完成】耗时: {results['training_time']:.2f}秒")
    return results


def fedprox_training(train_loaders, test_loader, global_model, args):
    print("\n" + "="*80)
    print(f"【开始训练：联邦近端 (FedProx) μ={args.mu}】")
    print("="*80)
    
    start_time = time.time()
    results = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': [], 'communication_rounds': []}
    client_data_sizes = [len(loader.dataset) for loader in train_loaders]
    
    for round_idx in range(args.num_rounds):
        selected_clients = np.random.choice(args.num_clients, args.clients_per_round, replace=False)
        client_weights, round_losses, round_accuracies = [], [], []
        global_params = copy.deepcopy(global_model.state_dict())
        
        for client_id in selected_clients:
            local_model = global_model.clone().to(args.device)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            total_loss, total_correct, total_samples = 0, 0, 0
            for epoch in range(args.local_epochs):
                local_model.train()
                for batch in train_loaders[client_id]:
                    input_ids = batch['input_ids'].to(args.device)
                    attention_mask = batch['attention_mask'].to(args.device)
                    labels = batch['labels'].to(args.device)
                    
                    optimizer.zero_grad()
                    loss, logits = local_model(input_ids, attention_mask, labels)
                    
                    proximal_term = sum(((param - global_params[name].to(args.device)) ** 2).sum() for name, param in local_model.named_parameters())
                    (loss + (args.mu / 2) * proximal_term).backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                    total_samples += labels.size(0)
            
            round_losses.append(total_loss / (len(train_loaders[client_id]) * args.local_epochs))
            round_accuracies.append(total_correct / total_samples)
            client_weights.append({'state_dict': copy.deepcopy(local_model.state_dict()), 'data_size': client_data_sizes[client_id]})
        
        aggregated_state_dict = {}
        selected_data_size = sum([cw['data_size'] for cw in client_weights])
        for key in global_model.state_dict().keys():
            aggregated_state_dict[key] = sum([cw['state_dict'][key] * (cw['data_size'] / selected_data_size) for cw in client_weights])
        global_model.load_state_dict(aggregated_state_dict)
        
        avg_loss, avg_acc = np.mean(round_losses), np.mean(round_accuracies)
        test_loss, test_acc = evaluate_model(global_model, test_loader, args.device)
        
        results['train_loss'].append(avg_loss)
        results['train_accuracy'].append(avg_acc)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        results['communication_rounds'].append(round_idx + 1)
        
        print(f"[FedProx][Round {round_idx+1:03d}][mu={args.mu}] AvgClientLoss {avg_loss:.6f} AvgClientAcc {avg_acc*100:.2f}% | TestLoss {test_loss:.6f} TestAcc {test_acc*100:.2f}%")
    
    results['training_time'] = time.time() - start_time
    print(f"\n【FedProx训练完成】耗时: {results['training_time']:.2f}秒")
    return results


def scaffold_training(train_loaders, test_loader, global_model, args):
    print("\n" + "="*80)
    print("【开始训练：SCAFFOLD】")
    print("="*80)
    
    start_time = time.time()
    results = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': [], 'communication_rounds': []}
    client_data_sizes = [len(loader.dataset) for loader in train_loaders]
    
    c_global = {key: torch.zeros_like(value) for key, value in global_model.state_dict().items()}
    c_clients = [{key: torch.zeros_like(value) for key, value in global_model.state_dict().items()} for _ in range(args.num_clients)]
    
    for round_idx in range(args.num_rounds):
        selected_clients = np.random.choice(args.num_clients, args.clients_per_round, replace=False)
        client_weights, client_control_updates, round_losses, round_accuracies = [], [], [], []
        
        for client_id in selected_clients:
            local_model = global_model.clone().to(args.device)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr)
            init_local_params = copy.deepcopy(local_model.state_dict())
            
            total_loss, total_correct, total_samples = 0, 0, 0
            for epoch in range(args.local_epochs):
                local_model.train()
                for batch in train_loaders[client_id]:
                    input_ids = batch['input_ids'].to(args.device)
                    attention_mask = batch['attention_mask'].to(args.device)
                    labels = batch['labels'].to(args.device)
                    
                    optimizer.zero_grad()
                    loss, logits = local_model(input_ids, attention_mask, labels)
                    loss.backward()
                    
                    with torch.no_grad():
                        for name, param in local_model.named_parameters():
                            if param.grad is not None:
                                param.grad.data.add_(c_global[name].to(args.device) - c_clients[client_id][name].to(args.device))
                    
                    optimizer.step()
                    total_loss += loss.item()
                    total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                    total_samples += labels.size(0)
            
            round_losses.append(total_loss / (len(train_loaders[client_id]) * args.local_epochs))
            round_accuracies.append(total_correct / total_samples)
            
            c_client_new = {}
            K = len(train_loaders[client_id]) * args.local_epochs
            with torch.no_grad():
                for key in c_clients[client_id].keys():
                    c_client_new[key] = c_clients[client_id][key] - c_global[key] + (init_local_params[key] - local_model.state_dict()[key]) / (K * args.lr)
            
            client_control_updates.append(c_client_new)
            c_clients[client_id] = c_client_new
            client_weights.append({'state_dict': copy.deepcopy(local_model.state_dict()), 'data_size': client_data_sizes[client_id]})
        
        aggregated_state_dict = {}
        selected_data_size = sum([cw['data_size'] for cw in client_weights])
        for key in global_model.state_dict().keys():
            aggregated_state_dict[key] = sum([cw['state_dict'][key] * (cw['data_size'] / selected_data_size) for cw in client_weights])
        global_model.load_state_dict(aggregated_state_dict)
        
        with torch.no_grad():
            for key in c_global.keys():
                delta_c = sum([(client_control_updates[i][key] - c_clients[selected_clients[i]][key]) for i in range(len(selected_clients))]) / len(selected_clients)
                c_global[key] = c_global[key] + args.server_lr * delta_c
        
        avg_loss, avg_acc = np.mean(round_losses), np.mean(round_accuracies)
        test_loss, test_acc = evaluate_model(global_model, test_loader, args.device)
        
        results['train_loss'].append(avg_loss)
        results['train_accuracy'].append(avg_acc)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        results['communication_rounds'].append(round_idx + 1)
        
        print(f"[SCAFFOLD][Round {round_idx+1:03d}] AvgClientLoss {avg_loss:.6f} AvgClientAcc {avg_acc*100:.2f}% | TestLoss {test_loss:.6f} TestAcc {test_acc*100:.2f}%")
    
    results['training_time'] = time.time() - start_time
    print(f"\n【SCAFFOLD训练完成】耗时: {results['training_time']:.2f}秒")
    return results


# =================================================================================
# 第五部分: 主程序
# =================================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='联邦学习 ALBERT 微调 (离线版本)')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--model_name', type=str, default='albert-base-v2')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=128)
    
    parser.add_argument('--method', type=str, default='fedavg', choices=['local', 'centralized', 'fedavg', 'fedprox', 'scaffold'])
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--clients_per_round', type=int, default=5)
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--local_epochs', type=int, default=3)
    
    parser.add_argument('--data_distribution', type=str, default='non-iid', choices=['iid', 'non-iid'])
    parser.add_argument('--alpha', type=float, default=0.5)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--server_lr', type=float, default=1.0)
    
    parser.add_argument('--run_all', action='store_true', default=True)
    parser.add_argument('--single', action='store_true')
    
    return parser.parse_args()


def run_single_method(args):
    train_data, test_data = load_ag_news_offline()
    
    print(f"训练样本数: {len(train_data)}")
    print(f"测试样本数: {len(test_data)}")
    
    if args.method != 'centralized':
        print(f"\n创建联邦数据分布 ({args.data_distribution})...")
        client_data = create_federated_data(train_data, args.num_clients, args.data_distribution, args.alpha)
        
        print("\n客户端数据分布:")
        for i, data in enumerate(client_data[:3]):
            labels = [item['label'] for item in data]
            label_dist = np.bincount(labels, minlength=args.num_classes)
            print(f"客户端 {i}: {len(data)} 样本, 类别分布: {label_dist}")
        if len(client_data) > 3:
            print(f"... (共{args.num_clients}个客户端)")
    
    print("\n创建数据加载器...")
    
    if args.method == 'centralized':
        train_loader = create_dataloaders([train_data], args.batch_size, args.model_name, args.max_length)[0]
        train_loaders = None
    else:
        train_loaders = create_dataloaders(client_data, args.batch_size, args.model_name, args.max_length)
        train_loader = None
    
    test_loader = create_dataloaders([test_data], args.batch_size, args.model_name, args.max_length)[0]
    
    print(f"\n初始化 {args.model_name} 模型...")
    global_model = ALBERTClassifier(model_name=args.model_name, num_classes=args.num_classes).to(args.device)
    
    if args.method == 'local':
        return local_training(train_loaders, test_loader, global_model, args)
    elif args.method == 'centralized':
        return centralized_training(train_loader, test_loader, global_model, args)
    elif args.method == 'fedavg':
        return fedavg_training(train_loaders, test_loader, global_model, args)
    elif args.method == 'fedprox':
        return fedprox_training(train_loaders, test_loader, global_model, args)
    elif args.method == 'scaffold':
        return scaffold_training(train_loaders, test_loader, global_model, args)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("\n" + "="*80)
    print("联邦学习 ALBERT 微调 - 完全离线版本 v2")
    print("="*80)
    print(f"设备: {args.device}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODEL_DIR}")
    
    if args.single:
        args.run_all = False
        print(f"\n【单方法运行模式】只运行: {args.method.upper()}")
        run_single_method(args)
    else:
        print("\n【批量运行模式】将依次运行所有5种训练方法")
        
        methods = ['local', 'centralized', 'fedavg', 'fedprox', 'scaffold']
        all_results = {}
        total_start_time = time.time()
        
        for idx, method in enumerate(methods, 1):
            print(f"\n\n{'='*80}")
            print(f"【{idx}/5】开始运行: {method.upper()}")
            print(f"{'='*80}")
            
            current_args = copy.deepcopy(args)
            current_args.method = method
            all_results[method] = run_single_method(current_args)
        
        total_time = time.time() - total_start_time
        print("\n\n" + "="*80)
        print("所有方法训练完成！汇总结果")
        print("="*80)
        print(f"\n{'方法':<15} {'最终测试准确率':<15} {'训练时间':<15}")
        print("-"*50)
        
        for method in methods:
            if method in all_results and all_results[method]:
                result = all_results[method]
                if 'test_accuracy' in result and result['test_accuracy']:
                    print(f"{method:<15} {result['test_accuracy'][-1]*100:>6.2f}%        {result.get('training_time', 0)/60:>6.2f} 分钟")
        
        print("-"*50)
        print(f"总运行时间: {total_time/60:.2f} 分钟")
    
    print("\n实验完成!")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=ResourceWarning)
    warnings.filterwarnings('ignore', message='.*ResourceTracker.*')
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
