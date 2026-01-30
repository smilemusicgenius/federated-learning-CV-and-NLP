"""
=================================================================================
联邦学习 ALBERT 微调 - 完整整合版本 v3.3 FINAL
=================================================================================

支持方法:
- Local Training: 每个客户端独立训练
- Centralized Training: 传统集中式训练
- FedAvg: 联邦平均算法
- FedProx: 带正则化的联邦学习
- SCAFFOLD: 使用控制变量的联邦学习

数据集: AG News (4类新闻分类)
模型: ALBERT (albert-base-v2)

功能特性:
✓ 默认批量运行 - 自动运行所有5种方法并生成对比结果
✓ 完全离线模式 - 检测到本地数据后无需网络
✓ Python 3.12兼容 - 自动处理多进程问题
✓ 自动本地缓存 - 模型和数据集只下载一次
✓ 策略参数打印 - 完整显示实验配置
✓ 时间统计 - 详细记录各模块耗时

使用方法:

  # 【推荐】批量运行所有方法（默认）- 自动运行5种方法并对比结果
  python federated_learning_albert_all_in_one_v3.3_FINAL.py
  python federated_learning_albert_all_in_one_v3.3_FINAL.py --num_rounds 30

  # 只运行单个方法
  python federated_learning_albert_all_in_one_v3.3_FINAL.py --single --method fedavg
  python federated_learning_albert_all_in_one_v3.3_FINAL.py --single --method local

  # 仅下载资源（首次运行推荐）
  python federated_learning_albert_all_in_one_v3.3_FINAL.py --download_only

  # 使用自定义缓存目录
  python federated_learning_albert_all_in_one_v3.3_FINAL.py --cache_dir ./my_cache --download_only
  python federated_learning_albert_all_in_one_v3.3_FINAL.py --cache_dir ./my_cache
    
查看所有参数:
  python federated_learning_albert_all_in_one_v3.3_FINAL.py --help

=================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import random
import copy
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict

# ✅ 关键修复：在导入 datasets 之前先检查并设置离线模式
# 检查本地是否已有数据集，如果有则强制离线模式
_script_dir = os.path.dirname(os.path.abspath(__file__))
_local_data_path = os.path.join(_script_dir, "data", "ag_news")

# ✅ 修复 Python 3.12 多进程兼容性问题 - 必须在最开始设置
# 这个问题来自 datasets 库使用的 multiprocess 包
import sys
import platform

if sys.version_info >= (3, 12):
    # 1. 完全禁用 multiprocess 的资源追踪
    os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    
    # 2. 禁用 datasets 库的多进程功能
    os.environ['HF_DATASETS_DISABLE_PROGRESS_BAR'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 3. 设置 multiprocessing 启动方法（仅 Linux/Mac）
    if platform.system() != 'Windows':
        os.environ['PYTORCH_DATALOADER_MULTIPROCESSING_CONTEXT'] = 'fork'
    
    # 4. 在 Python 层面抑制警告
    import warnings
    warnings.filterwarnings('ignore', category=ResourceWarning)
    warnings.filterwarnings('ignore', message='.*ResourceTracker.*')

# ✅ 关键修复：在导入 datasets 之前先检查并设置离线模式
# 检查本地是否已有数据集，如果有则强制离线模式
_script_dir = os.path.dirname(os.path.abspath(__file__))
_local_data_path = os.path.join(_script_dir, "data", "ag_news")

# 检查是否存在本地数据集（查找.arrow文件）
if os.path.exists(_local_data_path):
    for _root, _dirs, _files in os.walk(_local_data_path):
        _arrow_files = [f for f in _files if f.endswith('.arrow')]
        if len(_arrow_files) >= 2:  # 找到train和test
            print(f"✓ 检测到本地数据集，启用完全离线模式")
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            break

# Transformers & Datasets
from transformers import AlbertModel, AlbertTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


# =================================================================================
# 第一部分: 模型定义
# =================================================================================

class ALBERTClassifier(nn.Module):
    """基于ALBERT的文本分类器"""
    
    # 类变量：跟踪是否已经打印过模型加载信息
    _model_load_printed = {}
    
    def __init__(self, model_name='albert-base-v2', num_classes=4, dropout=0.1, cache_dir=None):
        super(ALBERTClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # 设置模型缓存目录 - 默认使用代码文件所在目录下的model文件夹
        if cache_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(script_dir, "model")
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载预训练的ALBERT模型
        try:
            # 只在第一次加载该模型时打印信息
            cache_key = f"{model_name}_{cache_dir}"
            if cache_key not in ALBERTClassifier._model_load_printed:
                if os.path.exists(os.path.join(cache_dir, model_name)):
                    print(f"从本地缓存加载模型: {cache_dir}")
                else:
                    print(f"首次运行，正在下载模型到: {cache_dir}")
                ALBERTClassifier._model_load_printed[cache_key] = True
            
            self.albert = AlbertModel.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("尝试使用默认缓存...")
            self.albert = AlbertModel.from_pretrained(model_name)
        
        # 获取隐藏层维度
        self.hidden_size = self.albert.config.hidden_size
        
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size] (可选)
            
        Returns:
            如果提供labels，返回 (loss, logits)
            否则返回 logits
        """
        # ALBERT编码
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS]标记的输出
        pooled_output = outputs.pooler_output
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        # 如果提供了标签，计算损失
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        
        return logits
    
    def clone(self):
        """克隆模型"""
        clone_model = ALBERTClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes
        )
        clone_model.load_state_dict(self.state_dict())
        return clone_model


# =================================================================================
# 第二部分: 数据处理
# =================================================================================

class AGNewsDataset(Dataset):
    """AG News 数据集类"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
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


def load_ag_news(cache_dir=None):
    """
    加载AG News数据集（支持完全离线）
    
    Args:
        cache_dir: 数据集缓存目录，如果为None则使用默认缓存（代码文件所在目录下的data文件夹）
        
    Returns:
        train_data: 训练数据列表
        test_data: 测试数据列表
    """
    print("\n正在加载 AG News 数据集...")
    
    # 设置缓存目录 - 默认使用代码文件所在目录下的data文件夹
    if cache_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "data")
    
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    # 检查本地是否已有完整数据集
    local_dataset_path = os.path.join(cache_dir, "ag_news")
    
    # 检查是否存在已下载的数据集
    has_local_data = False
    if os.path.exists(local_dataset_path):
        for root, dirs, files in os.walk(local_dataset_path):
            arrow_files = [f for f in files if f.endswith('.arrow')]
            if len(arrow_files) >= 2:  # 至少有train和test
                has_local_data = True
                print(f"✓ 使用本地数据集: {local_dataset_path}")
                break
    
    if not has_local_data:
        print(f"⚠ 未检测到本地数据集")
        print(f"正在下载到: {cache_dir}")
    
    try:
        # 加载数据集（离线模式已在全局设置）
        dataset = load_dataset('ag_news', cache_dir=cache_dir)
        
        if has_local_data:
            print(f"✓ 离线加载成功")
        else:
            print(f"✓ 下载完成")
            
    except Exception as e:
        print(f"❌ 加载失败: {str(e)}")
        print("尝试使用默认缓存...")
        dataset = load_dataset('ag_news')
    
    # 转换为字典列表格式
    train_data = []
    for item in dataset['train']:
        train_data.append({
            'text': item['text'],
            'label': item['label']
        })
    
    test_data = []
    for item in dataset['test']:
        test_data.append({
            'text': item['text'],
            'label': item['label']
        })
    
    return train_data, test_data


def create_iid_data(data, num_clients):
    """创建IID数据分布"""
    np.random.shuffle(data)
    
    client_data = []
    samples_per_client = len(data) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(data)
        client_data.append(data[start_idx:end_idx])
    
    return client_data


def create_non_iid_data(data, num_clients, alpha=0.5):
    """使用Dirichlet分布创建Non-IID数据"""
    # 按标签分组
    label_to_indices = defaultdict(list)
    for idx, item in enumerate(data):
        label_to_indices[item['label']].append(idx)
    
    num_classes = len(label_to_indices)
    client_indices = [[] for _ in range(num_clients)]
    
    # 对每个类别使用Dirichlet分布分配
    for label, indices in label_to_indices.items():
        np.random.shuffle(indices)
        
        # 生成Dirichlet分布
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        
        # 分割数据
        split_indices = np.split(indices, proportions)
        
        # 分配给客户端
        for client_id, split in enumerate(split_indices):
            client_indices[client_id].extend(split)
    
    # 创建客户端数据
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


def create_dataloaders(client_data_list, batch_size, model_name, max_length=128, cache_dir=None):
    """
    创建数据加载器
    
    Args:
        client_data_list: 客户端数据列表
        batch_size: 批次大小
        model_name: 模型名称
        max_length: 最大序列长度
        cache_dir: 模型缓存目录（默认使用代码文件所在目录下的model文件夹）
    """
    # 设置模型缓存目录 - 默认使用代码文件所在目录下的model文件夹
    if cache_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "model")
    
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    # 加载tokenizer
    try:
        if os.path.exists(os.path.join(cache_dir, model_name)):
            print(f"从本地缓存加载tokenizer: {cache_dir}")
        else:
            print(f"首次运行，正在下载tokenizer到: {cache_dir}")
        
        tokenizer = AlbertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as e:
        print(f"加载tokenizer时出错: {e}")
        print("尝试使用默认缓存...")
        tokenizer = AlbertTokenizer.from_pretrained(model_name)
    
    # 检测系统和Python版本，避免multiprocessing问题
    import platform
    import sys
    
    # Python 3.12+ 在某些环境下有 multiprocess 兼容性问题
    # Windows 系统也需要使用单进程
    python_version = sys.version_info
    if platform.system() == 'Windows' or python_version >= (3, 12):
        num_workers = 0
        if python_version >= (3, 12) and platform.system() != 'Windows':
            print(f"检测到 Python {python_version.major}.{python_version.minor}，使用单进程模式避免兼容性问题")
    else:
        num_workers = 2
    
    dataloaders = []
    for client_data in client_data_list:
        dataset = AGNewsDataset(client_data, tokenizer, max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=False  # 避免资源追踪问题
        )
        dataloaders.append(dataloader)
    
    return dataloaders


# =================================================================================
# 第三部分: 工具函数
# =================================================================================

def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, dataloader, device):
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
            
            loss, logits = model(input_ids, attention_mask, labels)
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch，返回平均损失和准确率"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        loss, logits = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def print_experiment_config(args, method_name):
    """打印实验配置和策略参数"""
    print("\n" + "=" * 80)
    print(f"实验策略参数 - {method_name.upper()}")
    print("=" * 80)
    print(f"训练方法: {method_name}")
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed}")
    print(f"\n模型配置:")
    print(f"  - 模型名称: {args.model_name}")
    print(f"  - 分类类别数: {args.num_classes}")
    print(f"  - 最大序列长度: {args.max_length}")
    
    if method_name != 'centralized':
        print(f"\n联邦学习配置:")
        print(f"  - 客户端总数: {args.num_clients}")
        print(f"  - 每轮参与客户端数: {args.clients_per_round}")
        print(f"  - 通信轮数: {args.num_rounds}")
        print(f"  - 本地训练轮数: {args.local_epochs}")
        print(f"\n数据分布:")
        print(f"  - 分布类型: {args.data_distribution}")
        if args.data_distribution == 'non-iid':
            print(f"  - Dirichlet参数α: {args.alpha}")
    else:
        print(f"\n训练配置:")
        print(f"  - 训练轮数: {args.num_rounds}")
    
    print(f"\n训练超参数:")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 权重衰减: {args.weight_decay}")
    
    if method_name == 'fedprox':
        print(f"\nFedProx特定参数:")
        print(f"  - 正则化系数μ: {args.mu}")
    
    if method_name == 'scaffold':
        print(f"\nSCAFFOLD特定参数:")
        print(f"  - 服务器学习率: {args.server_lr}")
    
    print("=" * 80)


def print_results_summary(results, method_name, elapsed_time):
    """打印训练结果摘要"""
    print("\n" + "=" * 80)
    print(f"训练结果摘要 - {method_name.upper()}")
    print("=" * 80)
    
    best_acc = max(results['test_accuracy'])
    final_acc = results['test_accuracy'][-1]
    best_epoch = results['test_accuracy'].index(best_acc) + 1
    final_loss = results['train_loss'][-1]
    
    print(f"最佳测试准确率: {best_acc:.4f} (第 {best_epoch} 轮)")
    print(f"最终测试准确率: {final_acc:.4f}")
    print(f"最终训练损失: {final_loss:.4f}")
    print(f"总训练时间: {elapsed_time}")
    
    # 打印最后5轮的准确率趋势
    print(f"\n最后5轮测试准确率:")
    for i, acc in enumerate(results['test_accuracy'][-5:], start=len(results['test_accuracy'])-4):
        print(f"  第 {i} 轮: {acc:.4f}")
    
    print("=" * 80)


def format_time(seconds):
    """格式化时间显示"""
    return str(timedelta(seconds=int(seconds)))


# =================================================================================
# 第四部分: 联邦学习算法
# =================================================================================

def local_training(train_loaders, test_loader, model_template, args):
    """本地训练 - 每个客户端独立训练"""
    
    # 打印策略参数
    print("\n" + "="*80)
    print("【开始训练：本地训练 (Local Training)】")
    print("="*80)
    print("\n策略说明:")
    print("- 每个客户端独立训练自己的模型，不进行任何聚合")
    print(f"- 优化器: AdamW")
    print(f"- 客户端数量: {args.num_clients}")
    print(f"- 训练轮数: {args.local_epochs}")
    print(f"- 批次大小: {args.batch_size}")
    print(f"- 学习率: {args.lr}")
    print(f"- 权重衰减: {args.weight_decay}")
    print(f"\n模型: {args.model_name}")
    # 计算并打印参数量
    total_params = sum(p.numel() for p in model_template.parameters())
    trainable_params = sum(p.numel() for p in model_template.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("-" * 80)
    
    start_time = time.time()
    
    results = {
        'train_loss': [],
        'test_accuracy': [],
        'test_loss': []
    }
    
    client_models = [model_template.clone().to(args.device) for _ in range(args.num_clients)]
    
    # 训练所有客户端
    for client_id, (model, train_loader) in enumerate(zip(client_models, train_loaders)):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 本地训练多个epoch
        for epoch in range(args.local_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, args.device)
    
    # 评估所有客户端模型
    test_accuracies = []
    test_losses = []
    
    for client_id, model in enumerate(client_models):
        test_loss, test_acc = evaluate_model(model, test_loader, args.device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        # 按Word文档格式打印
        print(f"[Local][Client {client_id:02d}] TestLoss {test_loss:.6f} TestAcc {test_acc*100:.2f}%")
    
    # 保存平均结果
    results['test_accuracy'].append(np.mean(test_accuracies))
    results['test_loss'].append(np.mean(test_losses))
    
    total_time = time.time() - start_time
    results['training_time'] = total_time
    
    print("\n" + "="*80)
    print(f"【本地训练完成】耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"平均测试准确率: {np.mean(test_accuracies)*100:.2f}%")
    print("="*80)
    
    return results


def centralized_training(train_loader, test_loader, model, args):
    """集中式训练 - 传统训练方式"""
    
    # 打印策略参数
    print("\n" + "="*80)
    print("【开始训练：集中式学习 (Centralized)】")
    print("="*80)
    print("\n策略说明:")
    print("- 使用所有训练数据在单一模型上进行训练")
    print(f"- 优化器: AdamW")
    print(f"- 训练轮数: {args.num_rounds}")
    print(f"- 批次大小: {args.batch_size}")
    print(f"- 学习率: {args.lr}")
    print(f"- 权重衰减: {args.weight_decay}")
    print(f"\n模型: {args.model_name}")
    # 计算并打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("-" * 80)
    
    start_time = time.time()
    
    results = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_epochs = args.num_rounds
    
    for epoch in range(total_epochs):
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, args.device)
        
        # 在测试集上评估
        test_loss, test_acc = evaluate_model(model, test_loader, args.device)
        
        # 保存结果
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        
        # 打印结果（匹配Word文档格式）
        print(f"[Centralized][Epoch {epoch+1:03d}] "
              f"TrainLoss {train_loss:.6f} TrainAcc {train_acc*100:.2f}% | "
              f"TestLoss {test_loss:.6f} TestAcc {test_acc*100:.2f}%")
    
    total_time = time.time() - start_time
    results['training_time'] = total_time
    
    print("\n" + "="*80)
    print(f"【集中式训练完成】耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print("="*80)
    
    return results


def fedavg_training(train_loaders, test_loader, global_model, args):
    """
    FedAvg 联邦平均算法
    公式: w_{t+1} = Σ(n_k/n * w_{t+1}^k)
    """
    
    # 打印策略参数
    print("\n" + "="*80)
    print("【开始训练：联邦平均 (FedAvg)】")
    print("="*80)
    print("\n策略说明:")
    print("- 服务器维护全局模型，每轮选择部分客户端参与训练")
    print("- 客户端本地训练后，服务器对模型参数进行加权平均聚合")
    print(f"- 优化器: AdamW")
    print(f"- 通信轮数: {args.num_rounds}")
    print(f"- 每轮参与客户端比例: {args.clients_per_round}/{args.num_clients}")
    print(f"- 本地训练轮数: {args.local_epochs}")
    print(f"- 批次大小: {args.batch_size}")
    print(f"- 学习率: {args.lr}")
    print(f"- 权重衰减: {args.weight_decay}")
    print(f"\n模型: {args.model_name}")
    # 计算并打印参数量
    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("-" * 80)
    
    start_time = time.time()
    
    results = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'communication_rounds': []
    }
    
    client_data_sizes = [len(loader.dataset) for loader in train_loaders]
    
    for round_idx in range(args.num_rounds):
        selected_clients = np.random.choice(
            args.num_clients,
            args.clients_per_round,
            replace=False
        )
        
        client_weights = []
        round_losses = []
        round_accuracies = []
        
        for client_id in selected_clients:
            local_model = global_model.clone().to(args.device)
            optimizer = torch.optim.AdamW(
                local_model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            
            # 本地训练多个epoch
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train_epoch(local_model, train_loaders[client_id], optimizer, args.device)
                total_loss += train_loss
                total_correct += train_acc * len(train_loaders[client_id].dataset)
                total_samples += len(train_loaders[client_id].dataset)
            
            avg_client_loss = total_loss / args.local_epochs
            avg_client_acc = total_correct / (total_samples * args.local_epochs)
            
            round_losses.append(avg_client_loss)
            round_accuracies.append(avg_client_acc)
            
            client_weights.append({
                'state_dict': copy.deepcopy(local_model.state_dict()),
                'data_size': client_data_sizes[client_id]
            })
        
        # 聚合模型 (FedAvg)
        aggregated_state_dict = {}
        selected_data_size = sum([cw['data_size'] for cw in client_weights])
        
        for key in global_model.state_dict().keys():
            aggregated_state_dict[key] = sum([
                cw['state_dict'][key] * (cw['data_size'] / selected_data_size)
                for cw in client_weights
            ])
        
        global_model.load_state_dict(aggregated_state_dict)
        
        # 计算平均客户端损失和准确率
        avg_client_loss = np.mean(round_losses)
        avg_client_acc = np.mean(round_accuracies)
        
        results['train_loss'].append(avg_client_loss)
        results['train_accuracy'].append(avg_client_acc)
        results['communication_rounds'].append(round_idx + 1)
        
        # 在测试集上评估全局模型
        test_loss, test_acc = evaluate_model(global_model, test_loader, args.device)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        
        # 按Word文档格式打印
        print(f"[FedAvg][Round {round_idx+1:03d}] "
              f"AvgClientLoss {avg_client_loss:.6f} AvgClientAcc {avg_client_acc*100:.2f}% | "
              f"TestLoss {test_loss:.6f} TestAcc {test_acc*100:.2f}%")
    
    total_time = time.time() - start_time
    results['training_time'] = total_time
    
    print("\n" + "="*80)
    print(f"【FedAvg训练完成】耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print("="*80)
    
    return results


def fedprox_training(train_loaders, test_loader, global_model, args):
    """
    FedProx 联邦学习算法（带近端项）
    公式: min F_k(w) + (μ/2)||w - w_t||^2
    """
    
    # 打印策略参数
    print("\n" + "="*80)
    print("【开始训练：联邦近端 (FedProx)】")
    print("="*80)
    print("\n策略说明:")
    print("- 基于FedAvg，在损失函数中添加近端项约束本地更新")
    print(f"- 近端项: (μ/2)||w - w_global||²，其中 μ={args.mu}")
    print("- 帮助处理数据异质性和部分客户端参与问题")
    print(f"- 优化器: AdamW")
    print(f"- 通信轮数: {args.num_rounds}")
    print(f"- 每轮参与客户端比例: {args.clients_per_round}/{args.num_clients}")
    print(f"- 本地训练轮数: {args.local_epochs}")
    print(f"- 批次大小: {args.batch_size}")
    print(f"- 学习率: {args.lr}")
    print(f"- 权重衰减: {args.weight_decay}")
    print(f"\n模型: {args.model_name}")
    # 计算并打印参数量
    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("-" * 80)
    
    start_time = time.time()
    
    results = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'communication_rounds': []
    }
    
    client_data_sizes = [len(loader.dataset) for loader in train_loaders]
    
    for round_idx in range(args.num_rounds):
        selected_clients = np.random.choice(
            args.num_clients,
            args.clients_per_round,
            replace=False
        )
        
        client_weights = []
        round_losses = []
        round_accuracies = []
        
        global_params = copy.deepcopy(global_model.state_dict())
        
        for client_id in selected_clients:
            local_model = global_model.clone().to(args.device)
            optimizer = torch.optim.AdamW(
                local_model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            
            # 本地训练多个epoch，同时计算损失和准确率
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            for epoch in range(args.local_epochs):
                local_model.train()
                for batch in train_loaders[client_id]:
                    input_ids = batch['input_ids'].to(args.device)
                    attention_mask = batch['attention_mask'].to(args.device)
                    labels = batch['labels'].to(args.device)
                    
                    optimizer.zero_grad()
                    
                    # 原始损失
                    loss, logits = local_model(input_ids, attention_mask, labels)
                    
                    # 添加近端项: (μ/2)||w - w_global||^2
                    proximal_term = 0
                    for name, param in local_model.named_parameters():
                        proximal_term += ((param - global_params[name].to(args.device)) ** 2).sum()
                    
                    total_loss_with_prox = loss + (args.mu / 2) * proximal_term
                    
                    total_loss_with_prox.backward()
                    optimizer.step()
                    
                    # 累计原始损失（不包括近端项）
                    total_loss += loss.item()
                    
                    # 计算准确率
                    predictions = torch.argmax(logits, dim=1)
                    total_correct += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
            
            # 计算平均损失和准确率
            avg_client_loss = total_loss / (len(train_loaders[client_id]) * args.local_epochs)
            avg_client_acc = total_correct / total_samples
            
            round_losses.append(avg_client_loss)
            round_accuracies.append(avg_client_acc)
            
            client_weights.append({
                'state_dict': copy.deepcopy(local_model.state_dict()),
                'data_size': client_data_sizes[client_id]
            })
        
        # 聚合模型
        aggregated_state_dict = {}
        selected_data_size = sum([cw['data_size'] for cw in client_weights])
        
        for key in global_model.state_dict().keys():
            aggregated_state_dict[key] = sum([
                cw['state_dict'][key] * (cw['data_size'] / selected_data_size)
                for cw in client_weights
            ])
        
        global_model.load_state_dict(aggregated_state_dict)
        
        # 计算平均客户端损失和准确率
        avg_client_loss = np.mean(round_losses)
        avg_client_acc = np.mean(round_accuracies)
        
        results['train_loss'].append(avg_client_loss)
        results['train_accuracy'].append(avg_client_acc)
        results['communication_rounds'].append(round_idx + 1)
        
        # 在测试集上评估全局模型
        test_loss, test_acc = evaluate_model(global_model, test_loader, args.device)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        
        # 按Word文档格式打印
        print(f"[FedProx][Round {round_idx+1:03d}][mu={args.mu}] "
              f"AvgClientLoss {avg_client_loss:.6f} AvgClientAcc {avg_client_acc*100:.2f}% | "
              f"TestLoss {test_loss:.6f} TestAcc {test_acc*100:.2f}%")
    
    total_time = time.time() - start_time
    results['training_time'] = total_time
    
    print("\n" + "="*80)
    print(f"【FedProx训练完成】耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print("="*80)
    
    return results


def scaffold_training(train_loaders, test_loader, global_model, args):
    """
    SCAFFOLD 联邦学习算法（使用控制变量）
    公式: w_{t+1}^k = w_t - η(g_t^k - c_t^k + c_t)
    """
    
    # 打印策略参数
    print("\n" + "="*80)
    print("【开始训练：SCAFFOLD】")
    print("="*80)
    print("\n策略说明:")
    print("- 使用控制变量减少客户端漂移问题")
    print("- 服务器维护全局控制变量c，每个客户端维护本地控制变量c_i")
    print("- 梯度校正: grad ← grad - c_i + c")
    print("- 通过方差减少技术加速收敛")
    print(f"- 优化器: SGD (momentum=0, 无动量以保持稳定性)")
    print(f"- 通信轮数: {args.num_rounds}")
    print(f"- 每轮参与客户端比例: {args.clients_per_round}/{args.num_clients}")
    print(f"- 本地训练轮数: {args.local_epochs}")
    print(f"- 批次大小: {args.batch_size}")
    print(f"- 学习率: {args.lr}")
    print(f"- 权重衰减: {args.weight_decay}")
    print(f"\n模型: {args.model_name}")
    # 计算并打印参数量
    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("-" * 80)
    
    start_time = time.time()
    
    results = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'communication_rounds': []
    }
    
    client_data_sizes = [len(loader.dataset) for loader in train_loaders]
    
    # 初始化控制变量
    c_global = {key: torch.zeros_like(value) for key, value in global_model.state_dict().items()}
    c_clients = [{key: torch.zeros_like(value) for key, value in global_model.state_dict().items()} 
                 for _ in range(args.num_clients)]
    
    for round_idx in range(args.num_rounds):
        selected_clients = np.random.choice(
            args.num_clients,
            args.clients_per_round,
            replace=False
        )
        
        client_weights = []
        client_control_updates = []
        round_losses = []
        round_accuracies = []
        
        for client_id in selected_clients:
            local_model = global_model.clone().to(args.device)
            
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=args.lr
            )
            
            init_local_params = copy.deepcopy(local_model.state_dict())
            
            # 本地训练多个epoch，同时计算损失和准确率
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            for epoch in range(args.local_epochs):
                local_model.train()
                for batch in train_loaders[client_id]:
                    input_ids = batch['input_ids'].to(args.device)
                    attention_mask = batch['attention_mask'].to(args.device)
                    labels = batch['labels'].to(args.device)
                    
                    optimizer.zero_grad()
                    loss, logits = local_model(input_ids, attention_mask, labels)
                    loss.backward()
                    
                    # SCAFFOLD修正: 添加控制变量
                    with torch.no_grad():
                        for name, param in local_model.named_parameters():
                            if param.grad is not None:
                                correction = c_global[name].to(args.device) - c_clients[client_id][name].to(args.device)
                                param.grad.data.add_(correction)
                    
                    optimizer.step()
                    
                    # 累计损失
                    total_loss += loss.item()
                    
                    # 计算准确率
                    predictions = torch.argmax(logits, dim=1)
                    total_correct += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
            
            # 计算平均损失和准确率
            avg_client_loss = total_loss / (len(train_loaders[client_id]) * args.local_epochs)
            avg_client_acc = total_correct / total_samples
            
            round_losses.append(avg_client_loss)
            round_accuracies.append(avg_client_acc)
            
            # 更新客户端控制变量
            c_client_new = {}
            K = len(train_loaders[client_id]) * args.local_epochs
            
            with torch.no_grad():
                for key in c_clients[client_id].keys():
                    param_diff = init_local_params[key] - local_model.state_dict()[key]
                    c_client_new[key] = c_clients[client_id][key] - c_global[key] + param_diff / (K * args.lr)
            
            client_control_updates.append(c_client_new)
            c_clients[client_id] = c_client_new
            
            client_weights.append({
                'state_dict': copy.deepcopy(local_model.state_dict()),
                'data_size': client_data_sizes[client_id]
            })
        
        # 聚合模型
        aggregated_state_dict = {}
        selected_data_size = sum([cw['data_size'] for cw in client_weights])
        
        for key in global_model.state_dict().keys():
            aggregated_state_dict[key] = sum([
                cw['state_dict'][key] * (cw['data_size'] / selected_data_size)
                for cw in client_weights
            ])
        
        global_model.load_state_dict(aggregated_state_dict)
        
        # 更新全局控制变量
        with torch.no_grad():
            for key in c_global.keys():
                delta_c = sum([
                    (client_control_updates[i][key] - c_clients[selected_clients[i]][key]) 
                    for i in range(len(selected_clients))
                ]) / len(selected_clients)
                c_global[key] = c_global[key] + args.server_lr * delta_c
        
        # 计算平均客户端损失和准确率
        avg_client_loss = np.mean(round_losses)
        avg_client_acc = np.mean(round_accuracies)
        
        results['train_loss'].append(avg_client_loss)
        results['train_accuracy'].append(avg_client_acc)
        results['communication_rounds'].append(round_idx + 1)
        
        # 在测试集上评估全局模型
        test_loss, test_acc = evaluate_model(global_model, test_loader, args.device)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        
        # 按Word文档格式打印
        print(f"[SCAFFOLD][Round {round_idx+1:03d}] "
              f"AvgClientLoss {avg_client_loss:.6f} AvgClientAcc {avg_client_acc*100:.2f}% | "
              f"TestLoss {test_loss:.6f} TestAcc {test_acc*100:.2f}%")
    
    total_time = time.time() - start_time
    results['training_time'] = total_time
    
    print("\n" + "="*80)
    print(f"【SCAFFOLD训练完成】耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print("="*80)
    
    return results


# =================================================================================
# 第五部分: 主程序
# =================================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='联邦学习 ALBERT 微调')
    
    # 基础设置
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    
    # 模型设置
    parser.add_argument('--model_name', type=str, default='albert-base-v2', help='ALBERT模型名称')
    parser.add_argument('--num_classes', type=int, default=4, help='分类类别数')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度')
    
    # 联邦学习设置
    parser.add_argument('--method', type=str, default='fedavg',
                        choices=['local', 'centralized', 'fedavg', 'fedprox', 'scaffold'],
                        help='训练方法')
    parser.add_argument('--num_clients', type=int, default=10, help='客户端数量')
    parser.add_argument('--clients_per_round', type=int, default=5, help='每轮参与的客户端数')
    parser.add_argument('--num_rounds', type=int, default=10, help='通信轮数')
    parser.add_argument('--local_epochs', type=int, default=1, help='本地训练轮数')
    
    # 数据分布设置
    parser.add_argument('--data_distribution', type=str, default='non-iid',
                        choices=['iid', 'non-iid'], help='数据分布类型')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet分布参数(Non-IID)')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    
    # 缓存目录设置
    parser.add_argument('--cache_dir', type=str, default=None, 
                        help='模型和数据集缓存目录（默认: 代码文件所在目录的data和model文件夹）')
    parser.add_argument('--download_only', action='store_true',
                        help='仅下载模型和数据集，不进行训练')
    
    # FedProx特定参数
    parser.add_argument('--mu', type=float, default=0.01, help='FedProx正则化系数')
    
    # SCAFFOLD特定参数
    parser.add_argument('--server_lr', type=float, default=1.0, help='SCAFFOLD服务器学习率')
    
    # 运行模式参数
    parser.add_argument('--run_all', action='store_true', default=True,
                        help='批量运行所有方法（默认行为）: Local → Centralized → FedAvg → FedProx → SCAFFOLD')
    parser.add_argument('--single', action='store_true',
                        help='只运行单个方法（使用 --method 指定，禁用批量运行）')
    
    return parser.parse_args()


def download_and_verify(args):
    """
    下载并验证模型和数据集
    
    Args:
        args: 命令行参数
    """
    print("="*80)
    print("资源下载工具")
    print("="*80)
    
    # 获取代码文件所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置缓存目录 - 默认使用代码文件所在目录下的data和model文件夹
    if args.cache_dir:
        print(f"\n使用自定义缓存目录: {args.cache_dir}")
        dataset_cache = args.cache_dir
        model_cache = args.cache_dir
    else:
        dataset_cache = os.path.join(script_dir, "data")
        model_cache = os.path.join(script_dir, "model")
        print(f"\n使用默认缓存目录（代码文件所在目录）:")
        print(f"  数据集: {dataset_cache}")
        print(f"  模型: {model_cache}")
    
    success = True
    
    # 下载数据集
    print(f"\n{'='*80}")
    print("下载 AG News 数据集")
    print(f"{'='*80}")
    try:
        print(f"缓存目录: {dataset_cache}")
        train_data, test_data = load_ag_news(cache_dir=dataset_cache)
        print(f"✓ 数据集下载成功")
        print(f"  - 训练集: {len(train_data)} 样本")
        print(f"  - 测试集: {len(test_data)} 样本")
    except Exception as e:
        print(f"✗ 数据集下载失败: {e}")
        success = False
    
    # 下载模型
    print(f"\n{'='*80}")
    print(f"下载 {args.model_name} 模型")
    print(f"{'='*80}")
    try:
        print(f"缓存目录: {model_cache}")
        
        # 下载模型
        print("\n下载模型权重...")
        _ = ALBERTClassifier(
            model_name=args.model_name,
            num_classes=args.num_classes,
            cache_dir=model_cache
        )
        print("✓ 模型权重下载成功")
        
        # 下载tokenizer
        print("\n下载tokenizer...")
        from transformers import AlbertTokenizer
        _ = AlbertTokenizer.from_pretrained(args.model_name, cache_dir=model_cache)
        print("✓ Tokenizer下载成功")
        
    except Exception as e:
        print(f"✗ 模型下载失败: {e}")
        success = False
    
    # 验证下载
    print(f"\n{'='*80}")
    print("验证下载")
    print(f"{'='*80}")
    
    print(f"\n检查数据集缓存: {dataset_cache}")
    dataset_path = os.path.join(dataset_cache, "ag_news")
    if os.path.exists(dataset_path):
        print("✓ 数据集缓存目录存在")
    else:
        print("✗ 数据集缓存目录不存在")
    
    print(f"\n检查模型缓存: {model_cache}")
    if os.path.exists(model_cache):
        print("✓ 模型缓存目录存在")
    else:
        print("✗ 模型缓存目录不存在")
    
    # 最终提示
    print(f"\n{'='*80}")
    if success:
        print("✓ 所有资源下载完成！")
        print("\n资源保存位置:")
        print(f"  数据集: {dataset_cache}")
        print(f"  模型: {model_cache}")
        print("\n现在可以运行实验了:")
        if args.cache_dir:
            print(f"  python {os.path.basename(__file__)} --cache_dir {args.cache_dir}")
        else:
            print(f"  python {os.path.basename(__file__)}")
    else:
        print("✗ 部分资源下载失败，请检查网络连接后重试")
    print(f"{'='*80}")
    
    return success


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 处理运行模式参数
    if args.single:
        # 如果指定了 --single，则禁用批量运行
        args.run_all = False
        print("\n" + "="*80)
        print("【单方法运行模式】")
        print(f"只运行: {args.method.upper()}")
        print("="*80)
    elif args.run_all:
        print("\n" + "="*80)
        print("【默认批量运行模式】")
        print("将依次运行所有5种训练方法")
        print("提示: 如需只运行单个方法，请使用 --single 参数")
        print("="*80)
    
    # 如果是仅下载模式，执行下载后退出
    if args.download_only:
        download_and_verify(args)
        return
    
    # 如果是批量运行模式
    if args.run_all:
        print("\n" + "="*80)
        print("批量运行模式：依次执行所有训练方法")
        print("="*80)
        print("执行顺序：")
        print("  1. Local Training")
        print("  2. Centralized Training")
        print("  3. FedAvg")
        print("  4. FedProx")
        print("  5. SCAFFOLD")
        print("="*80)
        
        # 定义要运行的方法列表（按指定顺序）
        methods = ['local', 'centralized', 'fedavg', 'fedprox', 'scaffold']
        all_results = {}
        total_start_time = time.time()
        
        for idx, method in enumerate(methods, 1):
            print(f"\n\n{'='*80}")
            print(f"【{idx}/5】开始运行: {method.upper()}")
            print(f"{'='*80}\n")
            
            # 复制参数并设置当前方法
            current_args = copy.deepcopy(args)
            current_args.method = method
            
            # 运行单个方法
            result = run_single_method(current_args)
            all_results[method] = result
            
            print(f"\n{'='*80}")
            print(f"【{idx}/5】{method.upper()} 完成")
            print(f"{'='*80}")
        
        # 打印汇总结果
        total_time = time.time() - total_start_time
        print("\n\n" + "="*80)
        print("所有方法训练完成！汇总结果")
        print("="*80)
        print(f"\n{'方法':<15} {'最终测试准确率':<15} {'训练时间':<15}")
        print("-"*80)
        
        for method in methods:
            if method in all_results and all_results[method] is not None:
                result = all_results[method]
                if 'test_accuracy' in result and len(result['test_accuracy']) > 0:
                    final_acc = result['test_accuracy'][-1]
                    train_time = result.get('training_time', 0)
                    print(f"{method:<15} {final_acc*100:>6.2f}%        {train_time/60:>6.2f} 分钟")
        
        print("-"*80)
        print(f"总运行时间: {total_time/60:.2f} 分钟 ({total_time/3600:.2f} 小时)")
        print("="*80)
        
    else:
        # 单方法运行模式
        run_single_method(args)
    
    print(f"\n实验完成!")


def run_single_method(args):
    """
    运行单个训练方法
    
    Args:
        args: 参数对象
        
    Returns:
        results: 训练结果字典
    """
    # 加载数据集
    print("\n加载 AG News 数据集...")
    
    # 设置数据集缓存目录
    dataset_cache_dir = args.cache_dir if args.cache_dir else None
    train_data, test_data = load_ag_news(cache_dir=dataset_cache_dir)
    
    print(f"训练样本数: {len(train_data)}")
    print(f"测试样本数: {len(test_data)}")
    
    # 创建联邦数据（如果需要）
    client_data = None
    train_loaders = None
    train_loader = None
    
    if args.method != 'centralized':
        print(f"\n创建联邦数据分布 ({args.data_distribution})...")
        client_data = create_federated_data(
            train_data,
            num_clients=args.num_clients,
            distribution=args.data_distribution,
            alpha=args.alpha
        )
        
        # 打印前3个客户端的数据分布
        print("\n客户端数据分布:")
        for i, data in enumerate(client_data[:3]):
            labels = [item['label'] for item in data]
            label_dist = np.bincount(labels, minlength=args.num_classes)
            print(f"客户端 {i}: {len(data)} 样本, 类别分布: {label_dist}")
        if len(client_data) > 3:
            print(f"... (共{args.num_clients}个客户端)")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    
    # 设置模型缓存目录
    model_cache_dir = args.cache_dir if args.cache_dir else None
    
    if args.method == 'centralized':
        train_loader = create_dataloaders(
            [train_data], 
            args.batch_size, 
            args.model_name,
            args.max_length,
            cache_dir=model_cache_dir
        )[0]
    else:
        train_loaders = create_dataloaders(
            client_data,
            args.batch_size,
            args.model_name,
            args.max_length,
            cache_dir=model_cache_dir
        )
    
    test_loader = create_dataloaders(
        [test_data],
        args.batch_size,
        args.model_name,
        args.max_length,
        cache_dir=model_cache_dir
    )[0]
    
    # 初始化全局模型
    print(f"\n初始化 {args.model_name} 模型...")
    global_model = ALBERTClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        cache_dir=model_cache_dir
    ).to(args.device)
    
    # 选择并执行训练方法
    results = None
    
    if args.method == 'local':
        results = local_training(
            train_loaders=train_loaders,
            test_loader=test_loader,
            model_template=global_model,
            args=args
        )
    elif args.method == 'centralized':
        results = centralized_training(
            train_loader=train_loader,
            test_loader=test_loader,
            model=global_model,
            args=args
        )
    elif args.method == 'fedavg':
        results = fedavg_training(
            train_loaders=train_loaders,
            test_loader=test_loader,
            global_model=global_model,
            args=args
        )
    elif args.method == 'fedprox':
        results = fedprox_training(
            train_loaders=train_loaders,
            test_loader=test_loader,
            global_model=global_model,
            args=args
        )
    elif args.method == 'scaffold':
        results = scaffold_training(
            train_loaders=train_loaders,
            test_loader=test_loader,
            global_model=global_model,
            args=args
        )
    
    return results


if __name__ == '__main__':
    # ✅ 修复 Python 3.12 多进程兼容性问题
    import sys
    import platform
    import warnings
    
    # 1. 完全抑制资源警告
    if sys.version_info >= (3, 12):
        warnings.filterwarnings('ignore', category=ResourceWarning)
        warnings.filterwarnings('ignore', message='.*ResourceTracker.*')
        warnings.filterwarnings('ignore', message='.*multiprocess.*')
        
        # 2. 设置 multiprocessing 启动方法
        if platform.system() != 'Windows':
            try:
                import multiprocessing as mp
                mp.set_start_method('fork', force=True)
            except RuntimeError:
                pass  # 已经设置过
    
    # 3. 捕获退出时的错误
    try:
        main()
    except SystemExit:
        pass  # 正常退出
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 4. 强制清理资源（Python 3.12）
        if sys.version_info >= (3, 12):
            try:
                import gc
                gc.collect()
            except:
                pass


