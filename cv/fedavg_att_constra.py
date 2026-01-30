import argparse
import copy
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def to_cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def load_state_dict(model: nn.Module, sd_cpu: Dict[str, torch.Tensor]):
    sd = model.state_dict()
    for k in sd.keys():
        sd[k] = sd_cpu[k].to(sd[k].device)
    model.load_state_dict(sd)

@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)

def format_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def weighted_average_state_dict(updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    total = sum(n for _, n in updates)
    keys = updates[0][0].keys()
    out = {}
    for k in keys:
        out[k] = sum(sd[k] * (n / total) for sd, n in updates)
    return out

def attentive_aggregate_state_dict(
    updates: List[Tuple[Dict[str, torch.Tensor], int]], 
    scores: List[float],
    beta: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    使用注意力机制聚合模型参数
    
    Args:
        updates: List of (state_dict, num_samples)
        scores: List of attention scores (e.g., accuracy or inverse loss)
        beta: Temperature parameter for softmax (higher = more uniform, lower = more peaked)
    Aggregated state_dict
    """
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    attention_weights = F.softmax(scores_tensor / beta, dim=0).tolist()
    
    keys = updates[0][0].keys()
    out = {}
    
    for k in keys:
        out[k] = sum(sd[k] * w for (sd, _), w in zip(updates, attention_weights))
    
    return out, attention_weights

def state_dict_add(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], alpha: float = 1.0):
    for k in a.keys():
        a[k] += alpha * b[k]

def state_dict_sub(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] - b[k] for k in a.keys()}

def state_dict_scale(a: Dict[str, torch.Tensor], s: float) -> Dict[str, torch.Tensor]:
    return {k: v * s for k, v in a.items()}

def get_param_names(model: nn.Module) -> List[str]:
    return [name for name, _ in model.named_parameters()]

def print_model_info(model: nn.Module):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型: {model.__class__.__name__}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")


# =========================
# ResNet110 for CIFAR with Feature Extraction
# =========================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        """
        Args:
            x: 输入张量
            return_features: 如果为True，返回(logits, features)；否则只返回logits
        """
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        features = torch.flatten(out, 1)  # 特征向量
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        return logits

def resnet110_cifar100():
    return ResNetCIFAR(BasicBlock, [18, 18, 18], num_classes=100)


# =========================
# Contrastive Loss Functions
# =========================
def contrastive_loss_l2(local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
    """
    基于L2距离的对比损失
    
    L_con = ||f_local(x) - f_global(x)||^2_2
    
    Args:
        local_features: 本地模型的特征 [batch_size, feature_dim]
        global_features: 全局模型的特征 [batch_size, feature_dim]
    
    Returns:
        对比损失标量值
    """
    return torch.mean((local_features - global_features) ** 2)


def contrastive_loss_cosine(local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
    """
    基于余弦相似度的对比损失
    
    L_con = 1 - cosine_similarity(f_local(x), f_global(x))
    
    Args:
        local_features: 本地模型的特征 [batch_size, feature_dim]
        global_features: 全局模型的特征 [batch_size, feature_dim]
    
    Returns:
        对比损失标量值
    """
    # 归一化特征
    local_norm = F.normalize(local_features, p=2, dim=1)
    global_norm = F.normalize(global_features, p=2, dim=1)
    
    # 计算余弦相似度
    cosine_sim = torch.sum(local_norm * global_norm, dim=1)
    
    # 损失 = 1 - 相似度（相似度越高，损失越低）
    return torch.mean(1.0 - cosine_sim)


def contrastive_loss_mse(local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
    """
    基于MSE的对比损失（等价于L2）
    
    Args:
        local_features: 本地模型的特征
        global_features: 全局模型的特征
    
    Returns:
        对比损失标量值
    """
    return F.mse_loss(local_features, global_features)


# =========================
# Data partition (IID / Dirichlet Non-IID)
# =========================
def split_iid(indices: List[int], num_clients: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    idx = indices[:]
    rng.shuffle(idx)
    return [idx[i::num_clients] for i in range(num_clients)]

def split_dirichlet_by_label(
    targets: List[int],
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 50
) -> List[List[int]]:
    rng = random.Random(seed)
    num_classes = len(set(targets))
    class_indices = [[] for _ in range(num_classes)]
    for i, y in enumerate(targets):
        class_indices[y].append(i)
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    for _ in range(10):
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = class_indices[c]
            if not idx_c:
                continue
            props = torch.distributions.Dirichlet(torch.full((num_clients,), alpha)).sample().tolist()
            counts = [int(p * len(idx_c)) for p in props]
            diff = len(idx_c) - sum(counts)
            for i in range(abs(diff)):
                counts[i % num_clients] += 1 if diff > 0 else -1

            start = 0
            for k, cnt in enumerate(counts):
                if cnt <= 0:
                    continue
                client_indices[k].extend(idx_c[start:start+cnt])
                start += cnt

        if min(len(ci) for ci in client_indices) >= min_size:
            return client_indices
    return client_indices


# =========================
# Train primitives with Contrastive Loss
# =========================
@dataclass
class TrainResult:
    loss: float
    acc: float
    orig_loss: float = 0.0
    con_loss: float = 0.0


def train_epoch_with_contrastive(
    local_model: nn.Module,
    global_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer,
    contrastive_lambda: float = 0.1,
    contrastive_type: str = "l2",
    max_grad_norm: float = 0.0
):
    """
    使用对比损失训练一个epoch
    
    L_total = L_orig + λ * L_con
    
    Args:
        local_model: 本地模型（正在训练）
        global_model: 全局模型（固定，用于提取特征）
        loader: 数据加载器
        device: 设备
        opt: 优化器
        contrastive_lambda: 对比损失的权重 λ
        contrastive_type: 对比损失类型 ("l2", "cosine", "mse")
        max_grad_norm: 梯度裁剪阈值
    
    Returns:
        (total_loss, accuracy, orig_loss, con_loss)
    """
    local_model.train()
    global_model.eval()  # 全局模型用于推理，不更新
    
    # 选择对比损失函数
    if contrastive_type == "l2":
        contrastive_fn = contrastive_loss_l2
    elif contrastive_type == "cosine":
        contrastive_fn = contrastive_loss_cosine
    elif contrastive_type == "mse":
        contrastive_fn = contrastive_loss_mse
    else:
        contrastive_fn = contrastive_loss_l2
    
    total_loss_sum, orig_loss_sum, con_loss_sum = 0.0, 0.0, 0.0
    correct, total = 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        opt.zero_grad(set_to_none=True)
        
        # 本地模型前向传播（获取logits和特征）
        logits_local, features_local = local_model(x, return_features=True)
        
        # 全局模型前向传播（只获取特征，不更新梯度）
        with torch.no_grad():
            _, features_global = global_model(x, return_features=True)
        
        # 原始任务损失（分类损失）
        loss_orig = F.cross_entropy(logits_local, y)
        
        # 对比损失（对齐本地和全局特征）
        loss_con = contrastive_fn(features_local, features_global)
        
        # 总损失 = 原始损失 + λ * 对比损失
        loss_total = loss_orig + contrastive_lambda * loss_con
        
        # 反向传播
        loss_total.backward()
        
        # 梯度裁剪
        if max_grad_norm and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)
        
        opt.step()
        
        # 统计
        total_loss_sum += loss_total.item() * x.size(0)
        orig_loss_sum += loss_orig.item() * x.size(0)
        con_loss_sum += loss_con.item() * x.size(0)
        correct += (logits_local.argmax(1) == y).sum().item()
        total += x.size(0)
    
    avg_total_loss = total_loss_sum / max(1, total)
    avg_orig_loss = orig_loss_sum / max(1, total)
    avg_con_loss = con_loss_sum / max(1, total)
    accuracy = correct / max(1, total)
    
    return avg_total_loss, accuracy, avg_orig_loss, avg_con_loss


def train_epoch_standard(model, loader, device, opt, max_grad_norm=0.0):
    """标准训练（无对比损失）"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def train_client_fedavg(
    model, loader, device,
    lr, local_epochs, weight_decay, momentum=0.9
) -> TrainResult:
    """标准FedAvg客户端训练"""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for _ in range(local_epochs):
        loss, acc = train_epoch_standard(model, loader, device, opt)
    return TrainResult(loss=loss, acc=acc)


def train_client_with_contrastive(
    local_model: nn.Module,
    global_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_epochs: int,
    weight_decay: float,
    contrastive_lambda: float = 0.1,
    contrastive_type: str = "l2",
    momentum: float = 0.9
) -> TrainResult:
    """
    使用对比损失的客户端训练
    
    Args:
        local_model: 本地模型
        global_model: 全局模型（用于特征对比）
        loader: 数据加载器
        device: 设备
        lr: 学习率
        local_epochs: 本地训练轮数
        weight_decay: 权重衰减
        contrastive_lambda: 对比损失权重 λ
        contrastive_type: 对比损失类型
        momentum: 动量
    
    Returns:
        TrainResult包含损失和准确率信息
    """
    opt = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    for _ in range(local_epochs):
        total_loss, acc, orig_loss, con_loss = train_epoch_with_contrastive(
            local_model, global_model, loader, device, opt,
            contrastive_lambda=contrastive_lambda,
            contrastive_type=contrastive_type
        )
    
    return TrainResult(
        loss=total_loss,
        acc=acc,
        orig_loss=orig_loss,
        con_loss=con_loss
    )


# =========================
# Methods
# =========================
def run_fedavg(
    trainset, client_splits, test_loader, device,
    rounds, clients_per_round_frac, local_epochs, batch_size, lr, weight_decay, seed,
    use_attentive: bool = False,
    attention_metric: str = "accuracy",
    attention_beta: float = 1.0,
    use_contrastive: bool = False,
    contrastive_lambda: float = 0.1,
    contrastive_type: str = "l2"
):
    """
    FedAvg训练主函数
    
    新增参数:
        use_contrastive: 是否使用对比损失
        contrastive_lambda: 对比损失权重 λ
        contrastive_type: 对比损失类型 ("l2", "cosine", "mse")
    """
    start_time = time.time()
    print("\n" + "="*80)
    
    # 打印训练配置
    method_name = "FedAvg"
    if use_contrastive:
        method_name += " + 对比损失"
    if use_attentive:
        method_name += " + 注意力聚合"
    
    print(f"【开始训练：{method_name}】")
    print("="*80)
    print("策略说明:")
    print("  - 服务器维护全局模型，每轮选择部分客户端参与训练")
    
    if use_contrastive:
        print("  - 使用对比损失对齐本地特征和全局特征")
        print(f"    · 损失函数: L_total = L_orig + λ * L_con")
        print(f"    · 对比损失类型: {contrastive_type}")
        print(f"    · 对比损失权重 λ: {contrastive_lambda}")
        if contrastive_type == "l2":
            print(f"    · 公式: L_con = ||f_local(x) - f_global(x)||²_2")
        elif contrastive_type == "cosine":
            print(f"    · 公式: L_con = 1 - cosine_similarity(f_local, f_global)")
        elif contrastive_type == "mse":
            print(f"    · 公式: L_con = MSE(f_local(x), f_global(x))")
    
    if use_attentive:
        print("  - 使用注意力机制进行模型聚合，根据客户端性能动态分配权重")
        print(f"    · 注意力指标: {attention_metric}")
        print(f"    · 注意力温度参数 β: {attention_beta}")
    else:
        print("  - 客户端本地训练后，服务器对模型参数进行加权平均聚合")
    
    print("  - 优化器: SGD (momentum=0.9)")
    print(f"  - 通信轮数: {rounds}")
    print(f"  - 每轮参与客户端比例: {clients_per_round_frac}")
    print(f"  - 本地训练轮数: {local_epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 学习率: {lr}")
    print(f"  - 权重衰减: {weight_decay}")
    
    global_model = resnet110_cifar100().to(device)
    print_model_info(global_model)
    print("-"*80)
    
    rng = random.Random(seed + 999)
    num_clients = len(client_splits)
    m = max(1, int(math.ceil(num_clients * clients_per_round_frac)))

    global_params = to_cpu_state_dict(global_model)

    for rnd in range(1, rounds + 1):
        chosen = rng.sample(range(num_clients), k=m)
        updates = []
        local_stats = []
        attention_scores = []

        for cid in chosen:
            # 创建本地模型
            local_model = resnet110_cifar100().to(device)
            load_state_dict(local_model, global_params)

            subset = Subset(trainset, client_splits[cid])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

            # 选择训练方式
            if use_contrastive:
                # 创建全局模型副本用于特征对比
                global_model_copy = resnet110_cifar100().to(device)
                load_state_dict(global_model_copy, global_params)
                
                stats = train_client_with_contrastive(
                    local_model, global_model_copy, loader, device,
                    lr, local_epochs, weight_decay,
                    contrastive_lambda=contrastive_lambda,
                    contrastive_type=contrastive_type
                )
            else:
                stats = train_client_fedavg(
                    local_model, loader, device,
                    lr, local_epochs, weight_decay
                )
            
            local_stats.append(stats)
            updates.append((to_cpu_state_dict(local_model), len(subset)))
            
            # 计算注意力分数
            if use_attentive:
                if attention_metric == "accuracy":
                    score = stats.acc
                elif attention_metric == "loss":
                    score = 1.0 / (stats.loss + 1e-8)
                elif attention_metric == "inverse_loss":
                    score = -stats.loss
                else:
                    score = stats.acc
                attention_scores.append(score)

        # 聚合模型参数
        if use_attentive:
            global_params, attn_weights = attentive_aggregate_state_dict(
                updates, attention_scores, beta=attention_beta
            )
        else:
            global_params = weighted_average_state_dict(updates)
            attn_weights = None
            
        load_state_dict(global_model, global_params)

        # 评估
        te_loss, te_acc = eval_model(global_model, test_loader, device)
        avg_tr_loss = sum(s.loss for s in local_stats) / len(local_stats)
        avg_tr_acc = sum(s.acc for s in local_stats) / len(local_stats)
        
        # 打印信息
        prefix = f"[{method_name}]"
        info_str = f"[Round {rnd:03d}] AvgClientLoss {avg_tr_loss:.6f} AvgClientAcc {format_pct(avg_tr_acc)} | TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)}"
        
        # 如果使用对比损失，添加对比损失信息
        if use_contrastive:
            avg_orig_loss = sum(s.orig_loss for s in local_stats) / len(local_stats)
            avg_con_loss = sum(s.con_loss for s in local_stats) / len(local_stats)
            info_str += f" | OrigLoss {avg_orig_loss:.6f} ConLoss {avg_con_loss:.6f}"
        
        # 如果使用注意力，添加注意力权重信息
        if use_attentive and attn_weights is not None:
            max_weight = max(attn_weights)
            min_weight = min(attn_weights)
            avg_weight = sum(attn_weights) / len(attn_weights)
            info_str += f" | AttnW: max={max_weight:.4f} min={min_weight:.4f} avg={avg_weight:.4f}"
        
        print(prefix + info_str)

    te_loss, te_acc = eval_model(global_model, test_loader, device)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"【{method_name}训练完成】耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print(f"{'='*80}\n")
    
    return global_model, te_loss, te_acc, elapsed_time


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    # Federated setup
    parser.add_argument("--num_clients", type=int, default=30)
    parser.add_argument("--clients_per_round", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=3)
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    
    # Data distribution
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration parameter")
    
    # Method selection
    parser.add_argument("--method", type=str, default="fedavg_full", 
                        choices=["fedavg", "fedavg_attentive", "fedavg_contrastive", "fedavg_full"])
    
    # Attentive aggregation parameters
    parser.add_argument("--attention_metric", type=str, default="accuracy", 
                        choices=["accuracy", "loss", "inverse_loss"])
    parser.add_argument("--attention_beta", type=float, default=0.1, 
                        help="Temperature for attention softmax (higher = more uniform)")
    
    # Contrastive learning parameters
    parser.add_argument("--contrastive_lambda", type=float, default=1.0, 
                        help="Weight for contrastive loss")
    parser.add_argument("--contrastive_type", type=str, default="cosine", 
                        choices=["l2", "cosine", "mse"],
                        help="Type of contrastive loss function")
    
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device)
    
    print("\n" + "="*80)
    print("联邦学习训练系统 - CIFAR-100 + ResNet110")
    print("="*80)
    print("\n【系统配置】")
    print(f"  设备: {device}")
    print(f"  随机种子: {args.seed}")
    print(f"  数据目录: {args.data_dir}")
    print(f"\n【联邦学习设置】")
    print(f"  客户端总数: {args.num_clients}")
    print(f"  每轮参与客户端比例: {args.clients_per_round}")
    print(f"  通信轮数: {args.rounds}")
    print(f"  本地训练轮数: {args.local_epochs}")
    print(f"\n【训练超参数】")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"\n【数据划分】")
    print(f"  划分方式: {args.partition}")
    if args.partition == "dirichlet":
        print(f"  Dirichlet α: {args.alpha} ({'较高非IID' if args.alpha < 0.5 else '中等非IID' if args.alpha < 1.0 else '较低非IID'})")
    print(f"\n【训练方法】")
    print(f"  方法: {args.method}")
    
    if args.method in ["fedavg_attentive", "fedavg_full"]:
        print(f"  注意力聚合: 启用")
        print(f"    - 评估指标: {args.attention_metric}")
        print(f"    - 温度参数 β: {args.attention_beta}")
    
    if args.method in ["fedavg_contrastive", "fedavg_full"]:
        print(f"  对比学习: 启用")
        print(f"    - 对比损失类型: {args.contrastive_type}")
        print(f"    - 对比损失权重 λ: {args.contrastive_lambda}")
    
    print("="*80)

    # Load CIFAR-100
    print("\n正在加载 CIFAR-100 数据集...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Partition training data
    print(f"正在按 {args.partition} 方式划分数据到 {args.num_clients} 个客户端...")
    if args.partition == "iid":
        client_splits = split_iid(list(range(len(trainset))), args.num_clients, args.seed)
    else:
        client_splits = split_dirichlet_by_label(
            trainset.targets, args.num_clients, args.alpha, args.seed, min_size=50
        )

    # Print data distribution statistics
    print(f"\n【数据划分统计】")
    sizes = [len(cs) for cs in client_splits]
    print(f"  每个客户端样本数 - 最小: {min(sizes)}, 最大: {max(sizes)}, 平均: {sum(sizes)/len(sizes):.1f}")
    
    # Calculate class distribution for some clients
    if args.partition == "dirichlet":
        print(f"  前5个客户端的类别分布（示例）:")
        for i in range(min(5, args.num_clients)):
            client_labels = [trainset.targets[idx] for idx in client_splits[i]]
            unique_classes = len(set(client_labels))
            print(f"    客户端 {i}: {len(client_labels)} 个样本, {unique_classes} 个类别")

    # Determine method parameters
    use_attentive = args.method in ["fedavg_attentive", "fedavg_full"]
    use_contrastive = args.method in ["fedavg_contrastive", "fedavg_full"]

    # Run training
    model, test_loss, test_acc, elapsed = run_fedavg(
        trainset=trainset,
        client_splits=client_splits,
        test_loader=test_loader,
        device=device,
        rounds=args.rounds,
        clients_per_round_frac=args.clients_per_round,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_attentive=use_attentive,
        attention_metric=args.attention_metric,
        attention_beta=args.attention_beta,
        use_contrastive=use_contrastive,
        contrastive_lambda=args.contrastive_lambda,
        contrastive_type=args.contrastive_type
    )

    # Final results
    print("\n" + "="*80)
    print("【最终结果】")
    print("="*80)
    print(f"方法: {args.method}")
    print(f"测试集损失: {test_loss:.6f}")
    print(f"测试集准确率: {format_pct(test_acc)}")
    print(f"总训练时间: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
    print("="*80 + "\n")

    # Save model
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.method}_alpha{args.alpha}_rounds{args.rounds}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
        'args': vars(args)
    }, save_path)
    print(f"模型已保存到: {save_path}\n")


if __name__ == "__main__":
    main()