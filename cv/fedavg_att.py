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
    
    Returns:
        Aggregated state_dict
    """
    # 计算注意力权重 (使用softmax归一化)
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
# ResNet100 for CIFAR
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def resnet110_cifar100():
    return ResNetCIFAR(BasicBlock, [18, 18, 18], num_classes=100)


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

    for _ in range(10):  # best-effort to satisfy min_size
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
# Train primitives
# =========================
@dataclass
class TrainResult:
    loss: float
    acc: float

def train_epoch_standard(model, loader, device, opt, max_grad_norm=0.0):
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
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for _ in range(local_epochs):
        loss, acc = train_epoch_standard(model, loader, device, opt)
    return TrainResult(loss=loss, acc=acc)


# =========================
# Methods
# =========================
def run_fedavg(
    trainset, client_splits, test_loader, device,
    rounds, clients_per_round_frac, local_epochs, batch_size, lr, weight_decay, seed,
    use_attentive: bool = False,
    attention_metric: str = "accuracy",
    attention_beta: float = 1.0
):
    start_time = time.time()
    print("\n" + "="*80)
    if use_attentive:
        print("【开始训练：联邦平均 + 注意力聚合 (FedAvg + Attentive Aggregation)】")
    else:
        print("【开始训练：联邦平均 (FedAvg)】")
    print("="*80)
    print("策略说明:")
    print("  - 服务器维护全局模型，每轮选择部分客户端参与训练")
    if use_attentive:
        print("  - 使用注意力机制进行模型聚合，根据客户端性能动态分配权重")
        print(f"  - 注意力指标: {attention_metric}")
        print(f"  - 注意力温度参数 β: {attention_beta}")
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
            local_model = resnet110_cifar100().to(device)
            load_state_dict(local_model, global_params)

            subset = Subset(trainset, client_splits[cid])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

            stats = train_client_fedavg(local_model, loader, device, lr, local_epochs, weight_decay)
            local_stats.append(stats)
            updates.append((to_cpu_state_dict(local_model), len(subset)))
            
            # 计算注意力分数
            if use_attentive:
                if attention_metric == "accuracy":
                    # 使用准确率作为注意力分数
                    score = stats.acc
                elif attention_metric == "loss":
                    # 使用损失的倒数作为注意力分数（损失越小，分数越高）
                    score = 1.0 / (stats.loss + 1e-8)
                elif attention_metric == "inverse_loss":
                    # 使用负损失（损失越小，分数越高）
                    score = -stats.loss
                else:
                    score = stats.acc  # 默认使用准确率
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

        te_loss, te_acc = eval_model(global_model, test_loader, device)
        avg_tr_loss = sum(s.loss for s in local_stats) / len(local_stats)
        avg_tr_acc = sum(s.acc for s in local_stats) / len(local_stats)
        
        # 打印信息
        if use_attentive and attn_weights is not None:
            # 计算注意力权重的统计信息
            max_weight = max(attn_weights)
            min_weight = min(attn_weights)
            avg_weight = sum(attn_weights) / len(attn_weights)
            print(f"[FedAvg+Attn][Round {rnd:03d}] AvgClientLoss {avg_tr_loss:.6f} AvgClientAcc {format_pct(avg_tr_acc)} | "
                  f"TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)} | "
                  f"AttnWeight: max={max_weight:.4f} min={min_weight:.4f} avg={avg_weight:.4f}")
        else:
            print(f"[FedAvg][Round {rnd:03d}] AvgClientLoss {avg_tr_loss:.6f} AvgClientAcc {format_pct(avg_tr_acc)} | "
                  f"TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)}")

    te_loss, te_acc = eval_model(global_model, test_loader, device)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    if use_attentive:
        print(f"【FedAvg+注意力聚合训练完成】耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    else:
        print(f"【FedAvg训练完成】耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print(f"{'='*80}\n")
    
    return global_model, te_loss, te_acc, elapsed_time


# =========================
# Main: run all + compare
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

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # Partition
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    
    # Attentive Aggregation
    parser.add_argument("--attention_metric", type=str, default="accuracy", 
                        choices=["accuracy", "loss", "inverse_loss"],
                        help="注意力分数计算方式: accuracy, loss, inverse_loss")
    parser.add_argument("--attention_beta", type=float, default=0.1,
                        help="注意力温度参数 (值越大权重分布越均匀，越小越集中)")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # CIFAR-100 transforms
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

    # client split uses targets (no need to remove augmentation)
    all_train_indices = list(range(len(trainset)))
    if args.partition == "iid":
        client_splits = split_iid(all_train_indices, args.num_clients, seed=args.seed)
    else:
        targets = trainset.targets
        client_splits = split_dirichlet_by_label(
            targets=targets,
            num_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
            min_size=max(10, len(trainset) // (args.num_clients * 20))
        )

    print("\n" + "="*80)
    print("实验配置信息")
    print("="*80)
    print(f"设备: {device}")
    print(f"数据集: CIFAR-100")
    print(f"数据划分方式: {args.partition}" + (f" (alpha={args.dirichlet_alpha})" if args.partition=="dirichlet" else ""))
    print(f"客户端数量: {args.num_clients}")
    print(f"通信轮数: {args.rounds}")
    print(f"每轮客户端参与比例: {args.clients_per_round}")
    print(f"本地训练轮数: {args.local_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"运行方法: FedAvg" + (" + 注意力聚合" if args.use_attentive else ""))
    if args.use_attentive:
        print(f"注意力指标: {args.attention_metric}")
        print(f"注意力温度: {args.attention_beta}")
    print("="*80)

    # 记录总开始时间
    total_start_time = time.time()
    
    # Run FedAvg (with or without attention)
    _, te_loss, te_acc, elapsed = run_fedavg(
        trainset, client_splits, test_loader, device,
        rounds=args.rounds, clients_per_round_frac=args.clients_per_round,
        local_epochs=args.local_epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        use_attentive=True,
        attention_metric=args.attention_metric,
        attention_beta=args.attention_beta
    )

    # Print final results
    print("\n" + "=" * 80)
    print("最终结果")
    print("=" * 80)
    print(f"方法: FedAvg" + (" + 注意力聚合" if args.use_attentive else ""))
    print(f"测试准确率: {format_pct(te_acc)}")
    print(f"测试损失: {te_loss:.4f}")
    
    # 计算总时间
    total_elapsed_time = time.time() - total_start_time
    
    # Print time statistics
    print("\n" + "=" * 80)
    print("时间统计报告")
    print("=" * 80)
    
    hours = elapsed / 3600
    minutes = elapsed / 60
    print(f"训练耗时: {elapsed:.2f}秒 = {minutes:.2f}分钟 = {hours:.4f}小时")
    
    total_hours = total_elapsed_time / 3600
    total_minutes = total_elapsed_time / 60
    print(f"总耗时: {total_elapsed_time:.2f}秒 = {total_minutes:.2f}分钟 = {total_hours:.4f}小时")
    
    print("=" * 80)
    print(f"\n✅ FedAvg训练完成！")
    print(f"⏱️  总耗时: {total_elapsed_time:.2f}秒 = {total_minutes:.2f}分钟 = {total_hours:.2f}小时")
    print("=" * 80)


if __name__ == "__main__":
    main()
