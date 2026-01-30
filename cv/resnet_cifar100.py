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

def train_client_fedprox(
    model, loader, device,
    lr, local_epochs, weight_decay, mu: float,
    global_params_cpu: Dict[str, torch.Tensor],
    momentum=0.9
) -> TrainResult:
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.train()

    # Cache global tensors on device for prox computation
    global_on_device = {k: v.to(device) for k, v in global_params_cpu.items()}

    total_loss, correct, total = 0.0, 0, 0
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            # Prox term: (mu/2)*||w - w_global||^2
            if mu and mu > 0:
                prox = 0.0
                for (name, p) in model.named_parameters():
                    prox = prox + (p - global_on_device[name]).pow(2).sum()
                loss = loss + 0.5 * mu * prox

            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

    return TrainResult(loss=total_loss / max(1, total), acc=correct / max(1, total))

def train_client_scaffold(
    model, loader, device,
    lr, local_epochs, weight_decay,
    c_global: Dict[str, torch.Tensor],
    c_client: Dict[str, torch.Tensor],
):
    """
    SCAFFOLD client update using gradient correction:
      grad <- grad - c_client + c_global
    We use SGD without momentum here for stability.
    Returns:
      w_local_cpu, n_samples, client_train_loss/acc, delta_c (for server update), num_steps
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)

    # Move controls to device
    c_g = {k: v.to(device) for k, v in c_global.items()}
    c_i = {k: v.to(device) for k, v in c_client.items()}

    total_loss, correct, total = 0.0, 0, 0
    steps = 0

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            # gradient correction
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    p.grad = p.grad - c_i[name] + c_g[name]

            opt.step()
            steps += 1

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

    # Compute delta_c using paper formula:
    # c_i_new = c_i - c + (w_global - w_local)/(K*lr)
    # We'll compute on CPU outside: need w_global and w_local
    return TrainResult(loss=total_loss / max(1, total), acc=correct / max(1, total)), steps


# =========================
# Methods
# =========================
def run_centralized(trainset, test_loader, device, epochs, batch_size, lr, weight_decay):
    start_time = time.time()
    print("\n" + "="*80)
    print("【开始训练：集中式学习 (Centralized)】")
    print("="*80)
    print("策略说明:")
    print("  - 使用所有训练数据在单一模型上进行训练")
    print("  - 优化器: SGD (momentum=0.9)")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 学习率: {lr}")
    print(f"  - 权重衰减: {weight_decay}")
    
    model = resnet110_cifar100().to(device)
    print_model_info(model)
    print("-"*80)
    
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch_standard(model, loader, device, opt)
        te_loss, te_acc = eval_model(model, test_loader, device)
        print(f"[Centralized][Epoch {ep:03d}] TrainLoss {tr_loss:.6f} TrainAcc {format_pct(tr_acc)} | TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)}")

    te_loss, te_acc = eval_model(model, test_loader, device)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"【集中式训练完成】耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print(f"{'='*80}\n")
    
    return model, te_loss, te_acc, elapsed_time

def run_local(trainset, client_splits, test_loader, device, epochs, batch_size, lr, weight_decay):
    start_time = time.time()
    print("\n" + "="*80)
    print("【开始训练：本地训练 (Local Training)】")
    print("="*80)
    print("策略说明:")
    print("  - 每个客户端独立训练自己的模型，不进行任何聚合")
    print("  - 优化器: SGD (momentum=0.9)")
    print(f"  - 客户端数量: {len(client_splits)}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 学习率: {lr}")
    print(f"  - 权重衰减: {weight_decay}")
    
    model = resnet110_cifar100().to(device)
    print_model_info(model)
    print("-"*80)
    
    results = []
    models = []
    for cid, idxs in enumerate(client_splits):
        model = resnet110_cifar100().to(device)
        subset = Subset(trainset, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        for _ in range(epochs):
            train_epoch_standard(model, loader, device, opt)

        # 评估训练集和测试集
        tr_loss, tr_acc = eval_model(model, loader, device)
        te_loss, te_acc = eval_model(model, test_loader, device)
        results.append((te_loss, te_acc))
        models.append(model)
        print(f"[Local][Client {cid:02d}] TrainLoss {tr_loss:.6f} TrainAcc {format_pct(tr_acc)} | TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)}")

    accs = [a for _, a in results]
    losses = [l for l, _ in results]
    summary = {
        "mean_loss": sum(losses)/len(losses),
        "mean_acc": sum(accs)/len(accs),
        "best_acc": max(accs),
        "worst_acc": min(accs),
    }
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"【本地训练完成】耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print(f"{'='*80}\n")
    
    return models, summary, elapsed_time

def run_fedavg(
    trainset, client_splits, test_loader, device,
    rounds, clients_per_round_frac, local_epochs, batch_size, lr, weight_decay, seed
):
    start_time = time.time()
    print("\n" + "="*80)
    print("【开始训练：联邦平均 (FedAvg)】")
    print("="*80)
    print("策略说明:")
    print("  - 服务器维护全局模型，每轮选择部分客户端参与训练")
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

        for cid in chosen:
            local_model = resnet110_cifar100().to(device)
            load_state_dict(local_model, global_params)

            subset = Subset(trainset, client_splits[cid])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

            stats = train_client_fedavg(local_model, loader, device, lr, local_epochs, weight_decay)
            local_stats.append(stats)
            updates.append((to_cpu_state_dict(local_model), len(subset)))

        global_params = weighted_average_state_dict(updates)
        load_state_dict(global_model, global_params)

        te_loss, te_acc = eval_model(global_model, test_loader, device)
        avg_tr_loss = sum(s.loss for s in local_stats) / len(local_stats)
        avg_tr_acc = sum(s.acc for s in local_stats) / len(local_stats)
        print(f"[FedAvg][Round {rnd:03d}] AvgClientLoss {avg_tr_loss:.6f} AvgClientAcc {format_pct(avg_tr_acc)} | TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)}")

    te_loss, te_acc = eval_model(global_model, test_loader, device)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"【FedAvg训练完成】耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print(f"{'='*80}\n")
    
    return global_model, te_loss, te_acc, elapsed_time

def run_fedprox(
    trainset, client_splits, test_loader, device,
    rounds, clients_per_round_frac, local_epochs, batch_size, lr, weight_decay, seed,
    mu
):
    start_time = time.time()
    print("\n" + "="*80)
    print("【开始训练：联邦近端 (FedProx)】")
    print("="*80)
    print("策略说明:")
    print("  - 基于FedAvg，在损失函数中添加近端项约束本地更新")
    print(f"  - 近端项: (μ/2)||w - w_global||²，其中 μ={mu}")
    print("  - 帮助处理数据异质性和部分客户端参与问题")
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
    
    rng = random.Random(seed + 1999)
    num_clients = len(client_splits)
    m = max(1, int(math.ceil(num_clients * clients_per_round_frac)))

    global_params = to_cpu_state_dict(global_model)

    for rnd in range(1, rounds + 1):
        chosen = rng.sample(range(num_clients), k=m)
        updates = []
        local_stats = []

        for cid in chosen:
            local_model = resnet110_cifar100().to(device)
            load_state_dict(local_model, global_params)

            subset = Subset(trainset, client_splits[cid])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

            stats = train_client_fedprox(
                local_model, loader, device,
                lr, local_epochs, weight_decay,
                mu=mu, global_params_cpu=global_params
            )
            local_stats.append(stats)
            updates.append((to_cpu_state_dict(local_model), len(subset)))

        global_params = weighted_average_state_dict(updates)
        load_state_dict(global_model, global_params)

        te_loss, te_acc = eval_model(global_model, test_loader, device)
        avg_tr_loss = sum(s.loss for s in local_stats) / len(local_stats)
        avg_tr_acc = sum(s.acc for s in local_stats) / len(local_stats)
        print(f"[FedProx][Round {rnd:03d}][mu={mu}] AvgClientLoss {avg_tr_loss:.6f} AvgClientAcc {format_pct(avg_tr_acc)} | TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)}")

    te_loss, te_acc = eval_model(global_model, test_loader, device)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"【FedProx训练完成】耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print(f"{'='*80}\n")
    
    return global_model, te_loss, te_acc, elapsed_time

def run_scaffold(
    trainset, client_splits, test_loader, device,
    rounds, clients_per_round_frac, local_epochs, batch_size, lr, weight_decay, seed
):
    """
    SCAFFOLD (basic version):
      - Server keeps (w, c)
      - Each client keeps c_i
      - Each round:
          * send (w, c) to selected clients
          * client trains with gradient correction
          * client computes c_i_new and sends delta_c
          * server aggregates w like FedAvg (weighted) and updates c by avg(delta_c)
    """
    start_time = time.time()
    print("\n" + "="*80)
    print("【开始训练：SCAFFOLD】")
    print("="*80)
    print("策略说明:")
    print("  - 使用控制变量减少客户端漂移问题")
    print("  - 服务器维护全局控制变量c，每个客户端维护本地控制变量c_i")
    print("  - 梯度校正: grad ← grad - c_i + c")
    print("  - 通过方差减少技术加速收敛")
    print("  - 优化器: SGD (momentum=0, 无动量以保持稳定性)")
    print(f"  - 通信轮数: {rounds}")
    print(f"  - 每轮参与客户端比例: {clients_per_round_frac}")
    print(f"  - 本地训练轮数: {local_epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 学习率: {lr}")
    print(f"  - 权重衰减: {weight_decay}")
    
    global_model = resnet110_cifar100().to(device)
    print_model_info(global_model)
    print("-"*80)
    
    rng = random.Random(seed + 2999)
    num_clients = len(client_splits)
    m = max(1, int(math.ceil(num_clients * clients_per_round_frac)))

    w_global = to_cpu_state_dict(global_model)

    param_names = get_param_names(global_model)

    # SCAFFOLD controls only for trainable parameters (avoid BN buffers like num_batches_tracked)
    c_global = {k: torch.zeros_like(w_global[k], dtype=torch.float32) for k in param_names}
    c_clients = [{k: torch.zeros_like(w_global[k], dtype=torch.float32) for k in param_names}
                for _ in range(num_clients)]


    for rnd in range(1, rounds + 1):
        chosen = rng.sample(range(num_clients), k=m)

        updates_w = []
        deltas_c = []
        local_stats = []

        for cid in chosen:
            local_model = resnet110_cifar100().to(device)
            load_state_dict(local_model, w_global)

            subset = Subset(trainset, client_splits[cid])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

            stats, steps = train_client_scaffold(
                local_model, loader, device,
                lr=lr, local_epochs=local_epochs, weight_decay=weight_decay,
                c_global=c_global, c_client=c_clients[cid]
            )
            local_stats.append(stats)

            w_local = to_cpu_state_dict(local_model)
            updates_w.append((w_local, len(subset)))

            # c_i_new = c_i - c + (w_global - w_local) / (K*lr)
            # K = number of optimizer steps
            K = max(1, steps)
            factor = 1.0 / (K * lr)

            c_i_old = c_clients[cid]
            c_i_new = {}
            for k in c_i_old.keys():
                c_i_new[k] = c_i_old[k] - c_global[k] + (w_global[k] - w_local[k]) * factor

            delta_c = {k: c_i_new[k] - c_i_old[k] for k in c_i_old.keys()}
            c_clients[cid] = c_i_new
            deltas_c.append(delta_c)

        # aggregate w (FedAvg-style weighted)
        w_global = weighted_average_state_dict(updates_w)
        load_state_dict(global_model, w_global)

        # update c_global by average of delta_c over selected clients
        avg_delta_c = {k: torch.zeros_like(v) for k, v in c_global.items()}
        for dc in deltas_c:
            state_dict_add(avg_delta_c, dc, alpha=1.0)
        avg_delta_c = state_dict_scale(avg_delta_c, 1.0 / len(deltas_c))
        state_dict_add(c_global, avg_delta_c, alpha=1.0)

        te_loss, te_acc = eval_model(global_model, test_loader, device)
        avg_tr_loss = sum(s.loss for s in local_stats) / len(local_stats)
        avg_tr_acc = sum(s.acc for s in local_stats) / len(local_stats)
        print(f"[SCAFFOLD][Round {rnd:03d}] AvgClientLoss {avg_tr_loss:.6f} AvgClientAcc {format_pct(avg_tr_acc)} | TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)}")

    te_loss, te_acc = eval_model(global_model, test_loader, device)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"【SCAFFOLD训练完成】耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
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
    parser.add_argument("--clients_per_round", type=float, default=1.0)  # 使用所有客户端以获得更好的性能
    parser.add_argument("--rounds", type=int, default=50)  # 通信轮数
    parser.add_argument("--local_epochs", type=int, default=3)  # 每轮本地训练3个epoch

    # Centralized/Local epochs (for fair comparison, should equal rounds*local_epochs)
    parser.add_argument("--central_epochs", type=int, default=150)  # 默认等于 rounds(50) × local_epochs(3)
    parser.add_argument("--local_epochs_total", type=int, default=150)  # 保持与联邦学习相同的总训练量

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.03)  # ResNet110需要更大的学习率
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # Partition
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)

    # FedProx
    parser.add_argument("--fedprox_mu", type=float, default=0.05)  # 对于中等非IID（alpha=0.3）使用更强的正则化

    # Which methods to run
    parser.add_argument("--methods", type=str, default="all",
                        help="Comma-separated: local,central,fedavg,fedprox,scaffold or 'all'")

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

    if args.methods.strip().lower() == "all":
        methods = ["local", "central", "fedavg", "fedprox", "scaffold"]
    else:
        methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

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
    print(f"运行方法: {', '.join(methods)}")
    print("="*80)

    # 记录总开始时间
    total_start_time = time.time()
    
    summary_rows = []
    time_records = {}  # 记录每个方法的时间

    # 1) Local
    if "local" in methods:
        _, local_summary, elapsed = run_local(
            trainset, client_splits, test_loader, device,
            epochs=args.local_epochs_total,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay
        )
        time_records["Local"] = elapsed
        summary_rows.append({
            "Method": "Local (mean over clients)",
            "TestAcc": local_summary["mean_acc"],
            "TestLoss": local_summary["mean_loss"],
            "Note": f"best={format_pct(local_summary['best_acc'])}, worst={format_pct(local_summary['worst_acc'])}"
        })

    # 2) Centralized
    if "central" in methods:
        _, te_loss, te_acc, elapsed = run_centralized(
            trainset, test_loader, device,
            epochs=args.central_epochs,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay
        )
        time_records["Centralized"] = elapsed
        summary_rows.append({"Method": "Centralized", "TestAcc": te_acc, "TestLoss": te_loss, "Note": ""})

    # 3) FedAvg
    if "fedavg" in methods:
        _, te_loss, te_acc, elapsed = run_fedavg(
            trainset, client_splits, test_loader, device,
            rounds=args.rounds, clients_per_round_frac=args.clients_per_round,
            local_epochs=args.local_epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay, seed=args.seed
        )
        time_records["FedAvg"] = elapsed
        summary_rows.append({"Method": "FedAvg", "TestAcc": te_acc, "TestLoss": te_loss, "Note": ""})

    # 4) FedProx
    if "fedprox" in methods:
        _, te_loss, te_acc, elapsed = run_fedprox(
            trainset, client_splits, test_loader, device,
            rounds=args.rounds, clients_per_round_frac=args.clients_per_round,
            local_epochs=args.local_epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
            mu=args.fedprox_mu
        )
        time_records["FedProx"] = elapsed
        summary_rows.append({"Method": f"FedProx(mu={args.fedprox_mu})", "TestAcc": te_acc, "TestLoss": te_loss, "Note": ""})

    # 5) SCAFFOLD
    if "scaffold" in methods:
        _, te_loss, te_acc, elapsed = run_scaffold(
            trainset, client_splits, test_loader, device,
            rounds=args.rounds, clients_per_round_frac=args.clients_per_round,
            local_epochs=args.local_epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay, seed=args.seed
        )
        time_records["SCAFFOLD"] = elapsed
        summary_rows.append({"Method": "SCAFFOLD", "TestAcc": te_acc, "TestLoss": te_loss, "Note": "SGD(no momentum) w/ control variates"})

    # Print comparison
    print("\n" + "=" * 80)
    print("最终对比结果（相同数据划分和超参数）")
    print("=" * 80)
    # Pretty table
    colw = {"Method": 26, "TestAcc": 10, "TestLoss": 10, "Note": 28}
    header = f"{'Method':<{colw['Method']}}  {'TestAcc':<{colw['TestAcc']}}  {'TestLoss':<{colw['TestLoss']}}  {'Note':<{colw['Note']}}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        line = f"{r['Method']:<{colw['Method']}}  {format_pct(r['TestAcc']):<{colw['TestAcc']}}  {r['TestLoss']:<{colw['TestLoss']}.4f}  {r['Note']:<{colw['Note']}}"
        print(line)
    
    # 计算总时间
    total_elapsed_time = time.time() - total_start_time
    
    # Print time statistics
    print("\n" + "=" * 80)
    print("时间统计报告")
    print("=" * 80)
    
    if time_records:
        # 打印每个方法的时间
        time_colw = {"Method": 26, "Time(s)": 15, "Time(m)": 15, "Time(h)": 15}
        time_header = f"{'方法':<{time_colw['Method']}}  {'时间(秒)':<{time_colw['Time(s)']}}  {'时间(分钟)':<{time_colw['Time(m)']}}  {'时间(小时)':<{time_colw['Time(h)']}}"
        print(time_header)
        print("-" * len(time_header))
        
        for method_name, elapsed in time_records.items():
            hours = elapsed / 3600
            minutes = elapsed / 60
            time_line = f"{method_name:<{time_colw['Method']}}  {elapsed:<{time_colw['Time(s)']}.2f}  {minutes:<{time_colw['Time(m)']}.2f}  {hours:<{time_colw['Time(h)']}.4f}"
            print(time_line)
        
        print("-" * len(time_header))
        
        # 打印总时间
        total_hours = total_elapsed_time / 3600
        total_minutes = total_elapsed_time / 60
        total_line = f"{'总计':<{time_colw['Method']}}  {total_elapsed_time:<{time_colw['Time(s)']}.2f}  {total_minutes:<{time_colw['Time(m)']}.2f}  {total_hours:<{time_colw['Time(h)']}.4f}"
        print(total_line)
        
        print("=" * 80)
        
        # 打印可读的总结
        print(f"\n✅ 所有训练完成！")
        print(f"📊 运行了 {len(time_records)} 个方法")
        print(f"⏱️  总耗时: {total_elapsed_time:.2f}秒 = {total_minutes:.2f}分钟 = {total_hours:.2f}小时")
        
        # 找出最快和最慢的方法
        if len(time_records) > 1:
            fastest_method = min(time_records.items(), key=lambda x: x[1])
            slowest_method = max(time_records.items(), key=lambda x: x[1])
            print(f"🚀 最快方法: {fastest_method[0]} ({fastest_method[1]/60:.2f}分钟)")
            print(f"🐌 最慢方法: {slowest_method[0]} ({slowest_method[1]/60:.2f}分钟)")
        
        print("=" * 80)


if __name__ == "__main__":
    main()
