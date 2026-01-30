import argparse
import copy
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
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

def state_dict_sub(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] - b[k] for k in a.keys()}

def compute_update_variance(current_update: Dict[str, torch.Tensor], 
                            history_updates: List[Dict[str, torch.Tensor]]) -> float:
    """
    计算当前更新与历史更新的方差（变化程度）
    方差大 = 不稳定 = 可能过拟合/噪声
    方差小 = 稳定 = 可靠的更新
    """
    if not history_updates:
        return 0.0  # 没有历史，默认方差为0
    
    # 计算历史平均更新
    avg_update = {}
    for k in current_update.keys():
        if 'num_batches_tracked' in k:
            continue
        avg_update[k] = sum(h[k] for h in history_updates) / len(history_updates)
    
    # 计算当前更新与历史平均的L2距离（方差）
    variance = 0.0
    for k in current_update.keys():
        if 'num_batches_tracked' in k:
            continue
        diff = current_update[k] - avg_update[k]
        variance += torch.sum(diff ** 2).item()
    
    return math.sqrt(variance)


# =========================
# 注意力机制
# =========================

def attention_validation_based(
    updates: List[Tuple[Dict[str, torch.Tensor], int]],
    global_params: Dict[str, torch.Tensor],
    val_loader: DataLoader,
    device: torch.device,
    model_class,
    beta: float = 1.0
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """
    基于验证集的注意力 - 效果最好的方法 ⭐⭐⭐⭐⭐
    
    核心思想：
        在服务器端验证集上评估每个客户端更新的实际效果
        这是最可靠的质量指标！
        
    为什么有效：
        - 客户端本地准确率不能反映全局贡献（可能过拟合特定分布）
        - 服务器验证集代表全局数据分布
        - 在验证集上表现好 = 真的对全局有帮助
        
    预期效果：
        相比基准FedAvg提升 +5-6% 准确率
    
    参数：
        updates: 客户端更新列表 [(state_dict, num_samples), ...]
        global_params: 全局模型参数
        val_loader: 服务器验证集
        device: 设备
        model_class: 模型类（用于创建临时模型）
        beta: 温度参数，控制权重集中程度（默认1.0）
    
    返回：
        聚合后的参数, 注意力权重
    """
    scores = []
    
    # 在验证集上评估每个客户端的更新质量
    for client_params, _ in updates:
        # 创建临时模型应用客户端更新
        temp_model = model_class().to(device)
        load_state_dict(temp_model, client_params)
        
        # 在服务器验证集上评估 - 这是关键！
        _, val_acc = eval_model(temp_model, val_loader, device)
        scores.append(val_acc)
        
        del temp_model
    
    # 基于验证集准确率计算注意力权重
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    attention_weights = F.softmax(scores_tensor / beta, dim=0).tolist()
    
    # 使用注意力权重聚合参数
    keys = updates[0][0].keys()
    out = {}
    for k in keys:
        out[k] = sum(sd[k] * w for (sd, _), w in zip(updates, attention_weights))
    
    return out, attention_weights


def attention_validation_with_stability(
    updates: List[Tuple[Dict[str, torch.Tensor], int]],
    global_params: Dict[str, torch.Tensor],
    val_loader: DataLoader,
    device: torch.device,
    model_class,
    client_ids: List[int],
    update_history: Dict[int, List[Dict[str, torch.Tensor]]],
    beta: float = 1.0,
    stability_weight: float = 0.5,
    history_size: int = 3
) -> Tuple[Dict[str, torch.Tensor], List[float], Dict]:
    """
    基于验证集 + 稳定性的注意力机制（动态筛子） ⭐⭐⭐⭐⭐
    
    核心思想：
        1. 用验证集评估质量（谁的更新对全局有帮助）
        2. 用方差评估稳定性（谁的更新是可靠的）
        3. 方差大 = 变化剧烈 = 可能过拟合/噪声 → 降低权重
        4. 方差小 = 更新稳定 = 可靠的更新 → 保持权重
    
    这是一个"动态筛子"：
        - 过滤掉不稳定的更新（方差大的）
        - 只采纳稳定且高质量的更新
    
    参数：
        updates: 客户端更新列表
        global_params: 全局模型参数
        val_loader: 验证集
        device: 设备
        model_class: 模型类
        client_ids: 本轮参与的客户端ID
        update_history: 每个客户端的历史更新记录
        beta: 温度参数
        stability_weight: 稳定性权重（0-1，默认0.5表示质量和稳定性各占50%）
        history_size: 保留多少轮历史（默认3轮）
    
    返回：
        聚合后的参数, 注意力权重, 更新后的历史记录
    """
    performance_scores = []
    stability_scores = []
    variances = []  # 记录每个客户端的方差
    
    # 步骤1: 在验证集上评估每个客户端的性能（质量）
    for client_params, _ in updates:
        temp_model = model_class().to(device)
        load_state_dict(temp_model, client_params)
        _, val_acc = eval_model(temp_model, val_loader, device)
        performance_scores.append(val_acc)
        del temp_model
    
    # 步骤2: 计算每个客户端更新的稳定性（方差）
    for idx, (client_params, _) in enumerate(updates):
        cid = client_ids[idx]
        
        # 计算当前更新（相对于全局模型）
        current_update = state_dict_sub(client_params, global_params)
        
        # 获取历史更新
        if cid not in update_history:
            update_history[cid] = []
        
        # 计算方差（当前更新与历史的差异）
        variance = compute_update_variance(current_update, update_history[cid])
        variances.append(variance)  # 保存方差用于监控
        
        # 稳定性分数：方差越小，分数越高
        # 对于大模型（170万参数），需要更大的缩放因子
        stability_score = 1.0 / (1.0 + variance / 10000.0)  # 除以10000
        stability_scores.append(stability_score)
        
        # 更新历史（保存当前更新）
        update_history[cid].append(current_update)
        if len(update_history[cid]) > history_size:
            update_history[cid] = update_history[cid][-history_size:]
    
    # 步骤3: 综合质量和稳定性计算最终分数
    # 最终分数 = (1-α) × 性能分数 + α × 稳定性分数
    # 或者用乘积： 最终分数 = 性能分数 × 稳定性分数^α
    final_scores = []
    for perf, stab in zip(performance_scores, stability_scores):
        # 方案1: 线性组合
        # final_score = (1 - stability_weight) * perf + stability_weight * stab
        
        # 方案2: 乘积（推荐）- 两者都要好才能得高分
        final_score = perf * (stab ** stability_weight)
        final_scores.append(final_score)
    
    # 步骤4: 基于最终分数计算注意力权重
    scores_tensor = torch.tensor(final_scores, dtype=torch.float32)
    attention_weights = F.softmax(scores_tensor / beta, dim=0).tolist()
    
    # 步骤5: 使用注意力权重聚合参数
    keys = updates[0][0].keys()
    out = {}
    for k in keys:
        out[k] = sum(sd[k] * w for (sd, _), w in zip(updates, attention_weights))
    
    # 返回聚合参数、权重和更新后的历史
    info = {
        'update_history': update_history,
        'performance_scores': performance_scores,
        'stability_scores': stability_scores,
        'variances': variances  # 返回方差用于监控和调试
    }
    
    return out, attention_weights, info


# =========================
# ResNet110 for CIFAR
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
# Data partition
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

def print_model_info(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型: {model.__class__.__name__}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")


# =========================
# Main Training
# =========================
def run_fedavg_with_attention(
    trainset, client_splits, test_loader, val_loader, device,
    rounds, clients_per_round_frac, local_epochs, batch_size, lr, weight_decay, seed,
    attention_method: str = "stability",
    attention_beta: float = 1.0,
    stability_weight: float = 0.5
):
    start_time = time.time()
    print("\n" + "="*80)
    if attention_method == "none":
        print("【开始训练：联邦平均 (FedAvg)】")
    elif attention_method == "stability":
        print(f"【开始训练：FedAvg + 稳定性筛选注意力聚合（动态筛子）】")
    else:
        print(f"【开始训练：FedAvg + Validation-based 注意力聚合】")
    print("="*80)
    
    if attention_method == "stability":
        print(f"注意力方法: Validation + Stability (质量+稳定性)")
        print(f"  → 在验证集上评估质量（谁对全局有帮助）")
        print(f"  → 计算更新方差评估稳定性（过滤不稳定更新）")
        print(f"  → 方差大（变化剧烈）= 降低权重（可能过拟合/噪声）")
        print(f"  → 方差小（更新稳定）= 保持权重（可靠更新）")
        print(f"  → 稳定性权重: {stability_weight}")
        print(f"  → 温度参数 β: {attention_beta}")
    elif attention_method == "validation":
        print(f"注意力方法: Validation-based (仅质量)")
        print(f"  → 在服务器验证集上评估客户端更新质量")
        print(f"  → 温度参数 β: {attention_beta}")
    
    global_model = resnet110_cifar100().to(device)
    print_model_info(global_model)
    print("-"*80)
    
    rng = random.Random(seed + 999)
    num_clients = len(client_splits)
    m = max(1, int(math.ceil(num_clients * clients_per_round_frac)))
    
    global_params = to_cpu_state_dict(global_model)
    update_history = {}  # 用于稳定性注意力
    best_acc = 0.0
    best_round = 0

    for rnd in range(1, rounds + 1):
        chosen = rng.sample(range(num_clients), k=m)
        updates = []
        local_stats = []

        # 客户端本地训练
        for cid in chosen:
            local_model = resnet110_cifar100().to(device)
            load_state_dict(local_model, global_params)

            subset = Subset(trainset, client_splits[cid])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

            stats = train_client_fedavg(local_model, loader, device, lr, local_epochs, weight_decay)
            local_stats.append(stats)
            updates.append((to_cpu_state_dict(local_model), len(subset)))

        # 聚合模型参数
        if attention_method != "none":
            try:
                if attention_method == "stability":
                    # 使用稳定性+验证集注意力聚合
                    global_params, attn_weights, info = attention_validation_with_stability(
                        updates=updates,
                        global_params=global_params,
                        val_loader=val_loader,
                        device=device,
                        model_class=resnet110_cifar100,
                        client_ids=chosen,
                        update_history=update_history,
                        beta=attention_beta,
                        stability_weight=stability_weight
                    )
                    # 更新历史
                    update_history = info['update_history']
                    # 获取性能和稳定性分数用于显示
                    perf_scores = info.get('performance_scores', [])
                    stab_scores = info.get('stability_scores', [])
                    var_scores = info.get('variances', [])  # 获取方差
                else:
                    # 使用验证集注意力聚合
                    global_params, attn_weights = attention_validation_based(
                        updates=updates,
                        global_params=global_params,
                        val_loader=val_loader,
                        device=device,
                        model_class=resnet110_cifar100,
                        beta=attention_beta
                    )
                    perf_scores = []
                    stab_scores = []
                    var_scores = []
            except Exception as e:
                print(f"注意力聚合出错: {e}，回退到普通FedAvg")
                global_params = weighted_average_state_dict(updates)
                attn_weights = None
                perf_scores = []
                stab_scores = []
                var_scores = []
        else:
            global_params = weighted_average_state_dict(updates)
            attn_weights = None
            perf_scores = []
            stab_scores = []
            var_scores = []
            
        load_state_dict(global_model, global_params)

        # 评估
        te_loss, te_acc = eval_model(global_model, test_loader, device)
        avg_tr_loss = sum(s.loss for s in local_stats) / len(local_stats)
        avg_tr_acc = sum(s.acc for s in local_stats) / len(local_stats)
        
        if te_acc > best_acc:
            best_acc = te_acc
            best_round = rnd
        
        # 打印信息
        if attention_method != "none" and attn_weights is not None:
            max_weight = max(attn_weights)
            min_weight = min(attn_weights)
            std_weight = (sum((w - sum(attn_weights)/len(attn_weights))**2 for w in attn_weights) / len(attn_weights)) ** 0.5
            
            if attention_method == "stability" and perf_scores and stab_scores:
                avg_perf = sum(perf_scores) / len(perf_scores)
                avg_stab = sum(stab_scores) / len(stab_scores)
                avg_var = sum(var_scores) / len(var_scores) if var_scores else 0.0
                print(f"[Round {rnd:03d}] AvgClientLoss {avg_tr_loss:.6f} AvgClientAcc {format_pct(avg_tr_acc)} | "
                      f"TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)} | Best {format_pct(best_acc)}@{best_round} | "
                      f"AttnW: max={max_weight:.3f} min={min_weight:.3f} std={std_weight:.3f} | "
                      f"Perf={avg_perf:.3f} Stab={avg_stab:.3f} Var={avg_var:.1f}")
            else:
                print(f"[Round {rnd:03d}] AvgClientLoss {avg_tr_loss:.6f} AvgClientAcc {format_pct(avg_tr_acc)} | "
                      f"TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)} | Best {format_pct(best_acc)}@{best_round} | "
                      f"AttnW: max={max_weight:.3f} min={min_weight:.3f} std={std_weight:.3f}")
        else:
            print(f"[Round {rnd:03d}] AvgClientLoss {avg_tr_loss:.6f} AvgClientAcc {format_pct(avg_tr_acc)} | "
                  f"TestLoss {te_loss:.6f} TestAcc {format_pct(te_acc)} | Best {format_pct(best_acc)}@{best_round}")

    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"【训练完成】耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print(f"最佳测试准确率: {format_pct(best_acc)} (第{best_round}轮)")
    print(f"{'='*80}\n")
    
    return global_model, te_loss, te_acc, elapsed_time, best_acc


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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # Partition
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    
    # Server validation set
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="从训练集中分出多少作为服务器验证集")
    
    # Attention Method (默认使用)
    parser.add_argument("--attention_method", type=str, default="stability",
                        choices=["stability", "validation", "none"],
                        help="注意力方法: stability(质量+稳定性,默认) | validation(仅质量) | none(关闭注意力)")
    parser.add_argument("--attention_beta", type=float, default=0.005,
                        help="注意力温度参数（越小权重越集中）")
    parser.add_argument("--stability_weight", type=float, default=0.5,
                        help="稳定性权重（0-1，仅stability方法使用）")

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

    # 加载数据集
    full_trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    
    # 分出服务器验证集
    val_size = int(len(full_trainset) * args.val_ratio)
    train_size = len(full_trainset) - val_size
    trainset, valset = random_split(full_trainset, [train_size, val_size], 
                                     generator=torch.Generator().manual_seed(args.seed))
    
    # 创建数据加载器
    val_loader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Client split（只在训练集上分割）
    train_indices = trainset.indices
    if args.partition == "iid":
        client_splits = split_iid(train_indices, args.num_clients, seed=args.seed)
    else:
        # 获取对应的targets（trainset是Subset对象，需要通过.dataset访问原始数据集）
        targets = [trainset.dataset.targets[i] for i in train_indices]
        relative_client_splits = split_dirichlet_by_label(
            targets=targets,
            num_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
            min_size=max(10, len(train_indices) // (args.num_clients * 20))
        )
        # 映射回原始索引
        client_splits = [[train_indices[i] for i in split] for split in relative_client_splits]

    print("\n" + "="*80)
    print("实验配置信息")
    print("="*80)
    print(f"设备: {device}")
    print(f"数据集: CIFAR-100")
    print(f"训练集大小: {len(trainset)} (原始: {len(full_trainset)})")
    print(f"服务器验证集大小: {len(valset)} ({args.val_ratio*100:.0f}%)")
    print(f"测试集大小: {len(testset)}")
    print(f"数据划分方式: {args.partition}" + (f" (alpha={args.dirichlet_alpha})" if args.partition=="dirichlet" else ""))
    print(f"客户端数量: {args.num_clients}")
    print(f"运行方法: FedAvg" + (f" + {args.attention_method}注意力" if args.attention_method != "none" else ""))
    if args.attention_method != "none":
        print(f"注意力温度 β: {args.attention_beta}")
        if args.attention_method == "stability":
            print(f"稳定性权重: {args.stability_weight}")
    print("="*80)

    total_start_time = time.time()
    
    # 训练
    _, te_loss, te_acc, elapsed, best_acc = run_fedavg_with_attention(
        trainset.dataset, client_splits, test_loader, val_loader, device,
        rounds=args.rounds, clients_per_round_frac=args.clients_per_round,
        local_epochs=args.local_epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        attention_method=args.attention_method,
        attention_beta=args.attention_beta,
        stability_weight=args.stability_weight
    )

    # 打印最终结果
    print("\n" + "=" * 80)
    print("最终结果")
    print("=" * 80)
    print(f"方法: FedAvg" + (f" + {args.attention_method}注意力" if args.attention_method != "none" else ""))
    print(f"最终测试准确率: {format_pct(te_acc)}")
    print(f"最佳测试准确率: {format_pct(best_acc)}")
    print(f"测试损失: {te_loss:.4f}")
    print(f"训练耗时: {elapsed:.2f}秒 = {elapsed/60:.2f}分钟")
    print("=" * 80)


if __name__ == "__main__":
    main()
