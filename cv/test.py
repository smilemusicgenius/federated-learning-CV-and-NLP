import argparse
import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------
# ResNet20 for CIFAR (6n+2, n=3 => 20)
# -----------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
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

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Kaiming init (optional but nice)
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


def resnet20_cifar(num_classes=100):
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


# -----------------------
# Federated helpers
# -----------------------
def get_model_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_model_params(model: nn.Module, params: Dict[str, torch.Tensor]):
    state = model.state_dict()
    for k in state.keys():
        state[k] = params[k].to(state[k].device)
    model.load_state_dict(state)

def fedavg(param_list: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """
    param_list: list of (state_dict_cpu, num_samples)
    """
    total = sum(n for _, n in param_list)
    assert total > 0
    keys = param_list[0][0].keys()
    avg = {}
    for k in keys:
        avg[k] = sum(sd[k] * (n / total) for sd, n in param_list)
    return avg


# -----------------------
# Data partitioning
# -----------------------
def split_iid(indices: List[int], num_clients: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    idx = indices[:]
    rng.shuffle(idx)
    shards = [idx[i::num_clients] for i in range(num_clients)]
    return shards

def split_dirichlet_by_label(
    targets: List[int],
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 50
) -> List[List[int]]:
    """
    Non-IID split using Dirichlet distribution per class.
    Ensures each client has at least min_size samples (best-effort).
    """
    rng = random.Random(seed)
    num_classes = len(set(targets))
    class_indices = [[] for _ in range(num_classes)]
    for i, y in enumerate(targets):
        class_indices[y].append(i)

    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    # Repeat until satisfied (best-effort, avoid infinite loops)
    for _ in range(10):
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = class_indices[c]
            if len(idx_c) == 0:
                continue

            # sample proportions
            proportions = torch.distributions.Dirichlet(
                torch.full((num_clients,), alpha)
            ).sample().tolist()

            # split according to proportions
            counts = [int(p * len(idx_c)) for p in proportions]
            # fix rounding so sum == len(idx_c)
            diff = len(idx_c) - sum(counts)
            for i in range(abs(diff)):
                counts[i % num_clients] += 1 if diff > 0 else -1

            start = 0
            for client_id, cnt in enumerate(counts):
                if cnt <= 0:
                    continue
                client_indices[client_id].extend(idx_c[start:start + cnt])
                start += cnt

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_size:
            return client_indices

    return client_indices


# -----------------------
# Train / Eval
# -----------------------
@dataclass
class TrainStats:
    loss: float
    acc: float

def train_one_client(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_epochs: int,
    weight_decay: float,
    max_grad_norm: float = 0.0
) -> TrainStats:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # (可选) 简单的 cosine 学习率（每个客户端本地跑 local_epochs）
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, local_epochs))

    total_loss, correct, total = 0.0, 0, 0
    for _ in range(local_epochs):
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
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        sched.step()

    return TrainStats(loss=total_loss / max(1, total), acc=correct / max(1, total))

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> TrainStats:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return TrainStats(loss=total_loss / max(1, total), acc=correct / max(1, total))


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients_per_round", type=float, default=0.5, help="fraction in (0,1]")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)

    # partition options
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="smaller => more non-iid")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # CIFAR-100 normalization
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

    # Prepare client splits
    all_train_indices = list(range(len(trainset)))

    if args.partition == "iid":
        client_splits = split_iid(all_train_indices, args.num_clients, seed=args.seed)
    else:
        # Use original targets (no augmentation transform here, so ok)
        targets = trainset.targets  # list[int]
        client_splits = split_dirichlet_by_label(
            targets=targets,
            num_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
            min_size=max(10, len(trainset) // (args.num_clients * 20))
        )

    # Build test loader
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Global model
    global_model = resnet20_cifar(num_classes=100).to(device)
    global_params = get_model_params(global_model)

    # For logging
    print(f"Device: {device}")
    print(f"Clients: {args.num_clients}, Rounds: {args.rounds}, "
          f"ClientFrac: {args.clients_per_round}, LocalEpochs: {args.local_epochs}")
    print(f"Partition: {args.partition}" + (f" (alpha={args.dirichlet_alpha})" if args.partition == "dirichlet" else ""))

    # Federated training
    clients_each_round = max(1, int(math.ceil(args.num_clients * args.clients_per_round)))
    rng = random.Random(args.seed + 999)

    for rnd in range(1, args.rounds + 1):
        chosen = rng.sample(range(args.num_clients), k=clients_each_round)

        updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
        local_losses, local_accs = [], []

        pbar = tqdm(chosen, desc=f"Round {rnd}/{args.rounds}", leave=False)
        for cid in pbar:
            # Create local model copy
            local_model = resnet20_cifar(num_classes=100).to(device)
            set_model_params(local_model, global_params)

            subset = Subset(trainset, client_splits[cid])
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True,
                                num_workers=2, pin_memory=True, drop_last=False)

            stats = train_one_client(
                local_model, loader, device=device,
                lr=args.lr, local_epochs=args.local_epochs,
                weight_decay=args.weight_decay,
                max_grad_norm=0.0
            )
            local_losses.append(stats.loss)
            local_accs.append(stats.acc)

            local_params = get_model_params(local_model)
            updates.append((local_params, len(subset)))

            pbar.set_postfix(loss=f"{stats.loss:.4f}", acc=f"{stats.acc*100:.2f}%")

        # Aggregate
        global_params = fedavg(updates)
        set_model_params(global_model, global_params)

        # Evaluate
        test_stats = evaluate(global_model, test_loader, device=device)

        print(
            f"[Round {rnd:03d}] "
            f"ClientTrain Loss: {sum(local_losses)/len(local_losses):.6f}, "
            f"ClientTrain Acc: {sum(local_accs)/len(local_accs)*100:.2f}% | "
            f"Test Loss: {test_stats.loss:.6f}, Test Acc: {test_stats.acc*100:.2f}%"
        )

    # Save final model
    save_path = "fed_resnet20_cifar100.pth"
    torch.save(global_model.state_dict(), save_path)
    print(f"Saved global model to: {save_path}")


if __name__ == "__main__":
    main()
