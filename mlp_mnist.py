#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A reproducible two-layer MLP baseline for MNIST classification.

与实验四参考代码相比，本实现保留了相同的网络结构（784-128-10）与优化设置
（SGD, lr=0.01, batch=2048），但补齐了可复现性与评估协议：
固定随机种子、训练集内部划分验证集、每轮验证、测试集仅用于最终一次评估、
按样本数加权的损失统计、最佳模型与完整实验记录的持久化。

Usage:
    python mlp_mnist.py
    python mlp_mnist.py --epochs 30 --batch-size 2048 --lr 0.01 --seed 42
    python mlp_mnist.py --device cpu --patience 5 --output-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


PALETTE: dict[str, str] = {
    "train": "#1F4E79",
    "val": "#3E86B0",
    "grid": "#DCE6EE",
    "text": "#1B2F42",
    "spine": "#9FB3C4",
}

# MNIST 训练集全局均值与标准差，用于标准化（常用经验值）
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
MNIST_TRAIN_SIZE = 60_000


@dataclass
class Config:
    """实验超参数集合；全部字段都会被写入 results/config.json。"""

    seed: int = 42
    epochs: int = 30
    batch_size: int = 2048
    lr: float = 0.01
    momentum: float = 0.0
    hidden: int = 128
    val_size: int = 5_000
    patience: int = 0
    device: str = "auto"
    data_dir: str = "data"
    output_dir: str = "results"
    num_workers: int = 0
    download: bool = True
    save: bool = True
    plot: bool = True
    deterministic: bool = True


# 可复现性控制
def set_seed(seed: int, deterministic: bool = True) -> None:
    """固定 Python / NumPy / CPU / CUDA 的随机源，并尽可能启用确定性算子。"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # 兼容旧版 PyTorch
            torch.use_deterministic_algorithms(True)


def resolve_device(preference: str) -> torch.device:
    """按优先级选择计算设备；显式指定不可用时直接报错，避免静默回退。"""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(preference)


def seeded_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# 数据与模型
def build_transform() -> transforms.Compose:
    """ToTensor 将像素缩放到 [0,1]，Normalize 再做通道标准化。"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def build_datasets(cfg: Config) -> tuple[Any, Any, Any]:
    """构建训练集 / 验证集 / 测试集。

    验证集从原始训练集中按固定 seed 划分，测试集全程独立，
    仅在训练结束后用于一次最终评估，以避免模型选择阶段的信息泄漏。
    """
    transform = build_transform()
    full_train = datasets.MNIST(
        cfg.data_dir, train=True, download=cfg.download, transform=transform
    )
    test_set = datasets.MNIST(
        cfg.data_dir, train=False, download=cfg.download, transform=transform
    )
    n_total = len(full_train)
    if not 0 < cfg.val_size < n_total:
        raise ValueError(f"val_size must be in (0, {n_total})")
    generator = seeded_generator(cfg.seed)
    train_set, val_set = random_split(
        full_train,
        [n_total - cfg.val_size, cfg.val_size],
        generator=generator,
    )
    return train_set, val_set, test_set


def build_loader(
    dataset: Any,
    cfg: Config,
    shuffle: bool,
    seed_offset: int,
    pin_memory: bool,
) -> DataLoader:
    """训练集打乱顺序由独立 seed 的 generator 控制，保证跨次运行一致。"""
    generator = seeded_generator(cfg.seed + seed_offset) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        generator=generator,
        pin_memory=pin_memory,
        drop_last=False,
    )


class MLP(nn.Module):
    """两层全连接网络：784 -> hidden -> 10，输出未归一化的 logits。"""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # [N,1,28,28] -> [N,784]
        x = F.relu(self.fc1(x))  # [N,784] -> [N,hidden]
        return self.fc2(x)  # [N,hidden] -> [N,10]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 训练与评估
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """执行一个 epoch：zero_grad -> forward/loss -> backward -> step。

    统计口径按样本数加权：每个 batch 的均值损失乘以该 batch 的样本数，
    再除以总样本数，避免最后一个不足 batch_size 的批次被等权计入。
    """
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += batch_size
    return {"loss": loss_sum / total, "acc": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """在给定数据集上计算平均交叉熵与 top-1 准确率，不构建计算图。"""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        batch_size = y.size(0)
        loss_sum += F.cross_entropy(logits, y).item() * batch_size
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += batch_size
    return {"loss": loss_sum / total, "acc": correct / total}


# 结果持久化
def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_history_csv(path: Path, history: Sequence[dict[str, Any]]) -> None:
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row[key] for key in fields})


def plot_curves(history: Sequence[dict[str, Any]], path: Path) -> None:
    """绘制损失与准确率曲线；冷色调、细线、无装饰，风格与 assets/ 一致。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    series = [
        ("train_loss", "val_loss", "Cross-entropy loss"),
        ("train_acc", "val_acc", "Accuracy"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=160)
    for ax, (train_key, val_key, title) in zip(axes, series):
        ax.plot(
            epochs,
            [row[train_key] for row in history],
            color=PALETTE["train"],
            linewidth=1.8,
            label="Training set",
        )
        ax.plot(
            epochs,
            [row[val_key] for row in history],
            color=PALETTE["val"],
            linewidth=1.8,
            label="Validation set",
        )
        ax.set_title(title, fontsize=10, color=PALETTE["text"], pad=10)
        ax.set_xlabel("Epoch", fontsize=9, color=PALETTE["text"])
        ax.grid(True, color=PALETTE["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(PALETTE["spine"])
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(colors=PALETTE["text"], labelsize=8, length=0)
    axes[1].legend(
        frameon=False, fontsize=8, labelcolor=PALETTE["text"], loc="lower right"
    )
    fig.tight_layout(w_pad=2.4)
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# 实验主流程
def run(cfg: Config) -> dict[str, Any]:
    set_seed(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    train_set, val_set, test_set = build_datasets(cfg)
    train_loader = build_loader(train_set, cfg, True, 1, device.type == "cuda")
    val_loader = build_loader(val_set, cfg, False, 0, device.type == "cuda")
    test_loader = build_loader(test_set, cfg, False, 0, device.type == "cuda")

    model = MLP(cfg.hidden).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum
    )

    out_dir = Path(cfg.output_dir)
    if cfg.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_acc": round(train_metrics["acc"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_acc": round(val_metrics["acc"], 6),
            }
        )
        print(
            f"epoch {epoch:02d}/{cfg.epochs:02d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if cfg.save:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "config": asdict(cfg),
                        "best_val_acc": best_val_acc,
                    },
                    out_dir / "best_model.pt",
                )
        else:
            stale_epochs += 1

        if cfg.patience > 0 and stale_epochs >= cfg.patience:
            print(f"early stopping at epoch {epoch} (patience={cfg.patience})")
            break

    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)  # 以验证集最优权重做最终测试
    test_metrics = evaluate(model, test_loader, device)

    metrics: dict[str, Any] = {
        "seed": cfg.seed,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "trainable_parameters": count_parameters(model),
        "dataset_sizes": {
            "train": len(train_set),
            "validation": len(val_set),
            "test": len(test_set),
        },
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 6),
        "test_loss": round(test_metrics["loss"], 6),
        "test_acc": round(test_metrics["acc"], 6),
        "elapsed_seconds": round(elapsed, 2),
    }

    if cfg.save:
        save_json(
            out_dir / "config.json",
            {
                **asdict(cfg),
                "resolved_device": str(device),
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
        )
        save_json(out_dir / "metrics.json", metrics)
        save_history_csv(out_dir / "history.csv", history)
        if cfg.plot:
            try:
                plot_curves(history, out_dir / "training_curves.png")
            except ImportError:
                print("matplotlib unavailable; skipped curve rendering")

    return metrics


def validate(cfg: Config) -> None:
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.hidden < 1 or cfg.lr <= 0:
        raise ValueError("epochs/batch_size/hidden must be positive and lr > 0")
    if not 0 < cfg.val_size < MNIST_TRAIN_SIZE:
        raise ValueError("val_size must be in (0, 60000)")
    if cfg.patience < 0 or cfg.num_workers < 0:
        raise ValueError("patience and num_workers must be non-negative")
    if not 0 <= cfg.momentum < 1:
        raise ValueError("momentum must be in [0, 1)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible MLP baseline for MNIST classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    parser.add_argument("--epochs", type=int, default=30, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, dest="batch_size")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=128, help="hidden units")
    parser.add_argument("--val-size", type=int, default=5000, dest="val_size")
    parser.add_argument(
        "--patience", type=int, default=0, help="early stopping patience; 0 disables"
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--data-dir", default="data", dest="data_dir")
    parser.add_argument("--output-dir", default="results", dest="output_dir")
    parser.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.add_argument("--no-save", action="store_false", dest="save")
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    parser.add_argument(
        "--no-deterministic", action="store_false", dest="deterministic"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    cfg = Config(**vars(build_parser().parse_args(argv)))
    validate(cfg)
    metrics = run(cfg)
    print(
        "\n[summary] "
        f"best epoch {metrics['best_epoch']} | "
        f"best val acc {metrics['best_val_acc']:.4f} | "
        f"test loss {metrics['test_loss']:.4f} | "
        f"test acc {metrics['test_acc']:.4f} | "
        f"params {metrics['trainable_parameters']} | "
        f"device {metrics['device']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
