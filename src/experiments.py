"""
experiments.py — Hyperparameter ablation study for Part 3.

This module trains the ColorPredictor under different regularization configurations
to systematically analyze the effects of BatchNorm and Dropout.

Ablation Study Design (2×2 Factorial)
--------------------------------------
Fixed: Learning Rate = 1e-2
Variables: BatchNorm (Yes/No) × Dropout (Yes/No)

Configurations tested:
  1. No BN, No Dropout : LR=1e-2, BN=False, Drop=False, p=0.0  (baseline, no regularization)
  2. BN only           : LR=1e-2, BN=True,  Drop=False, p=0.0  (normalization only)
  3. Dropout only      : LR=1e-2, BN=False, Drop=True,  p=0.2  (stochastic regularization only)
  4. BN + Dropout      : LR=1e-2, BN=True,  Drop=True,  p=0.2  (both regularization techniques)

This design allows us to observe:
- The individual effect of BatchNorm
- The individual effect of Dropout
- Whether they interact synergistically or antagonistically

Each configuration is trained for a fixed number of epochs (defaults to
cfg.NUM_EPOCHS so the demo stays fast — increase for a thorough study).
Results are saved as CSV and comparison plots in the outputs/ directory.

Usage (CLI):
    python -m src.experiments [--target B] [--epochs 10]
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import src.config as cfg
from src.data_loader import build_dataloaders
from src.model import ColorPredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Experiment configuration dataclass ──────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str
    lr: float = 1e-3
    use_batchnorm: bool = True
    use_dropout: bool = True
    dropout_rate: float = 0.2


# ─── Pre-defined sweep ────────────────────────────────────────────────────────

SWEEP: List[ExperimentConfig] = [
    ExperimentConfig("No BN, No Dropout", lr=1e-2, use_batchnorm=False, use_dropout=False, dropout_rate=0.0),
    ExperimentConfig("BN only",           lr=1e-2, use_batchnorm=True,  use_dropout=False, dropout_rate=0.0),
    ExperimentConfig("Dropout only",      lr=1e-2, use_batchnorm=False, use_dropout=True,  dropout_rate=0.2),
    ExperimentConfig("BN + Dropout",      lr=1e-2, use_batchnorm=True,  use_dropout=True,  dropout_rate=0.2),
]


# ─── Single-run training loop ─────────────────────────────────────────────────

def _train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    desc: str = "",
) -> float:
    """Run one epoch; return mean loss."""
    model.train(train)
    total_loss = 0.0
    num_batches = len(loader)
    
    # Reduce progress bar update frequency for MPS stability
    is_mps = str(device) == "mps"
    bar = tqdm(loader, desc=desc, leave=False, unit="batch", 
               dynamic_ncols=True, mininterval=1.0 if is_mps else 0.1)
    
    with torch.set_grad_enabled(train):
        for batch_idx, (x_batch, y_batch, *_) in enumerate(bar):
            # Use blocking transfers for MPS to avoid hangs
            x_batch = x_batch.to(device, non_blocking=False)
            y_batch = y_batch.to(device, non_blocking=False)
            
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Extract loss value before accumulating
            loss_val = loss.item()
            total_loss += loss_val * x_batch.size(0)
            
            # Update progress bar less frequently
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                bar.set_postfix(loss=f"{loss_val:.5f}")
            
            # Clear MPS cache every 20 batches to prevent memory buildup
            if is_mps and batch_idx % 20 == 0:
                torch.mps.empty_cache()
    
    # Final synchronization for MPS
    if is_mps:
        torch.mps.synchronize()
        torch.mps.empty_cache()
    
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


def run_experiment(
    exp: ExperimentConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int,
) -> Dict[str, object]:
    """
    Train a single experiment configuration and return performance metrics.
    
    Parameters
    ----------
    exp : ExperimentConfig
        Configuration specifying LR, BatchNorm, Dropout settings.
    train_loader : DataLoader
        Training data loader (shared across all experiments for fair comparison).
    val_loader : DataLoader
        Validation data loader (shared across all experiments).
    num_epochs : int
        Number of training epochs.
        
    Returns
    -------
    Dict with keys: name, lr, use_batchnorm, use_dropout, dropout_rate,
                    best_val_loss, final_train_loss, train_losses, 
                    val_losses, elapsed_s
    """
    logger.info(f"  ▶ {exp.name!r:30s}  lr={exp.lr}  BN={exp.use_batchnorm}  "
                f"Drop={exp.use_dropout}  p={exp.dropout_rate}")

    # Initialize model with specified regularization configuration
    model = ColorPredictor(
        use_batchnorm=exp.use_batchnorm,
        use_dropout=exp.use_dropout,
        dropout_rate=exp.dropout_rate,
    ).to(cfg.DEVICE)

    # Loss function and optimizer (same for all experiments)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=exp.lr)
    
    # LR scheduler with patience (provides early stopping behavior)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg.LR_SCHEDULER_FACTOR,     # Multiply LR by 0.5 on plateau
        patience=cfg.LR_SCHEDULER_PATIENCE,  # Wait 3 epochs before reducing
    )

    # Track best validation loss and loss curves
    best_val_loss = float("inf")
    train_losses: List[float] = []
    val_losses:   List[float] = []

    # Training loop with progress bar
    t_start = time.time()
    epoch_bar = tqdm(range(1, num_epochs + 1), desc=f"  {exp.name}", unit="epoch",
                     dynamic_ncols=True, leave=False)
    for epoch in epoch_bar:
        # Run training epoch (with gradient updates)
        tl = _train_one_epoch(
            model, train_loader, criterion, optimizer,
            cfg.DEVICE, train=True,
            desc=f"    Train {epoch:02d}/{num_epochs}",
        )
        # Run validation epoch (no gradient updates)
        vl = _train_one_epoch(
            model, val_loader, criterion, None,
            cfg.DEVICE, train=False,
            desc=f"    Val   {epoch:02d}/{num_epochs}",
        )
        
        # Record losses for analysis
        train_losses.append(tl)
        val_losses.append(vl)
        
        # Update learning rate based on validation loss (patience-based)
        scheduler.step(vl)
        
        # Track best validation performance
        best_val_loss = min(best_val_loss, vl)
        epoch_bar.set_postfix(train=f"{tl:.5f}", val=f"{vl:.5f}")

    elapsed = time.time() - t_start
    logger.info(f"    ↳ best val_loss={best_val_loss:.6f}  time={elapsed:.1f}s")

    return {
        "name":             exp.name,
        "lr":               exp.lr,
        "use_batchnorm":    exp.use_batchnorm,
        "use_dropout":      exp.use_dropout,
        "dropout_rate":     exp.dropout_rate,
        "best_val_loss":    best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else float("nan"),
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "elapsed_s":        elapsed,
    }


# ─── Save results ─────────────────────────────────────────────────────────────

def save_results_csv(results: List[Dict], target_channel: str) -> Path:
    """Save a CSV summary of all experiment results."""
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = cfg.OUTPUTS_DIR / f"hyperparameter_results_{target_channel}.csv"
    fieldnames = ["name", "lr", "use_batchnorm", "use_dropout", "dropout_rate",
                  "best_val_loss", "final_train_loss", "elapsed_s"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Experiment CSV saved → {save_path}")
    return save_path


def plot_hyperparameter_comparison(results: List[Dict], target_channel: str) -> Path:
    """
    Two-panel figure:
      Left  — bar chart of best validation loss per configuration
      Right — learning curves for all configurations on the same axes
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = cfg.OUTPUTS_DIR / f"hyperparameter_comparison_{target_channel}.png"

    names      = [r["name"]          for r in results]
    best_vals  = [r["best_val_loss"] for r in results]

    # Sort bar chart by val loss (lowest = best, left)
    sort_idx   = np.argsort(best_vals)
    names_s    = [names[i]     for i in sort_idx]
    vals_s     = [best_vals[i] for i in sort_idx]

    cmap   = plt.cm.tab10  # type: ignore[attr-defined]
    colors = [cmap(i / len(results)) for i in range(len(results))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: bar chart ────────────────────────────────────────────────────────
    bars = ax1.barh(names_s, vals_s,
                    color=[colors[list(names).index(n)] for n in names_s],
                    edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Best Validation MSE (lower is better)", fontsize=10)
    ax1.set_title("Hyperparameter Comparison\n(best val MSE per config)", fontsize=11, fontweight="bold")
    ax1.invert_yaxis()   # best at top
    val_range = max(vals_s) - min(vals_s) if len(vals_s) > 1 else vals_s[0]
    for bar, v in zip(bars, vals_s):
        ax1.text(bar.get_width() + val_range * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f"{v:.6f}", va="center", ha="left", fontsize=8.5)
    ax1.set_xlim(0, max(vals_s) * 1.22)
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Right: learning curves ────────────────────────────────────────────────
    for i, r in enumerate(results):
        epochs = range(1, len(r["val_losses"]) + 1)
        ax2.plot(epochs, r["val_losses"], label=r["name"],
                 color=colors[i], linewidth=1.8)

    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation MSE", fontsize=10)
    ax2.set_title("Validation Loss Curves\n(all configurations)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Hyperparameter Ablation Study  —  target channel: {target_channel}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Hyperparameter comparison plot saved → {save_path}")
    return save_path


def print_results_table(results: List[Dict]) -> None:
    """Print a ranked summary table to the console."""
    sorted_r = sorted(results, key=lambda r: r["best_val_loss"])
    hdr = (f"{'Rank':>4}  {'Configuration':<25} {'LR':>8}  "
           f"{'BN':>5}  {'Drop':>5}  {'p':>5}  {'Best Val MSE':>14}  {'Time(s)':>8}")
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for rank, r in enumerate(sorted_r, 1):
        print(
            f"{rank:>4}  {r['name']:<25} {r['lr']:>8.0e}  "
            f"{'Y' if r['use_batchnorm'] else 'N':>5}  "
            f"{'Y' if r['use_dropout'] else 'N':>5}  "
            f"{r['dropout_rate']:>5.1f}  "
            f"{r['best_val_loss']:>14.6f}  "
            f"{r['elapsed_s']:>8.1f}"
        )
    print(sep)
    best = sorted_r[0]
    logger.info(f"Best config: {best['name']!r}  val_loss={best['best_val_loss']:.6f}")


# ─── Full sweep entry-point ───────────────────────────────────────────────────

def run_sweep(
    target_channel: str = cfg.TARGET_CHANNEL,
    num_epochs: int = cfg.NUM_EPOCHS,
    configs: List[ExperimentConfig] | None = None,
) -> List[Dict]:
    """
    Run the full hyperparameter sweep and save outputs.

    Parameters
    ----------
    target_channel : str  —  'R', 'G', or 'B'
    num_epochs     : int  —  epochs per configuration
    configs        : list of ExperimentConfig  (defaults to SWEEP)

    Returns
    -------
    List of result dicts (one per configuration), sorted by best_val_loss.
    """
    if configs is None:
        configs = SWEEP

    logger.info(
        f"=== Hyperparameter Sweep  target='{target_channel}'  "
        f"epochs={num_epochs}  configs={len(configs)} ==="
    )

    # Build data loaders once (shared across all configs for fair comparison)
    logger.info("Building data loaders …")
    train_loader, val_loader, _ = build_dataloaders(target_channel)

    results: List[Dict] = []
    for exp in configs:
        result = run_experiment(exp, train_loader, val_loader, num_epochs)
        results.append(result)

    # Save outputs
    save_results_csv(results, target_channel)
    plot_hyperparameter_comparison(results, target_channel)
    print_results_table(results)

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for ColorPredictor")
    parser.add_argument(
        "--target", type=str, default=cfg.TARGET_CHANNEL,
        choices=["R", "G", "B"],
        help="Target channel to predict (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs", type=int, default=cfg.NUM_EPOCHS,
        help="Training epochs per configuration (default: %(default)s).",
    )
    args = parser.parse_args()
    run_sweep(target_channel=args.target, num_epochs=args.epochs)
