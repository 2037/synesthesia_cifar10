"""
evaluator.py — Metrics, visualisation, and per-channel comparison.

Metrics computed:
  - MSE  : Mean Squared Error
  - MAE  : Mean Absolute Error
  - PSNR : Peak Signal-to-Noise Ratio  (dB)
  - SSIM : Structural Similarity Index

Usage (CLI):
    python -m src.evaluator --target B
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # headless backend — safe on server / Colab
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

import src.config as cfg
from src.model import ColorPredictor
from src.data_loader import build_dataloaders, CHANNEL_IDX, load_class_names

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Helper: load a trained model ─────────────────────────────────────────────

def load_model(target_channel: str) -> ColorPredictor:
    """Load best checkpoint for *target_channel* onto DEVICE."""
    ckpt = cfg.MODELS_DIR / f"best_model_{target_channel}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt}. "
            "Run trainer.py first."
        )
    model = ColorPredictor().to(cfg.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))
    model.eval()
    logger.info(f"Loaded model from {ckpt}")
    return model


# ─── Core evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    target_channel: str,
) -> Dict[str, float]:
    """
    Compute MSE, MAE, PSNR, and SSIM on *loader*.

    Returns
    -------
    dict with keys 'MSE', 'MAE', 'PSNR', 'SSIM'.
    """
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for x_batch, y_batch, *_ in loader:   # *_ discards labels & full-rgb tensors
        x_batch = x_batch.to(cfg.DEVICE)
        preds = model(x_batch).cpu().numpy()          # (B, 1, 32, 32)
        targets = y_batch.numpy()                     # (B, 1, 32, 32)
        all_preds.append(preds)
        all_targets.append(targets)

    preds_arr = np.concatenate(all_preds, axis=0)     # (N, 1, 32, 32)
    tgts_arr  = np.concatenate(all_targets, axis=0)   # (N, 1, 32, 32)

    mse = float(np.mean((preds_arr - tgts_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - tgts_arr)))

    # PSNR and SSIM operate on 2-D images
    psnr_values, ssim_values = [], []
    for p, t in zip(preds_arr, tgts_arr):
        p2d = p[0]                    # (32, 32)
        t2d = t[0]
        psnr_values.append(psnr_fn(t2d, p2d, data_range=1.0))
        ssim_values.append(ssim_fn(t2d, p2d, data_range=1.0))

    metrics = {
        "MSE":  mse,
        "MAE":  mae,
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
    }
    for k, v in metrics.items():
        logger.info(f"  {k:4s} = {v:.4f}")
    return metrics


# ─── Reconstruction ───────────────────────────────────────────────────────────

def _reconstruct_rgb(
    input_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    target_channel: str,
) -> np.ndarray:
    """
    Reconstruct a full RGB image from the two input channels and the prediction.

    Parameters
    ----------
    input_tensor : torch.Tensor of shape (2, 32, 32) — the two input channels
    pred_tensor  : torch.Tensor of shape (1, 32, 32) — predicted channel
    target_channel : str — 'R', 'G', or 'B'

    Returns
    -------
    np.ndarray of shape (32, 32, 3), values in [0, 1].
    """
    target_idx = CHANNEL_IDX[target_channel]
    input_indices = [i for i in range(3) if i != target_idx]

    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    inp_np = input_tensor.cpu().numpy()
    pred_np = pred_tensor.cpu().numpy()[0]   # (32, 32)

    rgb[:, :, input_indices[0]] = inp_np[0]
    rgb[:, :, input_indices[1]] = inp_np[1]
    rgb[:, :, target_idx]       = pred_np
    return rgb


def _original_rgb(input_tensor: torch.Tensor, target_tensor: torch.Tensor, target_channel: str) -> np.ndarray:
    """Reconstruct original RGB from the two input channels + ground-truth channel."""
    return _reconstruct_rgb(input_tensor, target_tensor, target_channel)


# ─── Save sample comparisons ──────────────────────────────────────────────────

@torch.no_grad()
def save_sample_comparisons(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    target_channel: str,
    n_samples: int = cfg.NUM_SAMPLE_IMAGES,
) -> Path:
    """
    Generate and save a grid of *n_samples* comparisons:
    Original | Reconstructed | Difference map.

    Returns the path to the saved figure.
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = cfg.OUTPUTS_DIR / f"samples_{target_channel}.png"

    # Grab one batch (4-tuple: input, target, label, full_rgb)
    x_batch, y_batch, *_ = next(iter(loader))
    x_sample = x_batch[:n_samples]
    y_sample  = y_batch[:n_samples]
    preds = model(x_sample.to(cfg.DEVICE)).cpu()

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, n_samples * 2.5))
    fig.suptitle(
        f"Synesthesia CIFAR-10  |  Target channel: {target_channel}\n"
        "Left: Original  |  Centre: Reconstructed  |  Right: |Difference|",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for i in range(n_samples):
        orig = _original_rgb(x_sample[i], y_sample[i],  target_channel)
        recon= _reconstruct_rgb(x_sample[i], preds[i],  target_channel)
        diff = np.abs(orig - recon)

        axes[i, 0].imshow(np.clip(orig,  0, 1))
        axes[i, 1].imshow(np.clip(recon, 0, 1))
        im = axes[i, 2].imshow(diff, vmin=0, vmax=0.5, cmap="hot")

        for j, title in enumerate(["Original", "Reconstructed", "|Diff|"]):
            axes[i, j].set_title(title if i == 0 else "", fontsize=9)
            axes[i, j].axis("off")

    plt.colorbar(im, ax=axes[:, 2], shrink=0.6, label="Absolute error")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Sample comparisons saved to {save_path}")
    return save_path


# ─── Loss-curve plot ──────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    target_channel: str,
) -> Path:
    """Plot and save training / validation loss curves."""
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = cfg.OUTPUTS_DIR / f"loss_curve_{target_channel}.png"

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train loss", linewidth=2)
    plt.plot(epochs, val_losses,   label="Val loss",   linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training curves — target='{target_channel}'")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info(f"Loss curves saved to {save_path}")
    return save_path


# ─── Channel distribution visualisation ──────────────────────────────────────

@torch.no_grad()
def plot_channel_distributions(
    loader: torch.utils.data.DataLoader,
    target_channel: str,
    n_batches: int = 5,
) -> Path:
    """
    Plot the pixel-value distributions of each channel in the first *n_batches*
    of data.
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = cfg.OUTPUTS_DIR / f"channel_dist_{target_channel}.png"

    channel_names = ["R", "G", "B"]
    target_idx = CHANNEL_IDX[target_channel]
    input_indices = [i for i in range(3) if i != target_idx]
    all_data: Dict[str, list[np.ndarray]] = {c: [] for c in channel_names}

    for batch_idx, (x_batch, y_batch, *_) in enumerate(loader):
        if batch_idx >= n_batches:
            break
        inp_np = x_batch.numpy()   # (B, 2, 32, 32)
        tgt_np = y_batch.numpy()   # (B, 1, 32, 32)

        all_data[channel_names[input_indices[0]]].append(inp_np[:, 0].ravel())
        all_data[channel_names[input_indices[1]]].append(inp_np[:, 1].ravel())
        all_data[channel_names[target_idx]].append(tgt_np[:, 0].ravel())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=False)
    colours = {"R": "red", "G": "green", "B": "blue"}
    for ax, ch in zip(axes, channel_names):
        vals = np.concatenate(all_data[ch])
        ax.hist(vals, bins=64, color=colours[ch], alpha=0.7, edgecolor="none")
        ax.set_title(f"Channel {ch}" + (" ← target" if ch == target_channel else ""))
        ax.set_xlabel("Pixel value (normalised)")
        ax.set_ylabel("Count")

    plt.suptitle("Pixel-value distributions (first 5 batches)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info(f"Channel distributions saved to {save_path}")
    return save_path


# ─── Performance matrix ───────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def plot_performance_matrix(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    target_channel: str,
    n_per_tier: int = 4,
) -> Path:
    """
    Collect per-image MSE across the full *loader*, then plot a labelled matrix:

    - **Columns**: n_per_tier examples per tier  →  3 × n_per_tier columns total
    - **Rows**   : Original | Reconstructed | |Difference|  (3 rows)
    - Column groups labelled "Best", "Mean", "Worst"
    - Each column header shows the CIFAR-10 class name and per-image MSE

    Saves the figure to outputs/performance_matrix_{target_channel}.png.
    """
    # ── 1. Run the full loader, collect per-image tensors & MSE ───────────────
    all_inputs:  list[torch.Tensor] = []   # each (2, 32, 32)
    all_targets: list[torch.Tensor] = []   # each (1, 32, 32)
    all_preds:   list[torch.Tensor] = []   # each (1, 32, 32)
    all_fulls:   list[torch.Tensor] = []   # each (3, 32, 32)
    all_labels:  list[int]          = []

    for x_batch, y_batch, lbl_batch, img_batch in loader:
        preds = model(x_batch.to(cfg.DEVICE)).cpu()   # (B, 1, 32, 32)
        for i in range(len(x_batch)):
            all_inputs.append(x_batch[i])
            all_targets.append(y_batch[i])
            all_preds.append(preds[i])
            all_fulls.append(img_batch[i])
            all_labels.append(int(lbl_batch[i]))

    # Per-image MSE: mean over (1, 32, 32)
    mse_per_img = np.array([
        float(((p - t) ** 2).mean())
        for p, t in zip(all_preds, all_targets)
    ])

    # ── 2. Select indices for each tier ───────────────────────────────────────
    order = np.argsort(mse_per_img)          # low → high
    n_total = len(order)

    best_idx  = order[:n_per_tier]
    mid_start = (n_total - n_per_tier) // 2
    mean_idx  = order[mid_start : mid_start + n_per_tier]
    worst_idx = order[-n_per_tier:]

    tier_indices = {
        "Best\n(lowest MSE)":   best_idx,
        "Mean\n(median MSE)":   mean_idx,
        "Worst\n(highest MSE)": worst_idx,
    }

    # ── 3. Build figure: 3 rows × (n_per_tier * 3) columns ───────────────────
    n_cols = n_per_tier * len(tier_indices)   # e.g. 4*3 = 12
    n_rows = 3                                # Original | Reconstructed | |Diff|

    fig_w = n_cols * 1.6 + 0.5
    fig_h = n_rows * 1.8 + 2.0    # Extra space at bottom for colorbar
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    row_labels = ["Original", "Reconstructed", "|Difference|"]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=9, fontweight="bold",
                               rotation=90, labelpad=6)

    col = 0
    for tier_name, idx_arr in tier_indices.items():
        # Tier header spanning n_per_tier columns
        mid_col = col + n_per_tier // 2 - (0 if n_per_tier % 2 else 0)
        axes[0, col + n_per_tier // 2].set_title(
            tier_name, fontsize=10, fontweight="bold",
            color="white", pad=20,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc={"Best\n(lowest MSE)": "#2196F3",
                    "Mean\n(median MSE)": "#FF9800",
                    "Worst\n(highest MSE)": "#F44336"}[tier_name],
                ec="none",
            ),
        )

        for k, img_idx in enumerate(idx_arr):
            c = col + k
            inp   = all_inputs[img_idx]   # (2, 32, 32)
            tgt   = all_targets[img_idx]  # (1, 32, 32)
            pred  = all_preds[img_idx]    # (1, 32, 32)
            full  = all_fulls[img_idx]    # (3, 32, 32)
            label = all_labels[img_idx]
            mse_v = mse_per_img[img_idx]

            # Reconstruct images
            orig  = full.permute(1, 2, 0).numpy()                 # (32,32,3)
            recon = _reconstruct_rgb(inp, pred, target_channel)   # (32,32,3)
            diff  = np.abs(orig - recon)                           # (32,32,3)

            class_name = CIFAR10_CLASSES[label] if label < len(CIFAR10_CLASSES) else str(label)

            axes[0, c].imshow(np.clip(orig, 0, 1))
            axes[1, c].imshow(np.clip(recon, 0, 1))
            im = axes[2, c].imshow(diff.mean(axis=2),  # mean across channels
                                   vmin=0, vmax=0.3, cmap="hot")

            # Per-column header: class name + MSE
            axes[0, c].set_title(f"{class_name}\nMSE={mse_v:.4f}",
                                  fontsize=7, pad=3)

            for r in range(n_rows):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                for spine in axes[r, c].spines.values():
                    spine.set_visible(False)

        col += n_per_tier

    # Colorbar positioned below all images (outside subplot area)
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])   # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Mean absolute error per pixel", fontsize=9)

    fig.suptitle(
        f"Prediction Performance Matrix  —  target channel: {target_channel}"
        f"  (inputs: {', '.join(ch for ch in 'RGB' if ch != target_channel)})",
        fontsize=12, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0.04, 0.15, 1.0, 0.96])  # Leave space at bottom for colorbar

    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = cfg.OUTPUTS_DIR / f"performance_matrix_{target_channel}.png"
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Performance matrix saved → {save_path}")
    return save_path


# ─── Per-class metrics ────────────────────────────────────────────────────────

@torch.no_grad()
def plot_per_class_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    target_channel: str,
) -> Path:
    """
    Compute and plot per-class MSE and PSNR for each of the 10 CIFAR-10 classes.

    Produces a side-by-side horizontal bar chart sorted from best (lowest MSE)
    to worst, making it easy to see which object categories are easiest/hardest
    to reconstruct.

    Saves to outputs/per_class_metrics_{target_channel}.png and also logs a
    formatted table to the console.
    """
    # ── Collect per-image results ──────────────────────────────────────────────
    class_mse: Dict[int, list[float]]  = {i: [] for i in range(10)}
    class_psnr: Dict[int, list[float]] = {i: [] for i in range(10)}

    for x_batch, y_batch, lbl_batch, _ in loader:
        preds = model(x_batch.to(cfg.DEVICE)).cpu().numpy()   # (B, 1, 32, 32)
        tgts  = y_batch.numpy()                                # (B, 1, 32, 32)
        for i in range(len(x_batch)):
            p2d  = preds[i, 0]   # (32, 32)
            t2d  = tgts[i, 0]
            mse  = float(np.mean((p2d - t2d) ** 2))
            psnr = float(psnr_fn(t2d, p2d, data_range=1.0))
            lbl  = int(lbl_batch[i])
            class_mse[lbl].append(mse)
            class_psnr[lbl].append(psnr)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    class_names = CIFAR10_CLASSES
    mean_mse  = [float(np.mean(class_mse[i]))  for i in range(10)]
    mean_psnr = [float(np.mean(class_psnr[i])) for i in range(10)]
    counts    = [len(class_mse[i]) for i in range(10)]

    # Sort from best (lowest MSE) → worst
    order = np.argsort(mean_mse)   # ascending MSE

    sorted_names = [class_names[i] for i in order]
    sorted_mse   = [mean_mse[i]    for i in order]
    sorted_psnr  = [mean_psnr[i]   for i in order]
    sorted_counts = [counts[i]     for i in order]

    # ── Console table ──────────────────────────────────────────────────────────
    logger.info(f"Per-class metrics  (target='{target_channel}')")
    hdr = f"{'Class':<12} {'N':>5} {'MSE':>10} {'PSNR (dB)':>11}"
    logger.info("-" * len(hdr))
    logger.info(hdr)
    logger.info("-" * len(hdr))
    for n, mse_v, psnr_v, cnt in zip(sorted_names, sorted_mse, sorted_psnr, sorted_counts):
        logger.info(f"{n:<12} {cnt:>5} {mse_v:>10.6f} {psnr_v:>11.4f}")
    logger.info("-" * len(hdr))

    # ── Plot ───────────────────────────────────────────────────────────────────
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = cfg.OUTPUTS_DIR / f"per_class_metrics_{target_channel}.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Colour gradient: green (best) → red (worst)
    n_cls  = len(sorted_names)
    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, n_cls))   # type: ignore[attr-defined]

    # MSE bar chart (lower = better)
    bars1 = ax1.barh(sorted_names, sorted_mse, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Mean MSE (lower is better)", fontsize=10)
    ax1.set_title(f"Per-class MSE  (target={target_channel})", fontsize=11, fontweight="bold")
    ax1.invert_yaxis()   # best at top
    for bar, v in zip(bars1, sorted_mse):
        ax1.text(bar.get_width() + max(sorted_mse) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:.5f}", va="center", ha="left", fontsize=8)
    ax1.set_xlim(0, max(sorted_mse) * 1.25)
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # PSNR bar chart (higher = better) — same class order as MSE plot
    colors_psnr = plt.cm.RdYlGn(np.linspace(0.85, 0.15, n_cls))  # type: ignore[attr-defined]
    bars2 = ax2.barh(sorted_names, sorted_psnr, color=colors_psnr, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Mean PSNR in dB (higher is better)", fontsize=10)
    ax2.set_title(f"Per-class PSNR  (target={target_channel})", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()
    for bar, v in zip(bars2, sorted_psnr):
        ax2.text(bar.get_width() + max(sorted_psnr) * 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", ha="left", fontsize=8)
    ax2.set_xlim(0, max(sorted_psnr) * 1.12)
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Per-Class Prediction Quality  —  target channel: {target_channel}\n"
        f"(sorted by MSE, best → worst; n per class shown in parentheses)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-class metrics saved → {save_path}")
    return save_path


# ─── All-channel comparison table ────────────────────────────────────────────

def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print a comparison table of metrics for all three target channels."""
    header = f"{'Channel':>9} | {'MSE':>10} | {'MAE':>10} | {'PSNR (dB)':>10} | {'SSIM':>10}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for ch, m in results.items():
        print(
            f"{ch:>9} | {m['MSE']:>10.6f} | {m['MAE']:>10.6f} | "
            f"{m['PSNR']:>10.4f} | {m['SSIM']:>10.4f}"
        )
    print(sep)


# ─── Full evaluation pipeline ─────────────────────────────────────────────────

def evaluate(target_channel: str = cfg.TARGET_CHANNEL) -> Dict[str, float]:
    """
    Run complete evaluation for a single target channel:
      1. Load model.
      2. Build test DataLoader.
      3. Compute metrics.
      4. Save sample comparisons.
      5. Plot channel distributions.

    Returns the metrics dictionary.
    """
    logger.info(f"=== Evaluating target='{target_channel}' ===")
    model = load_model(target_channel)
    _, _, test_loader = build_dataloaders(target_channel)

    metrics = compute_metrics(model, test_loader, target_channel)
    save_sample_comparisons(model, test_loader, target_channel)
    plot_channel_distributions(test_loader, target_channel)
    plot_performance_matrix(model, test_loader, target_channel)
    plot_per_class_metrics(model, test_loader, target_channel)
    return metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ColorPredictor")
    parser.add_argument(
        "--target", type=str, default=cfg.TARGET_CHANNEL,
        choices=["R", "G", "B"],
        help="Target channel to evaluate (default: %(default)s)."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all three channels and print comparison table."
    )
    args = parser.parse_args()

    if args.all:
        results: Dict[str, Dict[str, float]] = {}
        for ch in ["B", "R", "G"]:
            results[ch] = evaluate(ch)
        print_comparison_table(results)
    else:
        metrics = evaluate(args.target)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
