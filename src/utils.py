"""
utils.py — Shared utilities: model I/O, device helpers, and plot functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import src.config as cfg

logger = logging.getLogger(__name__)


# ─── Device helper ────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return the best available device in the order: CUDA > MPS > CPU.

    This mirrors the logic in config.py but is exposed as a callable
    for use in notebooks and external scripts.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Model I/O ────────────────────────────────────────────────────────────────

def save_model(model: nn.Module, path: Path) -> None:
    """
    Save model state dict to *path*.

    Parameters
    ----------
    model : nn.Module
    path  : Path  — destination file (parents created automatically)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved → {path}")


def load_model(model: nn.Module, path: Path, device: Optional[torch.device] = None) -> nn.Module:
    """
    Load state dict from *path* into *model* and return it.

    Parameters
    ----------
    model  : nn.Module  — must have matching architecture
    path   : Path
    device : torch.device, optional  — defaults to cfg.DEVICE
    """
    if device is None:
        device = cfg.DEVICE
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Model loaded ← {path}")
    return model


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    target_channel: str,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot train/val MSE loss curves and optionally save to *save_path*.

    Returns the resolved save path.
    """
    if save_path is None:
        cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = cfg.OUTPUTS_DIR / f"loss_curve_{target_channel}.png"

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train MSE", linewidth=2, color="steelblue")
    plt.plot(epochs, val_losses,   label="Val MSE",   linewidth=2, color="coral", linestyle="--")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title(f"Training & Validation Loss — target='{target_channel}'", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info(f"Loss curve saved → {save_path}")
    return save_path


def show_sample_images(
    images: List[torch.Tensor],
    n: int = 8,
    title: str = "Sample CIFAR-10 images",
    save_path: Optional[Path] = None,
) -> Path:
    """
    Display a row of *n* sample images (full RGB tensors of shape (3, 32, 32))
    and save the figure.
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = cfg.OUTPUTS_DIR / "sample_images.png"

    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(n * 1.8, 2.2))
    if n == 1:
        axes = [axes]

    for ax, img_tensor in zip(axes, images[:n]):
        img_np = img_tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
        ax.imshow(np.clip(img_np, 0, 1))
        ax.axis("off")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Sample images saved → {save_path}")
    return save_path


def show_channel_splits(
    image_tensor: torch.Tensor,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Display one image split into its R, G, B channels side-by-side.

    Parameters
    ----------
    image_tensor : torch.Tensor of shape (3, 32, 32)
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = cfg.OUTPUTS_DIR / "channel_splits.png"

    img_np = image_tensor.permute(1, 2, 0).numpy()   # (H, W, 3)
    channel_names = ["Red", "Green", "Blue"]
    cmaps = ["Reds_r", "Greens_r", "Blues_r"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(np.clip(img_np, 0, 1))
    axes[0].set_title("Full RGB")
    axes[0].axis("off")

    for i, (name, cmap) in enumerate(zip(channel_names, cmaps)):
        axes[i + 1].imshow(img_np[:, :, i], cmap=cmap, vmin=0, vmax=1)
        axes[i + 1].set_title(name)
        axes[i + 1].axis("off")

    plt.suptitle("RGB Channel Decomposition", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info(f"Channel splits saved → {save_path}")
    return save_path


def print_model_summary(model: nn.Module, input_shape: tuple = (1, 2, 32, 32)) -> None:
    """
    Print a simple parameter summary table for *model*.

    Parameters
    ----------
    model       : nn.Module
    input_shape : tuple — a sample input shape (B, C, H, W)
    """
    print("=" * 60)
    print(f"{'Layer':<30} {'Output Shape':>20}")
    print("=" * 60)
    device = next(model.parameters()).device
    dummy = torch.zeros(*input_shape, device=device)

    # Walk through named children and record output shapes
    def hook_fn(module: nn.Module, inp, out, name: str = "") -> None:
        shape_str = str(tuple(out.shape)) if hasattr(out, "shape") else "—"
        print(f"  {name:<28} {shape_str:>20}")

    hooks = []
    for name, layer in model.named_children():
        h = layer.register_forward_hook(
            lambda m, i, o, n=name: hook_fn(m, i, o, n)
        )
        hooks.append(h)

    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"  Total trainable parameters: {total:,}")
    print("=" * 60)
