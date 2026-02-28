"""
trainer.py — Training loop with LR scheduling and patience-based early stopping.

This module implements the main training pipeline for the ColorPredictor model.
Key features:
  - Checkpoint saving/loading to avoid retraining
  - ReduceLROnPlateau scheduler with patience to prevent overfitting
  - Progress tracking with tqdm and detailed logging

Usage (CLI):
    python -m src.trainer --target B
    python -m src.trainer --target R
    python -m src.trainer --target G
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.config as cfg
from src.model import ColorPredictor
from src.data_loader import build_dataloaders

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Single epoch training/validation ─────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
    desc: str = "",
) -> float:
    """
    Run a single training or validation epoch.
    
    Parameters
    ----------
    model : nn.Module
        The ColorPredictor model to train/evaluate.
    loader : DataLoader
        Training or validation data loader.
    criterion : nn.Module
        Loss function (MSE for pixel-wise regression).
    optimizer : torch.optim.Optimizer | None
        Optimizer for training. None during validation.
    device : torch.device
        Device to run computations on (CPU/CUDA/MPS).
    train : bool
        If True, run in training mode with gradient updates.
    desc : str
        Description for the progress bar.
        
    Returns
    -------
    float
        Mean loss across all batches in the epoch.
    """
    # Set model to training or evaluation mode
    model.train(train)
    total_loss = 0.0
    num_batches = len(loader)
    
    # Detect MPS device for special handling
    is_mps = str(device) == "mps"

    # Create progress bar with reduced update frequency for MPS stability
    bar = tqdm(loader, desc=desc, leave=False, unit="batch", 
               dynamic_ncols=True, mininterval=1.0 if is_mps else 0.1)
    
    with torch.set_grad_enabled(train):
        for batch_idx, (x_batch, y_batch, *_) in enumerate(bar):   # *_ discards labels & full-rgb tensors
            # Move data to target device (blocking transfers for MPS to avoid hangs)
            x_batch = x_batch.to(device, non_blocking=False)
            y_batch = y_batch.to(device, non_blocking=False)

            # Forward pass: predict target channel
            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            # Backward pass and optimization (training only)
            if train and optimizer is not None:
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()         # Compute gradients
                optimizer.step()        # Update weights

            # Extract loss value and accumulate (weighted by batch size for accurate averaging)
            loss_val = loss.item()
            total_loss += loss_val * x_batch.size(0)
            
            # Update progress bar less frequently for stability
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                bar.set_postfix(loss=f"{loss_val:.5f}")
            
            # Clear MPS cache every 20 batches to prevent memory buildup
            if is_mps and batch_idx % 20 == 0:
                torch.mps.empty_cache()
    
    # Final synchronization for MPS to ensure all operations complete
    if is_mps:
        torch.mps.synchronize()
        torch.mps.empty_cache()

    # Return mean loss across all samples
    return total_loss / len(loader.dataset)   # type: ignore[arg-type]


# ─── Checkpoint loading ───────────────────────────────────────────────────────

def load_from_checkpoint(target_channel: str) -> ColorPredictor:
    """
    Load and return the best saved model for the specified target channel.
    
    Parameters
    ----------
    target_channel : str
        The RGB channel this model was trained to predict ('R', 'G', or 'B').
    
    Returns
    -------
    ColorPredictor
        The loaded model in evaluation mode.
        
    Raises
    ------
    FileNotFoundError
        If no checkpoint exists for the specified channel.
    """
    ckpt = cfg.MODELS_DIR / f"best_model_{target_channel}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt}. "
            "Run with --force-train to train from scratch."
        )
    
    # Initialize model architecture and load saved weights
    model = ColorPredictor().to(cfg.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))
    model.eval()
    logger.info(f"Loaded checkpoint from {ckpt}")
    return model


def train(
    target_channel: str = cfg.TARGET_CHANNEL,
    num_epochs: int = cfg.NUM_EPOCHS,
    lr: float = cfg.LEARNING_RATE,
    force_train: bool = False,
) -> Tuple[ColorPredictor, list[float], list[float]]:
    """
    Full training pipeline with checkpoint reuse and patience-based early stopping.
    
    This function implements a complete training loop with several strategies to
    prevent overfitting:
    
    1. **Validation monitoring**: Tracks validation loss each epoch
    2. **Best checkpoint saving**: Only saves models that improve validation loss
    3. **LR scheduling with patience**: ReduceLROnPlateau reduces learning rate
       when validation loss plateaus (controlled by LR_SCHEDULER_PATIENCE config)
    4. **Implicit early stopping**: Very low learning rates naturally stop learning

    Parameters
    ----------
    target_channel : str
        Channel to predict ('R', 'G', or 'B'). Default from config.
    num_epochs : int
        Maximum number of training epochs. Default from config.
    lr : float
        Initial learning rate for Adam optimizer. Default from config.
    force_train : bool
        If False (default) and a saved checkpoint already exists, skip
        training and load the checkpoint directly. Set to True to always
        retrain from scratch (useful for hyperparameter experiments).

    Returns
    -------
    model : ColorPredictor
        Best model (freshly trained or loaded from checkpoint), in eval mode.
    train_losses : list of float
        Per-epoch training losses (empty list when loaded from checkpoint).
    val_losses : list of float
        Per-epoch validation losses (empty list when loaded from checkpoint).
    """
    best_model_path = cfg.MODELS_DIR / f"best_model_{target_channel}.pth"

    # ── Checkpoint shortcut: load existing model if available ──────────────────
    if not force_train and best_model_path.exists():
        logger.info(
            f"Checkpoint found at {best_model_path}. "
            "Skipping training (use --force-train to retrain)."
        )
        model = load_from_checkpoint(target_channel)
        return model, [], []

    logger.info(f"=== Training  target='{target_channel}'  device={cfg.DEVICE} ===")

    # ── Build data loaders ─────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(target_channel)

    # ── Initialize model, loss, optimizer, and scheduler ───────────────────────
    model = ColorPredictor().to(cfg.DEVICE)
    criterion = nn.MSELoss()  # Pixel-wise mean squared error for regression
    optimizer = Adam(model.parameters(), lr=lr)
    
    # ReduceLROnPlateau: reduces LR when validation loss plateaus
    # The 'patience' parameter provides implicit early stopping behavior:
    # - If val_loss doesn't improve for 'patience' epochs, LR is reduced
    # - After several reductions, LR becomes very small and learning stops
    # - This prevents overfitting by stopping before the model memorizes training data
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",                                  # Minimize validation loss
        factor=cfg.LR_SCHEDULER_FACTOR,             # LR multiplier on plateau
        patience=cfg.LR_SCHEDULER_PATIENCE,         # Epochs to wait before reducing LR
    )

    # Track best validation loss and checkpoint path
    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    # ── Setup logging to file ──────────────────────────────────────────────────
    log_path = cfg.LOGS_DIR / f"train_{target_channel}.log"
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")

    def _log(msg: str) -> None:
        """Helper to log both to console and file."""
        logger.info(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    _log(f"Epochs={num_epochs}  LR={lr}  Target='{target_channel}'  "
         f"Batch={cfg.BATCH_SIZE}  Device={cfg.DEVICE}")
    _log(f"LR Scheduler: patience={cfg.LR_SCHEDULER_PATIENCE}, "
         f"factor={cfg.LR_SCHEDULER_FACTOR}")
    _log(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Main training loop ─────────────────────────────────────────────────────
    epoch_bar = tqdm(range(1, num_epochs + 1), desc=f"Training target='{target_channel}'",
                     unit="epoch", dynamic_ncols=True)
    for epoch in epoch_bar:
        t0 = time.time()
        
        # Run training epoch (with gradient updates)
        train_loss = _run_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE,
            train=True,  desc=f"  Train {epoch:03d}/{num_epochs}",
        )
        
        # Run validation epoch (no gradient updates)
        val_loss = _run_epoch(
            model, val_loader, criterion, None, cfg.DEVICE,
            train=False, desc=f"  Val   {epoch:03d}/{num_epochs}",
        )
        
        elapsed = time.time() - t0
        epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}",
                               lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        # Record losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update learning rate based on validation loss plateau
        # This implements patience-based early stopping:
        # - If val_loss doesn't improve, LR is reduced after 'patience' epochs
        # - Eventually LR becomes too small for meaningful updates
        scheduler.step(val_loss)

        _log(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Save checkpoint only when validation loss improves (prevents overfitting)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            _log(f"  ↳ New best model saved to {best_model_path}")

    _log(f"Training complete. Best val_loss={best_val_loss:.6f}")
    log_fh.close()

    # ── Load best weights before returning ────────────────────────────────────
    # This ensures we return the model with lowest validation loss, not the
    # final epoch weights (which may have overfit)
    model.load_state_dict(torch.load(best_model_path, map_location=cfg.DEVICE))
    model.eval()

    return model, train_losses, val_losses


# ─── CLI entry-point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ColorPredictor")
    parser.add_argument(
        "--target", type=str, default=cfg.TARGET_CHANNEL,
        choices=["R", "G", "B"],
        help="Which colour channel to predict (default: %(default)s)."
    )
    parser.add_argument(
        "--epochs", type=int, default=cfg.NUM_EPOCHS,
        help="Maximum training epochs (default: %(default)s)."
    )
    parser.add_argument(
        "--lr", type=float, default=cfg.LEARNING_RATE,
        help="Initial learning rate (default: %(default)s)."
    )
    parser.add_argument(
        "--force-train", action="store_true", default=False,
        help="Retrain from scratch even if a checkpoint already exists."
    )
    args = parser.parse_args()

    model, train_losses, val_losses = train(
        target_channel=args.target,
        num_epochs=args.epochs,
        lr=args.lr,
        force_train=args.force_train,
    )
    if train_losses:
        logger.info(
            f"Final train loss: {train_losses[-1]:.6f}  "
            f"Final val loss: {val_losses[-1]:.6f}"
        )
