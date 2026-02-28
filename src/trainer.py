"""
trainer.py — Training loop with early stopping and LR scheduling.

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


# ─── Early stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitors validation loss and stops training when it stops improving.

    Parameters
    ----------
    patience : int
        How many epochs to wait after last improvement.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ─── One epoch helpers ────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
    desc: str = "",
) -> float:
    """Run a single training or validation epoch and return the mean loss."""
    model.train(train)
    total_loss = 0.0

    bar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
    with torch.set_grad_enabled(train):
        for x_batch, y_batch, *_ in bar:   # *_ discards labels & full-rgb tensors
            x_batch = x_batch.to(device, non_blocking=cfg.PIN_MEMORY)
            y_batch = y_batch.to(device, non_blocking=cfg.PIN_MEMORY)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            bar.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / len(loader.dataset)   # type: ignore[arg-type]


# ─── Main training function ───────────────────────────────────────────────────

def load_from_checkpoint(target_channel: str) -> ColorPredictor:
    """
    Load and return the best saved model for *target_channel*.

    Raises FileNotFoundError if no checkpoint exists.
    """
    ckpt = cfg.MODELS_DIR / f"best_model_{target_channel}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt}. "
            "Run with --force-train to train from scratch."
        )
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
    Full training pipeline with optional checkpoint reuse.

    Parameters
    ----------
    target_channel : str
        Channel to predict ('R', 'G', or 'B').
    num_epochs : int
        Maximum number of training epochs.
    lr : float
        Initial learning rate for Adam.
    force_train : bool
        If False (default) and a saved checkpoint already exists, skip
        training and load the checkpoint directly.  Set to True to always
        retrain from scratch.

    Returns
    -------
    model : ColorPredictor
        Best model (freshly trained or loaded from checkpoint).
    train_losses : list of float
        Per-epoch training losses (empty list when loaded from checkpoint).
    val_losses : list of float
        Per-epoch validation losses (empty list when loaded from checkpoint).
    """
    best_model_path = cfg.MODELS_DIR / f"best_model_{target_channel}.pth"

    # ── Checkpoint shortcut ────────────────────────────────────────────────────
    if not force_train and best_model_path.exists():
        logger.info(
            f"Checkpoint found at {best_model_path}. "
            "Skipping training (use --force-train to retrain)."
        )
        model = load_from_checkpoint(target_channel)
        return model, [], []

    logger.info(f"=== Training  target='{target_channel}'  device={cfg.DEVICE} ===")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(target_channel)

    # ── Model / loss / optimizer / scheduler ──────────────────────────────────
    model = ColorPredictor().to(cfg.DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.LR_SCHEDULER_FACTOR,
        patience=cfg.LR_SCHEDULER_PATIENCE,
    )
    early_stopper = EarlyStopping(
        patience=cfg.EARLY_STOP_PATIENCE,
        min_delta=cfg.EARLY_STOP_MIN_DELTA,
    )

    best_val_loss = float("inf")
    # best_model_path already defined above the checkpoint-shortcut block
    train_losses: list[float] = []
    val_losses: list[float] = []

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = cfg.LOGS_DIR / f"train_{target_channel}.log"
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")

    def _log(msg: str) -> None:
        logger.info(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    _log(f"Epochs={num_epochs}  LR={lr}  Target='{target_channel}'  "
         f"Batch={cfg.BATCH_SIZE}  Device={cfg.DEVICE}")
    _log(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch_bar = tqdm(range(1, num_epochs + 1), desc=f"Training target='{target_channel}'",
                     unit="epoch", dynamic_ncols=True)
    for epoch in epoch_bar:
        t0 = time.time()
        train_loss = _run_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE,
            train=True,  desc=f"  Train {epoch:03d}/{num_epochs}",
        )
        val_loss = _run_epoch(
            model, val_loader, criterion, None, cfg.DEVICE,
            train=False, desc=f"  Val   {epoch:03d}/{num_epochs}",
        )
        elapsed = time.time() - t0
        epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}",
                               lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        early_stopper(val_loss)

        _log(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            _log(f"  ↳ New best model saved to {best_model_path}")

        if early_stopper.should_stop:
            _log(f"Early stopping triggered at epoch {epoch}.")
            break

    _log(f"Training complete. Best val_loss={best_val_loss:.6f}")
    log_fh.close()

    # ── Load best weights before returning ────────────────────────────────────
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
