"""
config.py — Central configuration for Synesthesia CIFAR-10 project.

All hyperparameters, paths, and device settings live here so that
every other module imports from one authoritative source.
"""

import os
import torch
from pathlib import Path

# ─── Device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent   # synesthesia_cifar10/
DATA_DIR: Path = ROOT_DIR / "data"

# The Toronto pickle format lives here after download+extraction
CIFAR_DIR: Path = DATA_DIR / "cifar-10-batches-py"

# Batch file names inside CIFAR_DIR
TRAIN_BATCHES: list = [f"data_batch_{i}" for i in range(1, 6)]   # 50,000 images
TEST_BATCH: str = "test_batch"                                      # 10,000 images

# Download URL (no authentication required)
CIFAR_URL: str = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

MODELS_DIR: Path = ROOT_DIR / "models"
LOGS_DIR: Path = ROOT_DIR / "logs"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

# Create output directories if they do not exist
for _d in (MODELS_DIR, LOGS_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Data ─────────────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10          # implied (1 - TRAIN - VAL)

# ─── DataLoader ───────────────────────────────────────────────────────────────
# Use smaller batch size on MPS (Apple Silicon) for stability
BATCH_SIZE: int = 64 if str(DEVICE) == "mps" else 128
# On MPS, num_workers > 0 can cause issues; fall back to 0.
NUM_WORKERS: int = 0 if str(DEVICE) == "mps" else 4
PIN_MEMORY: bool = str(DEVICE) != "mps"   # pin_memory not supported on MPS

# ─── Model ────────────────────────────────────────────────────────────────────
# Which channel to predict: 'R', 'G', or 'B'
# Default: predict Blue from Red + Green (inputs = R, G → target = B)
TARGET_CHANNEL: str = "B"

# ─── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 5                # set low for quick demo; increase for full run

# ReduceLROnPlateau scheduler parameters (provides implicit early stopping):
# - patience: epochs to wait before reducing LR when val_loss plateaus
# - factor: multiply LR by this value when reducing (e.g., 0.5 = halve LR)
# The patience mechanism prevents overfitting: when validation loss stops
# improving, LR is reduced, and eventually learning naturally stops.
LR_SCHEDULER_PATIENCE: int = 3
LR_SCHEDULER_FACTOR: float = 0.5

# ─── Evaluation ───────────────────────────────────────────────────────────────
NUM_SAMPLE_IMAGES: int = 10       # number of comparison images to save
