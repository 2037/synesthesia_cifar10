"""
synesthesia_cifar10.src
-----------------------
Public re-exports for convenient notebook usage.
"""

from src.config import DEVICE, TARGET_CHANNEL, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
from src.model import ColorPredictor
from src.data_loader import CIFARColorDataset, build_dataloaders, prepare_data
from src.trainer import train
from src.evaluator import evaluate, compute_metrics, save_sample_comparisons, print_comparison_table
from src.utils import get_device, save_model, load_model, plot_loss_curves

__all__ = [
    "DEVICE",
    "TARGET_CHANNEL",
    "BATCH_SIZE",
    "NUM_EPOCHS",
    "LEARNING_RATE",
    "ColorPredictor",
    "CIFARColorDataset",
    "build_dataloaders",
    "prepare_data",
    "train",
    "evaluate",
    "compute_metrics",
    "save_sample_comparisons",
    "print_comparison_table",
    "get_device",
    "save_model",
    "load_model",
    "plot_loss_curves",
]
