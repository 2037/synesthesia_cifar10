"""
data_loader.py — Dataset download, loading, custom Dataset, and DataLoaders.

Data source
-----------
CIFAR-10 Python version from the University of Toronto:
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

No authentication required.  The archive extracts to:
  data/cifar-10-batches-py/
    batches.meta
    data_batch_1  ...  data_batch_5   (training, 10 000 images each)
    test_batch                         (test,     10 000 images)

Each batch file is a Python pickle dict with keys:
  b'data'      : np.ndarray of shape (10000, 3072), dtype uint8
                 Layout: [R-plane (1024) | G-plane (1024) | B-plane (1024)]
  b'labels'    : list of 10000 ints  (0–9)
  b'filenames' : list of 10000 bytes

Steps performed when run as __main__:
  1. Download the .tar.gz from the Toronto URL if not already present.
  2. Extract it into data/.
  3. Load all 50 000 training images → float32 tensors in [0, 1].
  4. Build CIFARColorDataset and split 80 / 10 / 10.
  5. Wrap in three DataLoaders.
"""

from __future__ import annotations

import logging
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import src.config as cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Channel index mapping ────────────────────────────────────────────────────
CHANNEL_IDX: dict[str, int] = {"R": 0, "G": 1, "B": 2}


# ─── Step 1 & 2: Download and extract ────────────────────────────────────────

def _show_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Simple download progress callback for urllib.request.urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        print(f"\r  Downloading … {pct:5.1f}%  ({downloaded // 1_048_576} MB)", end="", flush=True)


def download_and_extract(url: str = cfg.CIFAR_URL, dest_dir: Path = cfg.DATA_DIR) -> Path:
    """
    Download the CIFAR-10 tar.gz from *url* and extract into *dest_dir*.

    Skips the download if ``cifar-10-batches-py/`` already exists.

    Returns the path to ``cifar-10-batches-py/``.
    """
    cifar_dir = dest_dir / "cifar-10-batches-py"
    if cifar_dir.exists() and any(cifar_dir.iterdir()):
        logger.info(f"CIFAR-10 already extracted at {cifar_dir} — skipping download.")
        return cifar_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "cifar-10-python.tar.gz"

    if not archive_path.exists():
        logger.info(f"Downloading CIFAR-10 from {url} …")
        urllib.request.urlretrieve(url, archive_path, reporthook=_show_progress)
        print()   # newline after progress bar
        logger.info(f"Saved archive → {archive_path}")
    else:
        logger.info(f"Archive already downloaded at {archive_path} — skipping download.")

    logger.info(f"Extracting {archive_path.name} → {dest_dir} …")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    logger.info(f"Extraction complete → {cifar_dir}")
    return cifar_dir


# ─── Step 3: Load pickle batches ──────────────────────────────────────────────

def _load_batch(batch_path: Path) -> Tuple[np.ndarray, list]:
    """
    Load one CIFAR-10 pickle batch file.

    Returns
    -------
    images : np.ndarray of shape (N, 3, 32, 32), dtype float32, values in [0, 1].
    labels : list of N ints (class indices 0–9).
    """
    with open(batch_path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    # d[b'data'] : (N, 3072) uint8
    # Layout: first 1024 values = R, next 1024 = G, last 1024 = B
    raw = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = list(d[b"labels"])
    return raw, labels


def load_class_names(cifar_dir: Path = cfg.CIFAR_DIR) -> List[str]:
    """
    Read the 10 CIFAR-10 class names from batches.meta.

    Returns
    -------
    list of 10 strings, e.g. ['airplane', 'automobile', ..., 'truck']
    """
    meta_path = cifar_dir / "batches.meta"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return [name.decode("utf-8") for name in meta[b"label_names"]]


def load_all_images(cifar_dir: Path = cfg.CIFAR_DIR) -> Tuple[torch.Tensor, List[int]]:
    """
    Load all 50 000 training images and their class labels.

    Returns
    -------
    images : torch.Tensor of shape (50000, 3, 32, 32), float32, values in [0, 1].
    labels : list of 50000 ints (class indices 0–9).
    """
    arrays: list[np.ndarray] = []
    all_labels: list[int] = []
    for name in cfg.TRAIN_BATCHES:
        path = cifar_dir / name
        logger.info(f"Loading {path.name} …")
        imgs, lbls = _load_batch(path)
        arrays.append(imgs)
        all_labels.extend(lbls)
    data = np.concatenate(arrays, axis=0)          # (50000, 3, 32, 32)
    tensor = torch.from_numpy(data)
    logger.info(f"Loaded {len(tensor):,} images  shape={tuple(tensor.shape)}")
    return tensor, all_labels


def load_test_images(cifar_dir: Path = cfg.CIFAR_DIR) -> Tuple[torch.Tensor, List[int]]:
    """
    Load all 10 000 test images and their class labels from test_batch.

    Returns
    -------
    images : torch.Tensor of shape (10000, 3, 32, 32), float32, values in [0, 1].
    labels : list of 10000 ints (class indices 0–9).
    """
    imgs, labels = _load_batch(cifar_dir / cfg.TEST_BATCH)
    tensor = torch.from_numpy(imgs)
    logger.info(f"Loaded {len(tensor):,} test images  shape={tuple(tensor.shape)}")
    return tensor, labels


# ─── Step 4: Custom Dataset ───────────────────────────────────────────────────

class CIFARColorDataset(Dataset):
    """
    Synesthesia CIFAR-10 regression dataset.

    Given full RGB image tensors, returns tuples of:
      - input  : (2, 32, 32) — the two channels that are NOT the target
      - target : (1, 32, 32) — the single channel to predict
      - label  : int          — CIFAR-10 class index (0–9)
      - image  : (3, 32, 32) — full original RGB image (for reconstruction)

    Parameters
    ----------
    images : torch.Tensor of shape (N, 3, 32, 32)
    labels : list of N ints (class indices 0–9)
    target_channel : str — one of 'R', 'G', 'B'
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: List[int],
        target_channel: str = "B",
    ) -> None:
        assert target_channel in CHANNEL_IDX, (
            f"target_channel must be one of {list(CHANNEL_IDX)}, got '{target_channel}'."
        )
        assert len(images) == len(labels), "images and labels must have the same length."
        self.images = images
        self.labels = labels
        self.target_channel = target_channel
        self.target_idx = CHANNEL_IDX[target_channel]
        self.input_indices = [i for i in range(3) if i != self.target_idx]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        img = self.images[idx]                                    # (3, 32, 32)
        input_tensor  = img[self.input_indices, :, :]             # (2, 32, 32)
        target_tensor = img[self.target_idx, :, :].unsqueeze(0)  # (1, 32, 32)
        label         = self.labels[idx]                          # int
        return input_tensor, target_tensor, label, img


# ─── Step 5: DataLoaders ──────────────────────────────────────────────────────

def build_dataloaders(
    target_channel: str = cfg.TARGET_CHANNEL,
    cifar_dir: Path = cfg.CIFAR_DIR,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders for the given target channel.

    Splits the 50 000 training images 80 / 10 / 10 with seed 42.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    images, labels = load_all_images(cifar_dir)
    dataset = CIFARColorDataset(images, labels, target_channel=target_channel)

    n_total = len(dataset)
    n_train = int(n_total * cfg.TRAIN_RATIO)
    n_val   = int(n_total * cfg.VAL_RATIO)
    n_test  = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(cfg.RANDOM_SEED)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    logger.info(
        f"Split — train: {n_train:,}  val: {n_val:,}  test: {n_test:,}  "
        f"target='{target_channel}'"
    )

    common = dict(
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)

    return train_loader, val_loader, test_loader


# ─── Prepare data (download if needed) ───────────────────────────────────────

def prepare_data() -> Path:
    """
    Ensure CIFAR-10 data is downloaded and extracted.
    Returns the path to cifar-10-batches-py/.
    """
    return download_and_extract(cfg.CIFAR_URL, cfg.DATA_DIR)


# ─── Smoke test when run directly ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== data_loader.py — smoke test ===")

    # 1. Download / verify data
    cifar_dir = prepare_data()

    # 2. Build loaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg.TARGET_CHANNEL, cifar_dir)

    # 3. Inspect a batch
    x_batch, y_batch, labels_batch, imgs_batch = next(iter(train_loader))
    logger.info(f"Input  batch shape : {x_batch.shape}")      # (128, 2, 32, 32)
    logger.info(f"Target batch shape : {y_batch.shape}")      # (128, 1, 32, 32)
    logger.info(f"Labels batch shape : {len(labels_batch)}")  # 128
    logger.info(f"Images batch shape : {imgs_batch.shape}")   # (128, 3, 32, 32)
    logger.info(f"Input  range       : [{x_batch.min():.3f}, {x_batch.max():.3f}]")
    logger.info(f"Target range       : [{y_batch.min():.3f}, {y_batch.max():.3f}]")
    logger.info("data_loader.py smoke test passed ✓")
