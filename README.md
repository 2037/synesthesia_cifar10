# Synesthesia CIFAR-10 — Color Channel Prediction

> **Assignment 2** | Deep Learning | CIFAR-10 Color Channel Regression

---

## Overview

Synesthesia is a neurological phenomenon where stimulation of one sensory pathway
leads to involuntary experiences in a second. This project simulates that idea by
training a CNN to predict one missing RGB channel of a CIFAR-10 image given the
other two — e.g. predict **Blue** from **Red + Green**.

---

## Project Structure

```
synesthesia_cifar10/
│
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── run_all.sh                        # End-to-end pipeline script
│
├── src/                              # All source code
│   ├── __init__.py                   # Public re-exports for notebook use
│   ├── config.py                     # Central config: device, paths, hyperparameters
│   ├── data_loader.py                # Download, extraction, Dataset, DataLoaders
│   ├── model.py                      # ColorPredictor CNN (BN + Dropout toggleable)
│   ├── trainer.py                    # Training loop, early stopping, checkpoint save/load
│   ├── evaluator.py                  # Metrics (MSE/MAE/PSNR/SSIM) + all plots
│   ├── experiments.py                # Hyperparameter sweep (Part 3 ablation study)
│   └── utils.py                      # Shared helpers: model I/O, plot utilities
│
├── data/                             # Created automatically on first run
│   └── cifar-10-batches-py/          # Extracted CIFAR-10 pickle batches
│       ├── batches.meta              # Class name lookup (10 class names)
│       ├── data_batch_1 … 5          # 50,000 training images (10k each)
│       └── test_batch                # 10,000 test images
│
├── models/                           # Saved model checkpoints
│   └── best_model_B.pth              # Best weights for target=Blue
│
├── logs/                             # Per-run log files
│   ├── data.log                      # Data download/verify output
│   ├── train_B.log                   # Training output (also printed to terminal)
│   ├── eval_B.log                    # Evaluation output
│   └── sweep_B.log                   # Hyperparameter sweep output (--sweep only)
│
├── outputs/                          # All saved plots and figures
│   ├── sample_grid.png               # 2×8 grid of raw CIFAR-10 samples
│   ├── channel_splits.png            # One image split into R / G / B channels
│   ├── channel_dist_B.png            # Pixel-value histograms per channel
│   ├── loss_curve_B.png              # Train vs. val MSE loss over epochs
│   ├── samples_B.png                 # 10 test images: Original | Recon | |Diff|
│   ├── performance_matrix_B.png      # Best / Mean / Worst prediction matrix
│   ├── per_class_metrics_B.png       # Per-class MSE & PSNR bar charts (Part 4)
│   ├── hyperparameter_comparison_B.png  # Ablation bar chart + learning curves (--sweep)
│   └── hyperparameter_results_B.csv     # Numeric ablation results CSV (--sweep)
│
└── notebooks/
    └── synesthesia_cifar10_final.ipynb   # Jupyter notebook (edit manually)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Default: loads existing checkpoint if present, skips training
./run_all.sh

# Force a full retrain from scratch (ignores any existing checkpoint)
./run_all.sh --force-train

# Run the hyperparameter ablation sweep after normal training + evaluation
./run_all.sh --sweep

# Combine: retrain + sweep
./run_all.sh --force-train --sweep
```

`run_all.sh` runs the following sequential steps:

| Step | Command | Output |
|------|---------|--------|
| 1 — Data | `python -m src.data_loader` | Downloads & extracts CIFAR-10 if needed → `data/` |
| 2 — Train | `python -m src.trainer --target B [--force-train]` | Trains model, saves `models/best_model_B.pth`, logs to `logs/train_B.log` |
| 3 — Evaluate | `python -m src.evaluator --target B` | Computes metrics, saves all plots to `outputs/`, logs to `logs/eval_B.log` |
| 4 — Sweep *(optional)* | `python -m src.experiments --target B` | Trains 7 configs, saves comparison plot + CSV to `outputs/`, logs to `logs/sweep_B.log` |

> **Data download**: `run_all.sh` fetches CIFAR-10 automatically from
> `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` — no account or
> authentication required. If `data/cifar-10-batches-py/` already exists the
> download is skipped entirely.

### 3. Run individual modules

Each `src/` module can be run directly for fine-grained control:

```bash
# Data only
python -m src.data_loader

# Train (with options)
python -m src.trainer --target B --epochs 30 --lr 1e-3
python -m src.trainer --target B --force-train   # ignore existing checkpoint

# Evaluate
python -m src.evaluator --target B

# Hyperparameter sweep (7 configs × NUM_EPOCHS each)
python -m src.experiments --target B --epochs 10
```

---

## Outputs Explained

| File | Description |
|------|-------------|
| `outputs/sample_grid.png` | 16 random CIFAR-10 training images |
| `outputs/channel_splits.png` | One image decomposed into R, G, B channels |
| `outputs/channel_dist_B.png` | Pixel intensity histograms for each channel |
| `outputs/loss_curve_B.png` | Train / validation MSE loss vs. epoch |
| `outputs/samples_B.png` | 10 test examples: Original · Reconstructed · \|Diff\| |
| `outputs/performance_matrix_B.png` | **Best / Mean / Worst** prediction matrix with CIFAR-10 class labels and per-image MSE |
| `outputs/per_class_metrics_B.png` | Horizontal bar charts of mean MSE and PSNR for each of the 10 CIFAR-10 classes, sorted best → worst |
| `outputs/hyperparameter_comparison_B.png` | Side-by-side bar chart (best val MSE) and learning curves for all 7 ablation configs |
| `outputs/hyperparameter_results_B.csv` | Full numeric ablation results (name, LR, BN, dropout, best val MSE, time) |

---

## Model Architecture

`ColorPredictor` is a fully-convolutional encoder that preserves 32×32 spatial
dimensions throughout via `padding='same'`. Both BatchNorm and Dropout are
independently toggleable for ablation studies.

| Block  | Layers                          | In ch | Out ch | Kernel |
|--------|---------------------------------|-------|--------|--------|
| Block 1 | Conv → BatchNorm → LeakyReLU → Dropout | 2 | 64 | 3×3 |
| Block 2 | Conv → BatchNorm → LeakyReLU → Dropout | 64 | 128 | 3×3 |
| Block 3 | Conv → BatchNorm → LeakyReLU → Dropout | 128 | 64 | 3×3 |
| Output  | Conv → Sigmoid                  | 64    | 1      | 1×1    |

- **Input**: `(B, 2, 32, 32)` — two known RGB channels
- **Output**: `(B, 1, 32, 32)` — predicted channel, values ∈ [0, 1]
- **Weight init**: Kaiming-normal for Conv, ones/zeros for BatchNorm

---

## Hyperparameters

All defaults live in `src/config.py` and can be overridden via CLI flags.

| Parameter | Default | Where to change |
|-----------|---------|-----------------|
| Target channel | `B` (predict Blue from R+G) | `config.py → TARGET_CHANNEL` |
| Batch size | 128 | `config.py → BATCH_SIZE` |
| Learning rate | 1e-3 | `config.py → LEARNING_RATE` |
| Optimizer | Adam | `trainer.py` |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) | `config.py` |
| Early stopping patience | 10 epochs | `config.py → EARLY_STOP_PATIENCE` |
| Max epochs | 5 (demo — increase for full training) | `config.py → NUM_EPOCHS` |
| Train / Val / Test split | 80 / 10 / 10 % | `config.py → TRAIN_RATIO / VAL_RATIO` |
| Random seed | 42 | `config.py → RANDOM_SEED` |

### Hyperparameter Ablation Sweep (`src/experiments.py`)

Seven configurations are tested by default to document the effect of each choice:

| # | Configuration | LR | BatchNorm | Dropout | p |
|---|---------------|----|-----------|---------|---|
| 1 | Baseline | 1e-3 | ✓ | ✓ | 0.2 |
| 2 | High LR | 1e-2 | ✓ | ✓ | 0.2 |
| 3 | Low LR | 1e-4 | ✓ | ✓ | 0.2 |
| 4 | No BatchNorm | 1e-3 | ✗ | ✓ | 0.2 |
| 5 | No Dropout | 1e-3 | ✓ | ✗ | 0.0 |
| 6 | No BN + No Dropout | 1e-3 | ✗ | ✗ | 0.0 |
| 7 | High Dropout | 1e-3 | ✓ | ✓ | 0.4 |

Results are ranked by best validation MSE and saved to
`outputs/hyperparameter_comparison_B.png` and `outputs/hyperparameter_results_B.csv`.

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | mean((ŷ − y)²) | Lower is better |
| MAE | mean(\|ŷ − y\|) | Lower is better |
| PSNR | 10·log₁₀(1/MSE) dB | Higher is better; >30 dB is good |
| SSIM | Structural similarity ∈ [−1, 1] | Closer to 1 is better |

### Per-Class Analysis (`plot_per_class_metrics`)

After overall evaluation, per-class MSE and PSNR are computed for each of the
10 CIFAR-10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse,
ship, truck) and plotted as horizontal bar charts sorted from best to worst. This
identifies which types of images or regions the model handles well or poorly —
e.g. uniform backgrounds (ships, airplanes) tend to be easier than textured
foregrounds (cats, dogs).

---

## Dependencies

```
torch torchvision torchaudio   # Deep learning
numpy pillow                   # Image loading and array ops
matplotlib seaborn             # Plotting
scikit-image                   # PSNR / SSIM metrics
tqdm                           # Training progress bars
pandas                         # (optional) tabular analysis
nbformat                       # Notebook utilities
```

No external download tools required — data fetched via Python's built-in `urllib`.
