#!/bin/bash
# run_all.sh — Synesthesia CIFAR-10 pipeline (target: Blue from Red + Green)
# --------------------------------------------------------------------------
# Usage:
#   ./run_all.sh                  # load existing checkpoint; skip training if found
#   ./run_all.sh --force-train    # always retrain from scratch
#   ./run_all.sh --sweep          # also run the hyperparameter sweep after evaluation
#   ./run_all.sh --force-train --sweep   # retrain + sweep

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────────────────
FORCE_TRAIN=""
RUN_SWEEP=""
for arg in "$@"; do
    case "$arg" in
        --force-train) FORCE_TRAIN="--force-train" ;;
        --sweep)       RUN_SWEEP="yes" ;;
        *) echo "Unknown argument: $arg  (valid: --force-train, --sweep)"; exit 1 ;;
    esac
done

# ── macOS / Apple Silicon: suppress duplicate OpenMP runtime error ────────────
# PyTorch and NumPy can both ship libomp, causing an abort on macOS.
# KMP_DUPLICATE_LIB_OK=TRUE is the standard workaround; it is safe for training.
export KMP_DUPLICATE_LIB_OK=TRUE

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

N_STEPS=3
[ -n "$RUN_SWEEP" ] && N_STEPS=4

echo "========================================"
echo " Synesthesia CIFAR-10  —  run_all.sh"
echo " Target: predict Blue from Red + Green"
echo " Mode  : ${FORCE_TRAIN:+force retrain}${FORCE_TRAIN:-load checkpoint if available}"
[ -n "$RUN_SWEEP" ] && echo " Sweep : hyperparameter ablation enabled"
echo " $(date)"
echo "========================================"

# ── 0. Create output directories ─────────────────────────────────────────────
mkdir -p logs outputs models data

# ── 1. Data download & extraction ────────────────────────────────────────────
# Downloads from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# (no authentication required).  Skipped automatically if already extracted.
echo ""
echo "[1/${N_STEPS}] Verifying / downloading CIFAR-10 data …"
python -m src.data_loader > logs/data.log 2>&1
echo "      Done. See logs/data.log"

# ── 2. Train — target channel B (Blue) ───────────────────────────────────────
echo ""
echo "[2/${N_STEPS}] Training — predict Blue from Red + Green …"
python -m src.trainer --target B $FORCE_TRAIN 2>&1 | tee logs/train_B.log
echo "      Done. See logs/train_B.log"

# ── 3. Evaluate — target channel B ───────────────────────────────────────────
# Produces: samples_B.png, channel_dist_B.png, loss_curve_B.png (if retrained),
#           performance_matrix_B.png, per_class_metrics_B.png
echo ""
echo "[3/${N_STEPS}] Evaluating — target=B …"
python -m src.evaluator --target B > logs/eval_B.log 2>&1
echo "      Done. See logs/eval_B.log"
echo "      Plots saved to outputs/"

# ── 4. (Optional) Hyperparameter sweep ───────────────────────────────────────
if [ -n "$RUN_SWEEP" ]; then
    echo ""
    echo "[4/${N_STEPS}] Hyperparameter sweep (7 configurations × ${NUM_EPOCHS_SWEEP:-5} epochs) …"
    # Uses cfg.NUM_EPOCHS by default; pass --epochs N to override
    python -m src.experiments --target B 2>&1 | tee logs/sweep_B.log
    echo "      Done. See logs/sweep_B.log"
    echo "      Results: outputs/hyperparameter_comparison_B.png"
    echo "               outputs/hyperparameter_results_B.csv"
fi

echo ""
echo "========================================"
echo " All runs completed."
echo " Logs    → logs/"
echo " Plots   → outputs/"
echo " Model   → models/best_model_B.pth"
echo ""
echo " Key outputs:"
echo "   outputs/samples_B.png              — Original vs Reconstructed vs |Diff|"
echo "   outputs/performance_matrix_B.png   — Best / Mean / Worst predictions"
echo "   outputs/per_class_metrics_B.png    — Per-class MSE & PSNR bar charts"
echo "   outputs/channel_dist_B.png         — Channel pixel distributions"
if [ -n "$RUN_SWEEP" ]; then
echo "   outputs/hyperparameter_comparison_B.png — Ablation study comparison"
echo "   outputs/hyperparameter_results_B.csv    — Full numeric results"
fi
echo "========================================"
