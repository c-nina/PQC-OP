#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_verification.sh — Pre-launch verification sample (36 + 36 experiments)
#
# Intermediate architecture only (7x7 for M1, 5x5 for M2), 3 reps, 3 strikes,
# all dataset sizes. Saves results to results/verification/.
#
# Usage:
#   bash run_verification.sh            # CPU (default)
#   bash run_verification.sh cuda       # GPU
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DEVICE="${1:-cpu}"
RESULTS_DIR="results/verification"
LOG_DIR="results/verification/logs"
SRC_DIR="$(dirname "$0")/src"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  QML Verification run"
echo "  Device   : $DEVICE"
echo "  Output   : $RESULTS_DIR"
echo "  Started  : $(date)"
echo "============================================================"

# ── Method I real (7×7, paper Sec. 3.2.1) ────────────────────────────────────
echo ""
echo ">>> Method I real — arch 7x7, 3 reps × 3 strikes × 4 datasets = 36 runs"
python "$SRC_DIR/run_experiments.py" \
    --methods 1 \
    --n_reps 3 \
    --architectures 7x7 \
    --device "$DEVICE" \
    --results_dir "$RESULTS_DIR" \
    2>&1 | tee "$LOG_DIR/M1_verification.log"

echo ""
echo ">>> Method II — arch 5x5, 3 reps × 3 strikes × 4 datasets = 36 runs"
python "$SRC_DIR/run_experiments.py" \
    --methods 2 \
    --n_reps 3 \
    --architectures 5x5 \
    --device "$DEVICE" \
    --results_dir "$RESULTS_DIR" \
    2>&1 | tee "$LOG_DIR/M2_verification.log"

echo ""
echo "============================================================"
echo "  Both methods finished: $(date)"
echo "  Analysing results..."
echo "============================================================"

python "$SRC_DIR/../analyze_verification.py" --results_dir "$RESULTS_DIR"
