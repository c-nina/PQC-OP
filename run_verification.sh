#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_verification.sh — Pre-launch verification sample
#
# Two modes selectable with --mode:
#
#   quick  (default) — 9 experiments per method
#          M1: arch 7×7, N=2500, 3 reps, 3 strikes         =  9 runs
#          M2: arch 5×5, N=5000, 3 reps, 3 strikes         =  9 runs
#          Total: 18 runs.  Goal: compare M2 errors vs baseline (5.58 ATM).
#
#   full   — 36 experiments per method (all dataset sizes)
#          M1: arch 7×7, 3 reps × 3 strikes × 4 N sizes    = 36 runs
#          M2: arch 5×5, 3 reps × 3 strikes × 4 N sizes    = 36 runs
#          Total: 72 runs.
#
# Usage:
#   bash run_verification.sh cuda              # quick mode on GPU  (default)
#   bash run_verification.sh cuda full         # full  mode on GPU
#   bash run_verification.sh cpu               # quick mode on CPU  (slow)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DEVICE="${1:-cpu}"
MODE="${2:-quick}"
RESULTS_DIR="results/verification"
LOG_DIR="results/verification/logs"
SRC_DIR="$(dirname "$0")/src"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  QML Verification run"
echo "  Device   : $DEVICE"
echo "  Mode     : $MODE"
echo "  Output   : $RESULTS_DIR"
echo "  Started  : $(date)"
echo "============================================================"

if [[ "$MODE" == "quick" ]]; then
    M1_N_REPS=3
    M1_DATASETS=""          # only N=2500 for quick mode (largest M1 size)
    M2_N_REPS=3
    M2_DATASETS=""          # only N=5000 for quick mode (second-largest M2 size)
    M1_EXTRA="--datasets 2500"
    M2_EXTRA="--datasets 5000"
    M1_DESC="arch 7×7, N=2500, 3 reps × 3 strikes = 9 runs"
    M2_DESC="arch 5×5, N=5000, 3 reps × 3 strikes = 9 runs"
else
    M1_EXTRA=""
    M2_EXTRA=""
    M1_DESC="arch 7×7, 3 reps × 3 strikes × 4 datasets = 36 runs"
    M2_DESC="arch 5×5, 3 reps × 3 strikes × 4 datasets = 36 runs"
fi

# ── Method I real (7×7) ───────────────────────────────────────────────────────
echo ""
echo ">>> Method I real — $M1_DESC"
python "$SRC_DIR/run_experiments.py" \
    --methods 1 \
    --n_reps 3 \
    --architectures 7x7 \
    --device "$DEVICE" \
    --results_dir "$RESULTS_DIR" \
    $M1_EXTRA \
    2>&1 | tee "$LOG_DIR/M1_verification.log"

# ── Method II (5×5) ──────────────────────────────────────────────────────────
echo ""
echo ">>> Method II — $M2_DESC"
python "$SRC_DIR/run_experiments.py" \
    --methods 2 \
    --n_reps 3 \
    --architectures 5x5 \
    --device "$DEVICE" \
    --results_dir "$RESULTS_DIR" \
    $M2_EXTRA \
    2>&1 | tee "$LOG_DIR/M2_verification.log"

echo ""
echo "============================================================"
echo "  Both methods finished: $(date)"
echo "  Analysing results..."
echo "============================================================"

python "$SRC_DIR/../analyze_verification.py" --results_dir "$RESULTS_DIR"
