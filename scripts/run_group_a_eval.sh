#!/usr/bin/env bash
# Evaluate Group A with the corrected coarse-only protocol.
# Usage: scripts/run_group_a_eval.sh [checkpoint] [tag] [name]
#   checkpoint: last.pt (default) or best.pt
#   tag:        output subdirectory under runs/<run>/
#   name:       label in class_group_metrics output
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export SPACENET_FFT_THREADS=1
RUN=yolo11s_c1_directreg_coarseonly_2048x512
CKPT="${1:-last.pt}"
TAG="${2:-stage5_val}"
NAME="${3:-A_last}"
OUT="runs/${RUN}/${TAG}"

.venv/bin/python scripts/validate_run.py \
  --run "$RUN" --checkpoint "$CKPT" --view-mode coarse \
  --workers 8 --batch-views 8 --output "$OUT"

.venv/bin/python scripts/analyze_signal_characteristics.py \
  --run "$RUN" --val-dir "$OUT"

.venv/bin/python scripts/class_group_metrics.py \
  --val-dir "$OUT" --name "$NAME" \
  --output "reports/group_a_class_groups_${TAG}.json" \
  --markdown "reports/group_a_class_groups_${TAG}.md"
