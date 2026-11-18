#!/usr/bin/env bash
# Regression gates for the coarse-only code path.
# C all-view must stay 0.6427; C coarse-only must reproduce 0.5794.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export SPACENET_FFT_THREADS=1
C=/root/autodl-tmp/workspace/baseline/runs/yolo11s_c1_directreg_2048x512/best_corrected.pt
OUT=runs/regression
mkdir -p "$OUT"

.venv/bin/python scripts/validate_run.py \
  --run yolo11s_c1_directreg_2048x512 --checkpoint "$C" \
  --view-mode all --workers 8 --batch-views 8 \
  --output "$OUT/c_all"

.venv/bin/python scripts/validate_run.py \
  --run yolo11s_c1_directreg_2048x512 --checkpoint "$C" \
  --view-mode coarse --workers 8 --batch-views 8 \
  --output "$OUT/c_coarse"

.venv/bin/python scripts/analyze_signal_characteristics.py \
  --run yolo11s_c1_directreg_2048x512 --val-dir "$OUT/c_all"
.venv/bin/python scripts/analyze_signal_characteristics.py \
  --run yolo11s_c1_directreg_2048x512 --val-dir "$OUT/c_coarse"

.venv/bin/python scripts/class_group_metrics.py \
  --val-dir "$OUT/c_all" --name C_all \
  --output reports/group_c_class_groups.json \
  --markdown reports/group_c_class_groups.md
.venv/bin/python scripts/class_group_metrics.py \
  --val-dir "$OUT/c_coarse" --name B_coarse \
  --output reports/group_b_class_groups.json \
  --markdown reports/group_b_class_groups.md
