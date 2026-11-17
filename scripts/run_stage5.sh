#!/usr/bin/env bash
# Stage-5 deliverable generation for the first baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
RUN="${1:-yolo11s_c1_directreg_2048x512}"
CKPT="${2:-best_corrected.pt}"
.venv/bin/python -u scripts/validate_run.py --run "$RUN" --checkpoint "$CKPT" --workers 12
.venv/bin/python -u scripts/analyze_val.py --run "$RUN"
.venv/bin/python -u scripts/export_bt_table.py --run "$RUN"
.venv/bin/python -u scripts/predict_examples.py --run "$RUN" --checkpoint "$CKPT" --limit 10
.venv/bin/python -u scripts/bench_inference.py --run "$RUN" --checkpoint "$CKPT" --repeats 3
.venv/bin/python -u scripts/plot_training.py --run "$RUN"
echo "stage5 artifacts written under runs/$RUN/"
