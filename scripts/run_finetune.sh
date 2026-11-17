#!/usr/bin/env bash
# Post-baseline follow-up: 15-epoch classification-focused fine-tune from the
# first baseline's best checkpoint. This is NOT the frozen first baseline; it
# is reported separately. Same split/data/two-view definition.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export SPACENET_STFT_BACKEND=scipy
export SPACENET_FFT_THREADS=2
exec .venv/bin/python -u -m src.signals.train \
  --run yolo11s_c1_directreg_2048x512_clsft \
  --epochs 15 \
  --batch-sources 8 \
  --accum 1 \
  --target-views 16 \
  --workers 12 \
  --val-workers 12 \
  --val-batch-views 8 \
  --val-every 3 \
  --precision bf16 \
  --seed 1729 \
  --lr0 0.0002 \
  --weight-decay 0.0005 \
  --cls-weight 1.0 \
  --warmup 0.5 \
  --final-lr-frac 0.01 \
  --init-checkpoint runs/yolo11s_c1_directreg_2048x512/best.pt \
  "$@"
