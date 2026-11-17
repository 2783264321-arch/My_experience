#!/usr/bin/env bash
# Stage-4 first 50-epoch run. Launch from the project root.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export SPACENET_STFT_BACKEND=scipy
export SPACENET_FFT_THREADS=2
exec .venv/bin/python -u -m src.signals.train \
  --run yolo11s_c1_directreg_2048x512 \
  --epochs 50 \
  --batch-sources 8 \
  --accum 1 \
  --target-views 16 \
  --workers 12 \
  --val-workers 12 \
  --val-batch-views 8 \
  --val-every 5 \
  --precision bf16 \
  --seed 1729 \
  --lr0 0.001 \
  --weight-decay 0.0005 \
  --warmup 3 \
  --final-lr-frac 0.01 \
  "$@"
