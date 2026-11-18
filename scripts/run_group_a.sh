#!/usr/bin/env bash
# Group A: 100 epochs, one full-band coarse view per fit source.
# Budget parity with C: 16 views/step, 37,300 vs 37,250 optimizer steps,
# 595,700 training views, warmup 6 epochs == 2,238 steps.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export SPACENET_STFT_BACKEND=scipy
# Training reads the BF16 cache; validation rebuilds float32 val views.
export SPACENET_FFT_THREADS=1

exec .venv/bin/python -u -m src.signals.train \
  --run yolo11s_c1_directreg_coarseonly_2048x512 \
  --view-mode coarse \
  --coarse-cache cache/coarse_fit_bf16.npy \
  --epochs 100 \
  --batch-sources 16 \
  --accum 1 \
  --target-views 16 \
  --workers 8 \
  --val-workers 8 \
  --val-batch-views 8 \
  --val-every 10 \
  --precision bf16 \
  --seed 1729 \
  --lr0 0.001 \
  --weight-decay 0.0005 \
  --warmup 6 \
  --final-lr-frac 0.01 \
  --clip-grad 10.0 \
  --min-free-gib 8 \
  --print-every-s 30 \
  "$@"
