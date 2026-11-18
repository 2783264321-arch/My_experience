#!/usr/bin/env bash
# Wait for Baseline-1 GPU/CPU work to finish, run the coarse smoke, then run
# Group A training exclusively. Logs are kept outside the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export SPACENET_STFT_BACKEND=scipy
export SPACENET_FFT_THREADS=1

wait_free() {
  while true; do
    local busy=0
    if pgrep -f "scripts/train_local_representation_classifiers.py" >/dev/null 2>&1; then
      busy=1
    fi
    if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]; then
      busy=1
    fi
    if [ "$busy" -eq 0 ]; then
      sleep 30
      if ! pgrep -f "scripts/train_local_representation_classifiers.py" >/dev/null 2>&1 && \
         [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]; then
        echo "GPU free and no Baseline-1 classifier process; starting."
        return 0
      fi
    fi
    echo "waiting: baseline busy at $(date '+%H:%M:%S')"
    sleep 30
  done
}

wait_free

# Small smoke: 64 fit sources, 20 epochs, on-the-fly coarse views.
.venv/bin/python -u -m src.signals.train \
  --run smoke_coarse_64 --view-mode coarse \
  --epochs 20 --batch-sources 16 --accum 1 --target-views 16 \
  --workers 8 --val-workers 4 --val-batch-views 8 --val-every 10 \
  --precision bf16 --seed 1729 --limit-sources 64 \
  --min-free-gib 5 --print-every-s 10 --overwrite \
  2>&1 | tee /tmp/group_a_smoke.log

wait_free

# Formal Group A run.
scripts/run_group_a.sh 2>&1 | tee /tmp/group_a_train.log
