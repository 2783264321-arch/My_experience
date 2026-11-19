#!/usr/bin/env bash
# Replication/control: dense seed 1730 and zero-complex 3ch controls.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
COMMON=(--total-steps 20000 --batch-sources 16 --workers 8
        --val-every-steps 1280 --early-stop-evals 0)
wait_file() {
  local p="$1"
  while [ ! -f "$p" ]; do echo "waiting for $p"; sleep 30; done
}
.venv/bin/python -u scripts/train_c3_subset.py \
  --run c3_pilot_zero3_s1729 --frontend zero3 --channels 3 \
  "${COMMON[@]}" --seed 1729 --overwrite 2>&1 | tee /tmp/c3_zero3_s1729.log
.venv/bin/python -u scripts/train_c3_subset.py \
  --run c3_pilot_zero3_s1730 --frontend zero3 --channels 3 \
  "${COMMON[@]}" --seed 1730 --overwrite 2>&1 | tee /tmp/c3_zero3_s1730.log
wait_file cache/c3_dense/meta.json
.venv/bin/python -u scripts/train_c3_subset.py \
  --run c3_pilot_dense_s1730 --frontend dense --channels 3 \
  --complex-dir cache/c3_dense \
  "${COMMON[@]}" --seed 1730 --overwrite 2>&1 | tee /tmp/c3_dense_s1730.log
echo "C3 replication/control complete at $(date '+%H:%M:%S')"
