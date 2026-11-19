#!/usr/bin/env bash
# Sequential matched-step C3 pilot: C1 control, then overlap, sparse, dense.
# Assumes the overlap cache builder is already running (or the subset/val
# power metadata exists). One complex cache exists on disk at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export OMP_NUM_THREADS=4
ROOT=$(pwd)
COMMON=(--total-steps 20000 --batch-sources 16 --workers 8
        --val-every-steps 1280 --early-stop-evals 0 --seed 1729)

wait_file() {
  local path="$1"
  while [ ! -f "$path" ]; do
    echo "waiting for $path at $(date '+%H:%M:%S')"
    sleep 30
  done
}

# C1 can start as soon as the subset val power cache exists; the overlap cache
# builder can keep using the CPU in parallel.
wait_file "$ROOT/cache/c3_subset_val_power_bf16.json"
.venv/bin/python -u scripts/train_c3_subset.py \
  --run c3_pilot_c1 --frontend power --channels 1 \
  "${COMMON[@]}" --overwrite 2>&1 | tee /tmp/c3_c1.log

wait_file "$ROOT/cache/c3_overlap/meta.json"
cp "$ROOT/cache/c3_overlap/meta.json" "$ROOT/reports/c3_cache_meta_overlap.json"
.venv/bin/python -u scripts/train_c3_subset.py \
  --run c3_pilot_overlap --frontend overlap --channels 3 \
  --complex-dir cache/c3_overlap \
  "${COMMON[@]}" --overwrite 2>&1 | tee /tmp/c3_overlap.log
rm -rf "$ROOT/cache/c3_overlap"

.venv/bin/python -u scripts/build_c3_subset_cache.py \
  --frontend sparse --out cache/c3_sparse --overwrite \
  2>&1 | tee /tmp/c3_sparse_cache.log
cp "$ROOT/cache/c3_sparse/meta.json" "$ROOT/reports/c3_cache_meta_sparse.json"
.venv/bin/python -u scripts/train_c3_subset.py \
  --run c3_pilot_sparse --frontend sparse --channels 3 \
  --complex-dir cache/c3_sparse \
  "${COMMON[@]}" --overwrite 2>&1 | tee /tmp/c3_sparse.log
rm -rf "$ROOT/cache/c3_sparse"

# Build the dense cache on CPU and train the second C1 seed on GPU at the
# same time; this keeps both resources busy without sharing the GPU.
.venv/bin/python -u scripts/build_c3_subset_cache.py \
  --frontend dense --out cache/c3_dense --overwrite \
  2>&1 | tee /tmp/c3_dense_cache.log &
DENSE_CACHE_PID=$!
.venv/bin/python -u scripts/train_c3_subset.py \
  --run c3_pilot_c1_s1730 --frontend power --channels 1 \
  "${COMMON[@]}" --seed 1730 --overwrite 2>&1 | tee /tmp/c3_c1_s1730.log
wait "$DENSE_CACHE_PID"
cp "$ROOT/cache/c3_dense/meta.json" "$ROOT/reports/c3_cache_meta_dense.json"
.venv/bin/python -u scripts/train_c3_subset.py \
  --run c3_pilot_dense --frontend dense --channels 3 \
  --complex-dir cache/c3_dense \
  "${COMMON[@]}" --overwrite 2>&1 | tee /tmp/c3_dense.log
rm -rf "$ROOT/cache/c3_dense"

cp "$ROOT/cache/c3_subset.json" "$ROOT/reports/c3_subset.json"
echo "C3 pilot complete at $(date '+%H:%M:%S')"
