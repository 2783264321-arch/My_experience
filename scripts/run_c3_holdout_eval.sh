#!/usr/bin/env bash
# Evaluate frozen dense/zero3 best+last on the 1,159-sample val holdout.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
VAL_IDS=cache/c3_holdout/holdout.json
POWER=cache/c3_holdout/val_power_bf16.npy
DENSE=cache/c3_holdout_dense
wait_for() { while [ ! -f "$1" ]; do echo "waiting $1"; sleep 30; done; }
wait_for "${POWER%.npy}.json"
wait_for "$DENSE/meta.json"

run_eval() {
  local run=$1 front=$2 ck=$3 out=$4
  .venv/bin/python scripts/eval_c3_checkpoint.py \
    --run "$run" --checkpoint "$ck" --frontend "$front" \
    --val-ids-file "$VAL_IDS" --power-cache "$POWER" \
    $( [ "$front" = "dense" ] && echo --complex-dir "$DENSE" ) \
    --out "$out"
}
run_eval c3_pilot_dense dense best.pt runs/c3_pilot_dense/eval_holdout_best
run_eval c3_pilot_dense dense last.pt runs/c3_pilot_dense/eval_holdout_last
run_eval c3_pilot_dense_s1730 dense best.pt runs/c3_pilot_dense_s1730/eval_holdout_best
run_eval c3_pilot_dense_s1730 dense last.pt runs/c3_pilot_dense_s1730/eval_holdout_last
run_eval c3_pilot_zero3_s1729 zero3 best.pt runs/c3_pilot_zero3_s1729/eval_holdout_best
run_eval c3_pilot_zero3_s1729 zero3 last.pt runs/c3_pilot_zero3_s1729/eval_holdout_last
run_eval c3_pilot_zero3_s1730 zero3 best.pt runs/c3_pilot_zero3_s1730/eval_holdout_best
run_eval c3_pilot_zero3_s1730 zero3 last.pt runs/c3_pilot_zero3_s1730/eval_holdout_last

for pair in "c3_pilot_dense best c3_pilot_zero3_s1729 best s1729_best" \
            "c3_pilot_dense last c3_pilot_zero3_s1729 last s1729_last" \
            "c3_pilot_dense_s1730 best c3_pilot_zero3_s1730 best s1730_best" \
            "c3_pilot_dense_s1730 last c3_pilot_zero3_s1730 last s1730_last"; do
  set -- $pair; dr=$1; dc=$2; zr=$3; zc=$4; tag=$5
  .venv/bin/python scripts/analyze_c3_matched.py \
    --a runs/$dr/eval_holdout_$dc.npz --b runs/$zr/eval_holdout_$zc.npz \
    --a-name dense_$tag --b-name zero3_$tag \
    --output reports/c3_holdout_matched_$tag.json
done
echo "holdout eval complete at $(date '+%H:%M:%S')"
