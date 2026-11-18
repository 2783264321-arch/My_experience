#!/usr/bin/env bash
# Final Group A evaluation: last EMA (primary), best EMA (secondary), latency,
# comparison report, training curve. Run only when the GPU is otherwise idle.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export SPACENET_FFT_THREADS=1
RUN=yolo11s_c1_directreg_coarseonly_2048x512
C_ROOT=/root/autodl-tmp/workspace/baseline/runs/yolo11s_c1_directreg_2048x512

scripts/run_group_a_eval.sh last.pt stage5_val A_last
scripts/run_group_a_eval.sh best.pt stage5_val_best A_best

.venv/bin/python scripts/bench_view_modes.py \
  --run-a "$RUN" --checkpoint-a last.pt \
  --run-c yolo11s_c1_directreg_2048x512 --checkpoint-c "$C_ROOT/best_corrected.pt" \
  --samples 128 --repeats 3 --batch-views 8 \
  --output reports/group_a_latency.json

.venv/bin/python scripts/compare_group_a.py \
  --a-dir "runs/$RUN/stage5_val" \
  --b-dir runs/regression/c_coarse \
  --c-dir "$C_ROOT/stage5_val" \
  --a-groups reports/group_a_class_groups_stage5_val.json \
  --b-groups reports/group_b_class_groups.json \
  --c-groups reports/group_c_class_groups.json \
  --latency reports/group_a_latency.json \
  --a-run "runs/$RUN" \
  --c-run-root "$C_ROOT" \
  --cache-meta cache/coarse_fit_bf16.json \
  --cache-verify reports/group_a_cache_verification.json \
  --output-md reports/group_a_results.md \
  --output-json reports/group_a_results.json

.venv/bin/python scripts/plot_training.py --run "$RUN"
cp "runs/$RUN/training_curve.png" reports/group_a_training_curve.png
