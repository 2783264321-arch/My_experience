---
name: SpaceNet Experiment Workflow
description: Use when planning, running, reviewing, or reporting a SpaceNet/D-FINE experiment, or when improving the agent's own efficiency on this project.
---

# SpaceNet experiment workflow

## Before starting

1. Read `docs/collaboration/README.md`, `docs/collaboration/spacenet-first-principles.md`,
   and `docs/collaboration/experiment-ledger.md`.
2. Write one sentence for each: hypothesis, baseline, single changed variable,
   pass/fail gate, compute cost, and what will be reused.
3. Check the ledger for dead ends before repeating a direction.
4. Use a dedicated branch/worktree for the experiment. Keep `test` sealed.
5. Prefer fixed candidate caches, precomputed features, and one model forward
   reused across post-processing variants.

## During the run

- Smoke test first; assert zero-residual/equivalence and shapes.
- Use a read-only subagent review before a long training or full val.
- Parallelize CPU extraction across up to 16 workers; keep GPU fed; avoid
  single-core loops for NMS/AP where vectorization or process pools are possible.
- Log progress at milestones: job start, data/model ready, inference/training
  complete, analysis complete, commit/push complete. Include ETA and blockers.
- Never tune on test. Use `fit` for training/calibration and fixed `val` for
  decisions.

## After the run

1. Report mAP50, mAP50:95, per-class and group AP, high-IoU/overlap behavior,
   extra latency, and candidate counts.
2. Compare against the correct control, not only the old baseline.
3. Update `docs/collaboration/experiment-ledger.md`:
   - confirmed win or dead end;
   - what assumption failed or held;
   - one process/efficiency lesson.
4. If negative, state exactly what was falsified. Do not say the whole direction
   is dead unless the representation and integration were both tested.
5. If positive, recommend exactly one next change and keep the old baseline as a
   fallback.

## Efficiency rules

- 16 vCPU, large RAM, one RTX 4090: use process pools for IQ/STFT/crops, cache
  repeated transforms, and overlap CPU preprocessing with GPU inference.
- Measure wall time and CPU/GPU utilization; a 100% CPU plateau is one core,
  not "the machine is busy".
- Prefer vectorized masked operations over per-box Python loops.
- Reuse verified candidate pools and result caches; never rerun a forward for a
  post-processing-only comparison.

## Self-improvement loop

After every experiment, add one lesson to the ledger. If the lesson changes how
future experiments should be run, update this skill. The goal is cumulative
process improvement: fewer repeated mistakes, cheaper tests, and clearer
stopping conditions.
