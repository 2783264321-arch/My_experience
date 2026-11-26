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

## Environment (AutoDL instance, verified 2026-11-26)

- The host reports 128 vCPU / ~1 TiB RAM, but the **rental quota** is
  16 vCPU (`cpu.max=1600000 100000`) and 120 GiB RAM
  (`memory.max=128849018880`), plus one RTX 4090 24 GB. Always read the
  cgroup; `nproc`/`free -h` can overstate the instance.
- Use **16 workers** (12–16) for IQ/STFT/crop extraction. Do not run many
  CPU-heavy merges in parallel; 4 already oversubscribe 16 cores.
- `/root/autodl-tmp` is fast but small (50 GB, often nearly full).
  `/root/autodl-fs` is 200 GB shared storage. Keep code/worktrees on
  `autodl-tmp`; write `.npz` caches, checkpoints and large reports to
  `autodl-fs`. Check `df -h` first.
- Network: `source /etc/network_turbo` enables the AutoDL academic proxy
  (`172.29.51.4:12798`) for github/huggingface; it is unstable, per-shell, and
  slows pip. Do not block on `git clone`/`webfetch`; use GitHub MCP
  `get_file_contents` or the configured SSH remote. Never leave the proxy
  sourced for package installs unless needed.
- Use OpenCode features: `execute` Code Mode for tool orchestration, `search`
  for tool discovery, background shell for long jobs, read-only subagents for
  review, worktrees for isolation, skills/ledger for durable lessons.

## Efficiency rules

- Use process pools sized to the real CPU count (32–64 here) for IQ/STFT/crops,
  cache repeated transforms, and overlap CPU preprocessing with GPU inference.
- Measure wall time and CPU/GPU utilization; a 100% CPU plateau is one core,
  not "the machine is busy". Check `ps -eo pcpu,pmem,etime,cmd --sort=-pcpu`.
- Prefer vectorized masked operations over per-box Python loops.
- Reuse verified candidate pools and result caches; never rerun a forward for a
  post-processing-only comparison.
- Write large artifacts to `autodl-fs`, not the nearly-full `autodl-tmp`.

## Self-improvement loop

After every experiment, add one lesson to the ledger. If the lesson changes how
future experiments should be run, update this skill. The goal is cumulative
process improvement: fewer repeated mistakes, cheaper tests, and clearer
stopping conditions.

## Parallelization pattern (verified 2026-11-26)

- Split on the **independent unit** (sample, GT, crop). The worker takes a
  small index and returns a small row/count; never pass a 742k-candidate
  array per task.
- Set large read-only arrays as module globals **before** creating the
  pool. On Linux use `mp_context="fork"` so workers share them
  copy-on-write.
- Pin each worker to one BLAS/torch thread (`OMP_NUM_THREADS=1`,
  `torch.set_num_threads(1)`); otherwise 16 workers × 8 threads oversubscribe
  the 16-vCPU quota.
- Use `chunksize=8..16`, 16 workers, and measure 1/4/8/16/32 before trusting
  a speedup.
- Measured: 742k-candidate Soft-NMS over 1543 val samples = **6.3 s** with
  16 workers vs ~4 min serial; per-GT NMS/Soft/WBF decomposition = ~2 min at
  ~1360% CPU (the full 16-core quota).
- Do not run several CPU-heavy merges concurrently beyond the quota. Keep
  GPU inference in one process with large batches; use DataLoader workers
  only for CPU preprocessing.
- Reusable helper: `src/parallel.py`; benchmark:
  `scripts/benchmark_parallel_merge.py`.
