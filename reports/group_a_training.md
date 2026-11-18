# Group A - full-band single-image baseline

Run: `yolo11s_c1_directreg_coarseonly_2048x512`
Worktree: `/root/autodl-tmp/worktrees/baseline-0-group-a`
Status: **waiting for exclusive GPU** (protocol/cache/gates complete)

## Protocol

- One normalized `2048x512` full-band coarse view per fit source; 0.7 ms
  periodic Hann STFT, native `F` frequency bins aggregated to 2048 rows by the
  existing area-weighted mean.
- Model: YOLO11s body, C=1, nc=14, `reg_max=1`, `end2end=False` (DirectReg),
  same `configs/yolo11s_c1_directreg.yaml` as C.
- Initialization: `yolo11s.pt` body transfer (same summed C1 stem), independent
  training; **not** initialized from C.
- Inference: one coarse view per sample; no fine slices, no boundary filtering.
- Primary checkpoint: last-epoch EMA. `best.pt` is reported separately.
- Seed 1729; BF16 autocast; AdamW; EMA 0.9999/tau 2000.

## Budget parity

| item | C (joint) | A (single-image) |
| --- | ---: | ---: |
| views per source per epoch | 1 coarse + 1 rotating fine | 1 coarse |
| epochs | 50 | 100 |
| physical batch | 8 sources = 16 views | 16 sources = 16 views |
| effective views | 16 | 16 |
| optimizer steps | 745 x 50 = 37,250 | 373 x 100 = 37,300 |
| training image presentations | 595,700 | 595,700 |
| warmup | 3 ep = 2,235 steps | 6 ep = 2,238 steps |
| cosine tail | 47 ep | 94 ep |
| validation cadence | every 5 ep = 3,725 steps | every 10 ep = 3,730 steps |

The two budgets differ by **+0.13%** optimizer steps and are equal in image
presentations. A sees each source's coarse view 100 times; C sees it 50 times
plus 50 rotating fine slices.

## Data cache

- `cache/coarse_fit_bf16.npy`: `5957x1x2048x512` BF16 bit patterns, 11.63 GiB.
- Build time: **2049.6 s** (single thread, overlapped with Baseline-1 CPU work).
- 32-sample verification: BF16 bit patterns exact (`max_abs_diff=0`), boxes and
  labels exact.
- Training-path check on 4 sources with the same `yolo11s.pt` weights and the
  same batch:
  - online repeat noise: loss 0, preds 0, grads 0;
  - cache vs online: loss diff 0, preds diff 0, grad max diff 0, grad rel-L2 0,
    cosine 1.0;
  - image BF16 bit patterns exact; targets exact.
- Artifact: `reports/group_a_cache_verification.json`.

## Regression gates

| path | expected | measured | views | predictions |
| --- | ---: | ---: | ---: | ---: |
| C all views | 0.6427 | **0.6427008648** | 24,975 | 65,025 |
| C coarse only (B) | 0.5794 | **0.5793865644** | 1,543 | 25,409 |

Both match the historical protocol exactly. B class-group AP: 0/1/2 0.4212,
3/4/5 0.4550, 10/11 0.5908, 9/12/13 0.4838.

## Training cost

- Pending A run. Expected ~2.2-2.5 h on the RTX 4090, ~80-90 s/epoch plus
  coarse-only validation (~10 x 1,543 views).
- Data-loader throughput on the cache: ~400 samples/s with 8 workers; training
  is GPU-bound, not STFT-bound.
- Cache build is a one-time 34 min CPU cost; it is reported separately from the
  training budget.

## Convergence and results

Pending. The final report will include the training curve, last-epoch EMA and
best EMA metrics, class groups, physical-bandwidth groups, and latency.
