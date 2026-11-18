# Group A - full-band single-image baseline

Run: `yolo11s_c1_directreg_coarseonly_2048x512`
Worktree: `/root/autodl-tmp/worktrees/baseline-0-group-a`
Status: **completed** (100 epochs, no early stopping)

## Protocol

- One normalized `2048x512` full-band coarse view per fit source; 0.7 ms
  periodic Hann STFT, native `F` frequency bins aggregated to 2048 rows by the
  existing area-weighted mean.
- Model: YOLO11s body, C=1, nc=14, `reg_max=1`, `end2end=False` (DirectReg),
  same `configs/yolo11s_c1_directreg.yaml` as C.
- Initialization: `yolo11s.pt` body transfer (same summed C1 stem), independent
  training; **not** initialized from C.
- Inference: one coarse view per sample; no fine slices, no boundary filtering.
- Primary checkpoint: last-epoch EMA (per the pre-registered A/C rule).
  `best.pt` is reported separately; it is the early-stop-style model.
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

## Training result

| metric | A last (ep99) | A best (ep29) |
| --- | ---: | ---: |
| mAP@0.5 | 0.5404 | **0.5763** |
| AP50 agnostic | 0.8020 | **0.8208** |
| localization recall | 0.8570 | **0.8789** |
| classification accuracy | 0.8328 | 0.8270 |
| predictions | 16,173 | 30,981 |
| val views | 1,543 | 1,543 |
| val time | 76 s | 82 s |

Validation history (last EMA, coarse-only):

| epoch | 9 | 19 | 29 | 39 | 49 | 59 | 69 | 79 | 89 | 99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mAP50 | .4405 | .5546 | **.5763** | .5745 | .5611 | .5557 | .5508 | .5468 | .5436 | .5403 |
| AP_agn | .7590 | .8195 | **.8208** | .8185 | .8177 | .8135 | .8114 | .8100 | .8084 | .8019 |
| R_agn | .8443 | **.8843** | .8789 | .8746 | .8721 | .8690 | .8657 | .8641 | .8615 | .8569 |

The model peaks at epoch 29 and then overfits: training loss falls from 66.5
to ~13, while val mAP/AP_agn/recall all decline. Last is **3.59 pp** below best.

## Convergence / early stopping

This A run intentionally has **no early stopping**: it completes the planned
100 epochs as the equal-budget, no-early-stop reference.

Future experiments will use **4 validation events** without a >=0.001 mAP50
improvement (`--early-stop-evals 4 --early-stop-min-delta 0.001
--early-stop-min-evals 3`), about 14,920 optimizer steps. This is 1.2x the
mature 3-event/15-epoch window: on A it would stop at epoch 69 (best epoch 29,
saving 30 epochs); on C it would stop at epoch 49 (the corrected best), not
before. Future runs select the **best EMA** checkpoint and also report last EMA,
stopped epoch, and actual optimizer steps.

## Cost

| item | C | A |
| --- | ---: | ---: |
| training wall time | 2.69 h | **1.62 h** |
| peak reserved | 10.1 GiB | 10.1 GiB |
| cache build | none | 2,049.6 s / 11.63 GiB |
| A including cache | — | ~2.19 h |
| data loader throughput | — | ~400 samples/s (8 workers) |

A's shorter wall time is partly the cheaper coarse-only validation (~77 s vs
~180 s per validation) and the cache; the model sees the same 595,700 images
and makes 37,300 vs 37,250 optimizer steps.

## Artifacts

- `runs/yolo11s_c1_directreg_coarseonly_2048x512/`
- `reports/group_a_results.md` / `.json`
- `reports/group_a_latency.json`
- `reports/group_a_training_curve.png`
- `reports/group_a_class_groups_stage5_val*.json`
- `docs/collaboration/opencode-group-a-fullband-baseline.md`
