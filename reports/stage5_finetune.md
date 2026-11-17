# Post-baseline experiment: classification-weighted fine-tune (not accepted)

Run: `yolo11s_c1_directreg_2048x512_clsft`
Directory: `runs/yolo11s_c1_directreg_2048x512_clsft/`
This is **not** the frozen first baseline; it is a separate follow-up.

## Setup

- initialized from the first baseline's `best.pt` (EMA epoch 29);
- 15 epochs, same data/split/two-view sampling and physical batch 8
  (16 views), BF16, AdamW lr0=2e-4, cosine to 1%, 0.5-epoch warmup;
- `cls` loss weight 1.0 (baseline 0.5);
- class-weighted BCE with weights
  `(mean_count / count)^0.5`, normalized to mean 1:
  `[1.038, 1.032, 1.019, 1.536, 1.562, 1.504, 0.695, 0.689,
    0.488, 0.493, 0.978, 0.991, 0.984, 0.990]`.

## Results (corrected stage-5 protocol)

Training-time val (unclipped inference, trend only): best 0.5379 at epoch 2,
final 0.5266 at epoch 14.

Corrected full-val on the 1543 val sources / 24,975 views:

| checkpoint | mAP50 | AP50 agnostic | recall | cls accuracy | pred |
| --- | ---: | ---: | ---: | ---: | ---: |
| fine-tune `best.pt` EMA | 0.6366 | 0.8951 | 0.9598 | 0.8385 | 72,403 |
| fine-tune `last.pt` EMA | 0.6366 | 0.9000 | 0.9573 | 0.8377 | 73,350 |
| **baseline `best_corrected.pt`** | **0.6427** | **0.9046** | **0.9580** | **0.8433** | **65,025** |

Per-class delta (fine-tune last minus baseline):

| class | baseline AP | fine-tune AP | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.354 | 0.341 | -0.013 |
| 1 | 0.329 | 0.320 | -0.009 |
| 2 | 0.523 | 0.533 | +0.010 |
| 3 | 0.492 | 0.445 | -0.047 |
| 4 | 0.427 | 0.390 | -0.037 |
| 5 | 0.405 | 0.459 | +0.053 |
| 6 | 0.959 | 0.958 | -0.000 |
| 7 | 0.947 | 0.944 | -0.004 |
| 8 | 0.941 | 0.936 | -0.005 |
| 9 | 0.948 | 0.945 | -0.004 |
| 10 | 0.708 | 0.684 | -0.024 |
| 11 | 0.571 | 0.581 | +0.010 |
| 12 | 0.739 | 0.728 | -0.011 |
| 13 | 0.654 | 0.649 | -0.006 |

## Verdict

**Not accepted.** The fine-tune improved class 5 (+0.053) and class 2/11 by
~0.01, but lost more on classes 3/4 (about -0.04), 10, 12, 13 and 1. Overall
class-aware mAP is 0.6366 vs the baseline **0.6427**; class-agnostic AP and
classification accuracy are also slightly lower. The baseline remains the
deliverable (`best_corrected.pt`).

Interpretation: a global class-weight/`cls`-weight change is too blunt for
the 0/1 and 3/4/5 confusion; it moves rare-class precision/recall at the cost
of neighbouring classes. Better follow-ups would be a larger model capacity,
mild time/frequency augmentation with careful label mapping, per-class score
calibration (evaluated on a separate split), or a targeted confusion-pair
loss. None of these was run in this baseline session.

Artifacts:
- `runs/yolo11s_c1_directreg_2048x512_clsft/config.json`
- `.../val_history.json` (training-time trend, unclipped inference)
- `.../stage5_val_best_ema/summary.json`, `.../stage5_val_last_ema/summary.json`
- `.../best.pt`, `.../last.pt`
