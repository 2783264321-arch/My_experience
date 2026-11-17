# Stage 4 - first 50-epoch run (completed)

Run ID: `yolo11s_c1_directreg_2048x512`
Directory: `runs/yolo11s_c1_directreg_2048x512/`
Launched 2026-11-18 03:42 CST; completed 06:31 CST; wall time **2.694 h**.
Training curve: `runs/yolo11s_c1_directreg_2048x512/training_curve.png`.

**Post-run correction (important).** Training-time validation did not clip
fine-view boxes to the view before remapping; out-of-band boxes depressed the
class-aware metric. Stage 5 added the required clamp and re-ran full
validation: `best.pt` (EMA epoch 29) = 0.6388, `last.pt` (EMA epoch 49) =
**0.6427**. The corrected best is `best_corrected.pt` (last EMA). The
training-time table below is retained as a trend, not as the final metric;
see `reports/stage5_results.md`.

## Locked configuration

| item | value |
| --- | --- |
| model | YOLO11s body, C=1, nc=14, `reg_max=1`, `end2end=False` |
| input | `[n,1,2048,512]` real views |
| pipeline | 0.7 ms periodic Hann, exactly 512 columns, scipy FFT backend |
| sampling | 1 coarse + 1 rotating fine view per fit source per epoch |
| physical batch | 8 sources = 16 views; accumulation 1 |
| effective batch | 16 views |
| optimizer | AdamW, lr0=1e-3, wd=5e-4 |
| schedule | cosine to 1%, 3-epoch warmup |
| precision | BF16 autocast (FP16 rejected in stage 2) |
| clip | 10.0 |
| seed | 1729 |
| workers | 12 train / 12 val |
| validation | every 5 epochs + final; all fine+coarse views of 1543 val sources |
| best selection | strict class-aware mAP@0.5 on EMA weights |
| split | fit 5957 / val 1543 / test 2500 (frozen manifest) |

## Cost

| item | value |
| --- | --- |
| epoch 0 | 393 s (startup + last-batch cuDNN autotune) |
| epochs 1+ | 155-164 s |
| full validation | 176-195 s for 24,975 views / 1543 sources |
| GPU reserved | 10.1 GiB |
| total | 2.694 h |

`cudnn.benchmark` was disabled after epoch 0 because the 10-view last batch
(`5957 % 8 = 5`) triggered a ~100 s autotune every epoch. The run resumed from
the epoch-0 `last.pt`; epoch 0 is preserved in the CSV/checkpoint. The
`train_log.csv` epoch-1 row count includes 89 discarded steps from the first
attempt before the restart; the resumed epoch 1 is the authoritative one.

## Validation history (EMA weights, all 24,975 views)

| epoch | mAP50 | AP50 agnostic | loc recall | cls acc | pred | val s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.3623 | 0.6789 | 0.843 | 0.746 | 101511 | 195 |
| 9 | 0.4465 | 0.7003 | 0.858 | 0.783 | 71602 | 180 |
| 14 | 0.5174 | 0.7395 | 0.878 | 0.812 | 55759 | 179 |
| 19 | 0.5299 | 0.7327 | 0.880 | 0.821 | 44433 | 184 |
| 24 | 0.5389 | 0.7388 | 0.882 | 0.825 | 40883 | 180 |
| **29** | **0.5429** | 0.7389 | 0.884 | 0.830 | 39520 | 178 |
| 34 | 0.5423 | 0.7415 | 0.883 | 0.830 | 38066 | 176 |
| 39 | 0.5399 | 0.7423 | 0.882 | 0.831 | 36522 | 183 |
| 44 | 0.5355 | 0.7410 | 0.879 | 0.831 | 35346 | 191 |
| 49 | 0.5319 | 0.7417 | 0.877 | 0.828 | 34288 | 186 |

Training-time best checkpoint: **epoch 29, mAP@0.5 0.5429**, saved as
`best.pt`. The class-agnostic AP peaks slightly later (0.7423 at epoch 39),
while the training-time class-aware AP peaks at 29. No early stopping was
used, per plan.

After the stage-5 view-clamp correction, the full-val metrics are:

| checkpoint | mAP50 | AP50 agnostic | recall |
| --- | ---: | ---: | ---: |
| `best.pt` (EMA epoch 29) | 0.6388 | 0.8937 | 0.9599 |
| `last.pt` (EMA epoch 49) | **0.6427** | 0.9046 | 0.9580 |

`best_corrected.pt` is the last EMA and is the corrected deliverable; the
training-time `best.pt` remains for reproducibility. The raw last model under
the old protocol was 0.5196, so EMA is clearly better.

## Interpretation

- Localization is substantially ahead of classification: agnostic AP ~0.74,
  localization recall ~0.88, but class-aware AP 0.543.
- Training loss fell from ~93 (epoch 0, includes the first huge step) to ~16
  (epoch 49); the class-aware validation peaked before the low-LR tail.
- Classes 6/7/8 are strong (AP 0.85-0.90); class 13 (0.246), class 1 (0.324),
  class 5 (0.354), class 4 (0.384) and class 0 (0.384) are weak.
- A post-baseline classification-focused fine-tune
  (`runs/yolo11s_c1_directreg_2048x512_clsft/`) was launched separately; the
  frozen first baseline remains this run.
