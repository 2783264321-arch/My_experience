# Stage 4 - first 50-epoch run

Run ID: `yolo11s_c1_directreg_2048x512`
Directory: `runs/yolo11s_c1_directreg_2048x512/`
Launched: 2026-11-18 03:42 CST, `screen -S train_main` (log `/tmp/train_main.log`).

## Locked configuration

| item | value |
| --- | --- |
| model | YOLO11s body, C=1, nc=14, `reg_max=1`, `end2end=False` |
| input | `[n,1,2048,512]` real views |
| pipeline | 0.7 ms Hann, 512 columns, scipy FFT backend |
| sampling | 1 coarse + 1 rotating fine view per fit source per epoch |
| physical batch | 8 sources = 16 views; accumulation 1 |
| effective batch | 16 views |
| optimizer | AdamW, lr0=1e-3, wd=5e-4 |
| schedule | cosine to 1%, 3-epoch warmup |
| precision | BF16 autocast (FP16 rejected in stage 2) |
| clip | 10.0 |
| seed | 1729 |
| workers | 12 train, 12 val |
| validation | all fine+coarse views of 1543 val sources, every 5 epochs + last |
| best selection | strict class-aware mAP@0.5 on EMA weights |
| split | fit 5957 / val 1543 / test 2500 (frozen manifest) |

## Measured cost

| item | value |
| --- | --- |
| epoch 0 | 393 s (startup + last-batch cuDNN autotune) |
| epochs 1+ | 155-164 s |
| full validation | 195 s (24,975 views, 1543 sources) |
| GPU reserved | 10.1 GiB |
| val overhead | ~20% of wall time at 5-epoch cadence |

`cudnn.benchmark` was disabled after epoch 0 because the 10-view last batch
(`5957 % 8 = 5`) triggered a ~100 s autotune every epoch. The run resumed from
the epoch-0 `last.pt`; epoch 0 remains in the CSV/checkpoint.

## Validation history

Filled from `val_history.json` at the end of the run.

| epoch | mAP50 class-aware | AP50 class-agnostic | localization recall | val time |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.3623 | 0.6789 | 0.843 | 195 s |
