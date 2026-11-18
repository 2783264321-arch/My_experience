# Baseline execution status

Baseline: `yolo11s_c1_directreg_2048x512`
Root: `/root/autodl-tmp/workspace/baseline`

Input pipeline: IQ -> 0.7 ms periodic Hann STFT (512 columns) -> log power
-> one 2048x512 coarse full-band view + 2048x512 fine slices (step 1639).
Model: YOLO11s body, C=1, nc=14, `reg_max=1`, `end2end=False` (DirectReg).

## Stage 0 - audit: PASSED

`reports/stage0_audit.json`

- Frozen split reused from `split_manifest.csv`:
  - fit 5957 samples / 27593 instances
  - val 1543 / 7162
  - test 2500 / 20018
- All 10000 files have size `4*N`; all JSON signal counts match; all labels are
  inside band/time and have positive area; `N = B*T*1e6` for all samples.
- 30 stratified samples were actually read back: 30/30 match manifest `N`.
- Normalizer: lo `-103.23661804199219`, hi `-45.258453369140625`
  (`configs/normalizer.json`).
- Hashes are recorded in the audit JSON.
- Earlier local 6749/751 fold audit is obsolete; not used.

## Stage 1 - data smoke: PASSED

`reports/stage1_smoke.md`, `reports/smoke_data.json`

Six B/T extremes: exact `[F,512]` STFT, `[1,2048,512]` views, empty fine
slices preserved, coordinate round-trip error ~1e-15 ms / 0 MHz.

## Stage 2 - loss / AMP / resources

`reports/stage2_resources.json`, `reports/stage2_grad_probe.json`

- FP32 forward/loss/backward on real view shapes: finite gradients.
- FP16 autocast: **rejected** - on this 21504-anchor head it produced millions
  of non-finite gradients and `GradScaler` skipped every step.
- BF16 autocast: finite gradients; losses decrease; accepted as the AMP mode.
- Real optimizer-step measurements (batch = sources, 2 views per source;
  AdamW + EMA + BF16):

| sources | views | step s | peak reserved GiB |
| ---: | ---: | ---: | ---: |
| 4 | 8 | 0.067 | 5.3 |
| 8 | 16 | 0.126 | 10.2 |
| 12 | 24 | 0.186 | 15.1 |

Chosen: 8 sources = 16 views per optimizer step, BF16, reserved ~10 GiB
(< 20 GiB gate).  AdamW lr0=1e-3, wd=5e-4, cosine to 1%, 3-epoch warmup.

Data pipeline (on-the-fly scipy FFT, 12 workers x 2 FFT threads):
about 17 sources/s => ~5.8 min/epoch for 5957 fit sources.
Validation uses all fine+coarse views; measured cadence decision is recorded
in the run config (default every 5 epochs if the measured cost stays under
1/3 of total).

## Stage 3 - 16-source overfit: PASSED

`reports/stage3_overfit.md`, `runs/overfit16b/`

16 fit sources, all 14 classes, B in {5,20,30,40,50,80} MHz and T in
{20,...,150} ms. Same two-view sampling; full-sample validation on all
fine+coarse views. Strict class-aware mAP@0.5 reached **0.8168 at update 600**
(target 0.8); after the stage-5 view-clamp fix the same checkpoint validates at
**0.8403**. Losses fell from 66.1 to 6.5.

## Stage 4 - 50 epoch training: COMPLETED

Completed 2026-11-18 06:31 CST in 2.694 h. Training-time best was epoch 29
(mAP@0.5 0.5429, EMA), but the stage-5 view-clamp correction changed the
metric. Corrected full val: `best.pt` EMA 0.6388, `last.pt` EMA **0.6427**;
`best_corrected.pt` is the corrected deliverable. Training-time table and
full explanation: `reports/stage4_training.md`.

## Stage 4 - launch details

Launched 2026-11-18 03:42 in `screen -S train_main`; log `/tmp/train_main.log`.
Run directory `runs/yolo11s_c1_directreg_2048x512/`.

Measured epoch 0: 393 s (first-step startup + a cuDNN autotune of the 10-view
last batch). `cudnn.benchmark` was disabled and the run resumed at epoch 1;
subsequent epochs are ~158-164 s. Epoch 0 is preserved in `last.pt`/CSV.

Full validation: 179-195 s for 24,975 views / 1543 sources (~20% wall-time
overhead at the 5-epoch cadence, kept fixed).

| epoch | mAP50 class-aware | AP50 agnostic | loc recall |
| ---: | ---: | ---: | ---: |
| 4 | 0.3623 | 0.6789 | 0.843 |
| 9 | 0.4465 | 0.7003 | 0.858 |
| 14 | 0.5174 | 0.7395 | 0.878 |

Localization is strong; class-aware AP is still climbing. Class 13 and the
dominant class 9 are the main confusion sources so far.

GitHub push: **completed**. `main` is pushed to
`https://github.com/2783264321-arch/My_experience` via a repo-scoped
write-enabled deploy key (`~/.ssh/id_ed25519_my_experience`); remote head
tracks local `main`.

Command:

```bash
cd /root/autodl-tmp/workspace/baseline
SPACENET_FFT_THREADS=2 PYTHONPATH=. .venv/bin/python -u -m src.signals.train \
  --run yolo11s_c1_directreg_2048x512 --epochs 50 \
  --batch-sources 8 --accum 1 --workers 12 --val-workers 12 \
  --val-batch-views 8 --val-every 5 --precision bf16 --seed 1729
```

Outputs: `runs/yolo11s_c1_directreg_2048x512/{config.json, train_log.csv,
val_history.json, best.pt, last.pt, summary.json}`.

## Stage 5 - validation / delivery: COMPLETED (baseline)

Corrected full val on the 1543 sources / 24,975 views:
mAP@0.5 **0.6427**, agnostic AP 0.9046, recall 0.9580, classification
accuracy 0.8433, 65,025 predictions, 202,142 internal-boundary candidates
withheld. Primary checkpoint `best_corrected.pt` (last EMA).
Artifacts under `runs/yolo11s_c1_directreg_2048x512/stage5_val/` and
`stage5_examples/`; report `reports/stage5_results.md`.
The 15-epoch classification-weighted fine-tune completed separately
(`runs/yolo11s_c1_directreg_2048x512_clsft/`) and did **not** beat the
baseline: corrected mAP 0.6366 vs 0.6427. It is reported as a negative
follow-up; the baseline `best_corrected.pt` remains the deliverable.

## Stage 5 - commands

```bash
PYTHONPATH=. .venv/bin/python scripts/validate_run.py \
  --run yolo11s_c1_directreg_2048x512 --checkpoint best_corrected.pt --workers 12
PYTHONPATH=. .venv/bin/python scripts/predict_examples.py \
  --run yolo11s_c1_directreg_2048x512 --checkpoint best_corrected.pt --limit 10
```

Test split stays sealed.
