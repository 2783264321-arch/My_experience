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

## Diagnostics (2026-11-18)

- Three-level val: R_agn **0.958** -> AP_agn **0.905** -> AP_cls **0.643**.
  Pre-boundary/post-boundary/post-NMS/final recall: 0.9645/0.9606/0.9583/0.9578;
  boundary+NMS+100-box cap cost <0.7 pp. The first ceiling is nearly saturated.
- Oracle-class AP **0.8639** (gap 22.1 pp): classification loss is concentrated
  in classes 0/1/2, 3/4/5, 10/11; 6/7/8/9/12/13 are near their oracle ceiling.
- AP75 **0.5945**, AP50:95 **0.5764** (agnostic 0.8543/0.8203). AP50 stays the
  primary/checkpoint metric; AP75/AP50:95 are secondary localization diagnostics.
- Full tables: `reports/stage5_diagnostics.md`, `stage5_val/signal_characteristics.json`,
  `stage5_val/oracle_class_ap.json`, `stage5_val/stage_recall.json`,
  `stage5_val/iou_sweep.json`.
- 8-epoch low-LR continuation completed:
  `runs/yolo11s_c1_directreg_2048x512_conv8/`; best **0.6423** (epoch 3) vs
  baseline **0.6427**; final 0.6405. No gain; convergence confirmed.

## Stage 5 - commands

```bash
PYTHONPATH=. .venv/bin/python scripts/validate_run.py \
  --run yolo11s_c1_directreg_2048x512 --checkpoint best_corrected.pt --workers 12
PYTHONPATH=. .venv/bin/python scripts/predict_examples.py \
  --run yolo11s_c1_directreg_2048x512 --checkpoint best_corrected.pt --limit 10
```

Test split stays sealed.

## Group A - full-band single-image baseline: COMPLETED

Worktree `/root/autodl-tmp/worktrees/baseline-0-group-a`, run
`yolo11s_c1_directreg_coarseonly_2048x512`.

- 100 epochs, one coarse view per source; equal budget with C: 595,700 image
  presentations, 37,300 vs 37,250 optimizer steps, effective batch 16.
- Best EMA (ep29) **0.5763**, last EMA (ep99) **0.5403**; clear overfitting
  after ep29 (train loss kept falling, val mAP/AP_agn/recall declined).
- B (same joint weights, coarse only) **0.5794**; C (all views) **0.6427**.
  B−A best +0.31 pp; C−A best +6.64 pp; C−B +6.33 pp.
- A best is strong on 10/11 (0.6399) and medium 0.3–2 MHz (0.8917); weak on
  narrow 9/12/13 (0.5002 vs C 0.7804) and the 0–5 wide classes.
- Inference: 1 view vs C 19.10; model-only 8.64x, end-to-end p50 1.56x.
- Future runs enable 4-event early stopping (min_delta 0.001, min_evals 3)
  and select best EMA. Artifacts:
  `reports/group_a_results.md`, `reports/group_a_training.md`,
  `docs/collaboration/opencode-group-a-fullband-baseline.md`.

## C3 coarse front-end pilot: COMPLETED (subset screening)

1536 fit / 384 val, 20,000 steps, best EMA; C1/overlap/sparse 2 seeds, dense 1.
C1 mean 0.4999; overlap mean 0.5078 (+0.79 pp); sparse mean 0.4998 (-0.01 pp);
dense 0.5072 (+0.73 pp). No stable 0/1/2 gain: overlap 0.3019->0.2473, sparse
0.2819->0.2336. 3/4/5 is the only replicated gain: overlap 0.4012/0.3940
(mean +4.01 pp); dense 0.4646 (+10.70 pp, wrong 79->69, single seed); sparse
unstable. Dense is a 3/4/5 specialist, not a global replacement. Capacity is
still confounded (C1 1-channel vs C3 3-channel); a C3-zero-complex control is
needed. Details: `docs/collaboration/opencode-c3-frontends.md`,
`reports/c3_pilot_results.md`.

## C3 replication/control: COMPLETED (screening)

dense seed1730 and zero-complex (power+0+0) seeds1729/1730, same 1536/384 subset,
20k steps. Best-mean: zero3 0.5156 > dense 0.5123 > overlap 0.5078 > sparse 0.4998.
No complex front-end beats the no-information 3ch control overall. 0/1/2:
zero3 0.2834 best; dense 0.2619. 3/4/5: dense 0.4543 vs zero3 0.4198
(+3.45 pp, both seeds positive); common-box 3/4/5 classification +2.95..+5.95 pp
(4/4 direction) but McNemar p>0.2 and recall not higher. Not enough for full
training; next: 3-5 seeds, 768+ val, live uninformative capacity control
(power+power+power/random), then full only if 3/4/5 residual survives.
Dense estimator: aggregate-first raises both GT and noise; keep current
per-pair normalized operator. Details: `reports/c3_replication_results.md`,
`docs/collaboration/opencode-c3-frontends.md`.

## C3 1,159 holdout replication: COMPLETED

Frozen dense/zero3 best+last on full val minus 384 subset (1,159 samples, 290
3/4/5 GT); no re-selection/threshold tuning. Best-mean: dense mAP 0.4983 vs
zero3 0.4990; 3/4/5 0.3874 vs 0.3489; common 3/4/5 acc 0.4804 vs 0.4454.
Last-mean: 0.4583 vs 0.4520; 3/4/5 0.3163 vs 0.3465; common acc 0.4697 vs
0.5032. Only s1730 best significantly favors dense (+10.45 pp, p=0.007);
s1729 best/last and s1730 last favor zero3. The 384-subset 4/4 dense direction
did not replicate; localization recall not higher. Do not expand full training.
Next, only a pre-registered checkpoint-rule multi-seed study + live capacity
control is defensible. Details: `reports/c3_holdout_replication.md`,
`docs/collaboration/opencode-c3-holdout.md`.
