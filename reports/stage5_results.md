# Stage 5 - full validation and error analysis (first baseline)

Run: `yolo11s_c1_directreg_2048x512`, checkpoint `best.pt` (EMA, epoch 29).
Split: frozen val 1543 sources, 24,975 views (all fine + coarse), 7162 GT
instances. Test remains sealed.

Artifacts:
- `runs/yolo11s_c1_directreg_2048x512/stage5_val/summary.json`
- `runs/yolo11s_c1_directreg_2048x512/stage5_val/val_predictions.npz`
  (per-sample predictions + GT, 39,529 predictions)
- `runs/yolo11s_c1_directreg_2048x512/stage5_val/per_bt.csv` (31 B/T groups)
- `runs/yolo11s_c1_directreg_2048x512/stage5_val/error_analysis.json`
- `runs/yolo11s_c1_directreg_2048x512/stage5_examples/` (10 overlays)
- `runs/yolo11s_c1_directreg_2048x512/stage5_inference.json`
- `runs/yolo11s_c1_directreg_2048x512/training_curve.png`

## Headline

| metric | value |
| --- | ---: |
| strict class-aware mAP@0.5 | **0.5428** |
| class-agnostic AP@0.5 | 0.7389 |
| localization recall @0.5 | 0.8834 |
| classification accuracy among matched | 0.8304 |
| GT / predictions | 7162 / 39529 |
| boundary candidates dropped (fine views) | 225,267 |
| validation wall time | 178.6 s |
| peak GPU reserved | 1.32 GiB |

## Per class

| class | AP50 | GT | pred |
| ---: | ---: | ---: | ---: |
| 0 | 0.384 | 337 | 2405 |
| 1 | 0.324 | 347 | 2460 |
| 2 | 0.476 | 313 | 2043 |
| 3 | 0.493 | 155 | 974 |
| 4 | 0.384 | 148 | 691 |
| 5 | 0.354 | 139 | 814 |
| 6 | 0.850 | 712 | 1859 |
| 7 | 0.902 | 704 | 2207 |
| 8 | 0.886 | 1431 | 3752 |
| 9 | 0.520 | 1411 | 12819 |
| 10 | 0.683 | 376 | 2119 |
| 11 | 0.629 | 340 | 1716 |
| 12 | 0.469 | 399 | 2691 |
| 13 | 0.246 | 350 | 2970 |

## Per bandwidth (all durations)

| bandwidth | samples | GT | mAP50 | AP50 agnostic |
| --- | ---: | ---: | ---: | ---: |
| 5 MHz | 261 | 459 | 0.5829 | 0.6317 |
| 20 MHz | 314 | 1446 | 0.6383 | 0.7719 |
| 30 MHz | 285 | 1390 | 0.6319 | 0.7889 |
| 40 MHz | 231 | 1304 | 0.5696 | 0.7750 |
| 50 MHz | 243 | 1305 | 0.5349 | 0.7876 |
| 80 MHz | 209 | 1258 | 0.5229 | 0.7460 |

The 31 individual B/T groups are in `per_bt.csv`; e.g. 20 MHz/20 ms
(n=149, GT=456) reached 0.6738, 30 MHz/150 ms (n=31, GT=186) 0.6924, while
80 MHz/60 ms (n=35, GT=208) was 0.5017. Small groups have wide error bars.

## Error analysis

Total matched at IoU>=0.5: 5,942 TP, 33,587 FP, 1,220 FN.

Main patterns:

1. **Class confusion on real signals.** Many top false positives have
   `best_gt_iou = 0.0` for their predicted class while the class-agnostic AP
   is 0.739. The same object is being classified as the wrong class, not
   localized wrongly. The confusion matrix shows, for GT classes:
   - class 0: 200 correct, 100 predicted as 1;
   - class 1: 114 correct, 179 predicted as 0;
   - class 4: 52 correct, 70 predicted as 3, 21 as 5;
   - class 5: 45 correct, 54 predicted as 3, 31 as 4;
   - class 13: 118 correct, 43 predicted as 9.
   Classes 0/1, 3/4/5 and 9/13 are the main confusable groups.

2. **Over-prediction of a few dominant classes.** Class 9 produced 12,819
   predictions for 1,411 GT (11,779 FP); class 13 2,970 for 350 (2,837 FP);
   class 12 2,691 for 399 (2,457 FP); class 8 3,752 for 1,431 (2,379 FP).
   Class 9 has high recall (1,045 TP) but its precision ranking is the largest
   single source of FPs.

3. **Duplicate/near-identical GT boxes are retained.** Several top missed GT
   entries have a same-class prediction at IoU 0.99 (e.g. train/2043,
   train/6491, train/2122, train/5700) but are unmatched because one prediction
   can match only one of the duplicate/near-duplicate GT instances. This is a
   direct consequence of the required "no silent dedup / no same-box ignore"
   policy. It caps recall/AP on those samples and is reported rather than
   filtered.

4. **Same-location multi-class outputs.** The mandated class-wise global NMS
   keeps different-class boxes at the same location: 2,397 prediction groups
   at IoU>=0.9 have more than one class (0 same-class duplicate groups).
   Class-agnostic NMS would remove many of these but would change the frozen
   inference rule, so it was not used for the baseline. A per-class score
   calibration would be a post-baseline option.

5. **Internal fine-slice boundary rule.** 225,267 fine-view boxes were within
   16 native bins of an internal slice boundary and were withheld as boundary
   candidates (not complete events), per the plan. The coarse view and
   overlapping adjacent fine slices still provide complete detections; the
   final per-sample output is capped at 100 boxes.

## Speed / memory

Full-sample inference on a single 4090, batch 8 views (median of 3, after warmup):

| sample | B/T | views | total s | STFT+views s | final dets |
| --- | --- | ---: | ---: | ---: | ---: |
| train/24 | 5 MHz / 20 ms | 3 | 0.060 | 0.044 | 12 |
| train/231 | 40 MHz / 80 ms | 18 | 0.564 | 0.461 | 21 |
| train/19 | 80 MHz / 150 ms | 35 | 1.277 | 1.106 | 11 |

The long 80 MHz/150 ms sample is dominated by the CPU STFT (1.1 s); the
model + NMS + remap add ~0.17 s. Training reserved 10.1 GiB; full validation
reserved 1.32 GiB.

## Caveats

- This is the **first baseline**, not a tuned detector. It uses DirectReg
  (`reg_max=1, end2end=False`), not stock YOLO11s; reports must say so.
- The class-aware AP is limited mainly by classification confusion and
  over-prediction, not by localization (agnostic AP 0.739, recall 0.883).
- Test was not evaluated; no test-driven selection was performed.
- The val set was used for checkpoint selection and for this error analysis;
  the reported numbers are val numbers, not test numbers.
