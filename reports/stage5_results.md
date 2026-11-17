# Stage 5 - corrected full validation and error analysis

Run: `yolo11s_c1_directreg_2048x512`.
Primary checkpoint: **`best_corrected.pt`** = EMA of `last.pt` (epoch 49),
selected by the corrected full-sample strict mAP@0.5 (0.6427) after the
inference view-clamp fix. `best.pt` (EMA epoch 29) scores 0.6388 under the same
corrected protocol, so the two are within 0.004; the corrected selection is
documented in `selection_corrected.json`.
Split: frozen val 1543 sources, 24,975 views (all fine + coarse), 7162 GT
instances. Test remains sealed.

## Important correction

Training-time validation (`val_history.json`) used `_map_view_dets` without
clipping fine-view boxes to the view. DirectReg boxes extending past a view
mapped to frequencies outside the original band and survived at the first/last
slices, inflating false positives and depressing class-aware AP (training-time
best 0.5429). Stage 5 found this, added the required clamp to
`[0,512] x [0,2048]`, re-ran full validation, and re-selected the checkpoint:

| checkpoint | weights | corrected mAP50 | AP50 agnostic | recall |
| --- | --- | ---: | ---: | ---: |
| `best.pt` | EMA epoch 29 | 0.6388 | 0.8937 | 0.9599 |
| `last.pt` | EMA epoch 49 | **0.6427** | 0.9046 | 0.9580 |
| `last.pt` | raw epoch 49 (old protocol) | 0.5196 | 0.7261 | 0.8688 |

The corrected numbers below supersede the training-time val table.

## Headline

| metric | value |
| --- | ---: |
| strict class-aware mAP@0.5 | **0.6427** |
| class-agnostic AP@0.5 | 0.9046 |
| localization recall @0.5 | 0.9580 |
| classification accuracy among matched | 0.8433 |
| GT / predictions | 7162 / 65025 |
| boundary candidates withheld (fine views) | 202,142 |
| validation wall time | 258.6 s |
| peak GPU reserved | 1.32 GiB |

## Per class

| class | AP50 | GT | pred |
| ---: | ---: | ---: | ---: |
| 0 | 0.354 | 337 | 4945 |
| 1 | 0.329 | 347 | 5717 |
| 2 | 0.523 | 313 | 6112 |
| 3 | 0.492 | 155 | 2781 |
| 4 | 0.427 | 148 | 1581 |
| 5 | 0.405 | 139 | 1761 |
| 6 | 0.959 | 712 | 2397 |
| 7 | 0.947 | 704 | 3648 |
| 8 | 0.941 | 1431 | 5513 |
| 9 | 0.948 | 1411 | 10310 |
| 10 | 0.708 | 376 | 4258 |
| 11 | 0.571 | 340 | 3309 |
| 12 | 0.739 | 399 | 4407 |
| 13 | 0.654 | 350 | 8286 |

## Per bandwidth (all durations)

| bandwidth | samples | GT | mAP50 | AP50 agnostic |
| --- | ---: | ---: | ---: | ---: |
| 5 MHz | 261 | 459 | 0.8243 | 0.8899 |
| 20 MHz | 314 | 1446 | 0.7377 | 0.8980 |
| 30 MHz | 285 | 1390 | 0.7067 | 0.9046 |
| 40 MHz | 231 | 1304 | 0.6351 | 0.9028 |
| 50 MHz | 243 | 1305 | 0.6298 | 0.9166 |
| 80 MHz | 209 | 1258 | 0.6522 | 0.9162 |

All 31 B/T groups are in `per_bt.csv`; e.g. 20 MHz/20 ms (n=149, GT=456) is
0.772, 80 MHz/100 ms (n=37, GT=220) 0.767, 5 MHz/20 ms 0.824. Small groups
have wide error bars.

## Error analysis

Matching at IoU>=0.5: 6,490 TP, 58,535 FP, 672 FN across 65,025 predictions
(the model emits many low-score boxes; AP is threshold-free). Main patterns:

1. **Residual class confusion is concentrated in 0/1 and 3/4/5.** Confusion
   matrix rows (GT -> predicted class): class 0 = 161 correct, 114 as 1;
   class 1 = 123 correct, 160 as 0; class 4 = 52 correct, 67 as 3, 22 as 5;
   class 5 = 58 correct, 48 as 3, 27 as 4. These are the weakest AP classes
   (0.33-0.49). Class 9/13 is no longer a major confusion after the clamp
   (class 9: 1,368/1,384 correct; class 13: 252/261).
2. **Over-prediction remains for some classes.** Class 9: 1,380 TP but 8,930
   FP; class 13: 256 TP / 8,030 FP; class 2: 237 TP / 5,875 FP; class 1:
   278 TP / 5,439 FP. The final per-sample cap is 100 boxes; the extra
   low-score candidates still affect the class precision-recall ranking.
3. **Duplicate/near-identical GT boxes are retained (by design).** Top missed
   GT entries have a same-class prediction at IoU 0.995-0.997 with high score
   (e.g. train/3618, train/1695, train/7228, train/6491), but one prediction
   can match only one of the near-identical GT instances. This is a direct
   consequence of "no silent dedup / no same-box ignore"; it caps recall/AP on
   those samples and is reported, not filtered.
4. **Class-wise global NMS keeps cross-class same-location boxes.** 3,800
   prediction groups at IoU>=0.9 contain more than one class (0 same-class
   duplicate groups). This is the frozen inference rule; class-agnostic NMS
   or per-class calibration would be a post-baseline experiment.
5. **Boundary candidates.** 202,142 fine-view boxes within 16 bins of an
   internal slice boundary were withheld as complete events. Adjacent
   overlapping slices and the coarse view still provide complete detections.

## Speed / memory

Single 4090, batch 8 views, full sample (median of 3, corrected protocol):

| sample | B/T | views | total s | STFT+views s | final dets |
| --- | --- | ---: | ---: | ---: | ---: |
| train/24 | 5 MHz / 20 ms | 3 | 0.097 | 0.048 | 10 |
| train/231 | 40 MHz / 80 ms | 18 | 0.786 | 0.495 | 45 |
| train/19 | 80 MHz / 150 ms | 35 | 1.665 | 1.209 | 27 |

The long 80 MHz/150 ms sample is CPU-STFT dominated. Training reserved
10.1 GiB; full validation reserved 1.32 GiB. (The inference run overlapped a
fine-tune job, so timings are slightly conservative.)

## Artifacts

- `runs/yolo11s_c1_directreg_2048x512/stage5_val_corrected_last_ema/summary.json`
- `.../val_predictions.npz` (65,025 predictions + GT, per-sample indices)
- `.../per_bt.csv` (31 B/T groups)
- `.../error_analysis.json`
- `runs/yolo11s_c1_directreg_2048x512/stage5_examples/` (10 corrected overlays)
- `runs/yolo11s_c1_directreg_2048x512/stage5_inference.json`
- `runs/yolo11s_c1_directreg_2048x512/best_corrected.pt`
- `runs/yolo11s_c1_directreg_2048x512/selection_corrected.json`

## Caveats

- This is the **first baseline**, not a tuned detector. It uses DirectReg
  (`reg_max=1, end2end=False`), not stock YOLO11s.
- Localization is strong (agnostic AP 0.905, recall 0.958); class confusion
  among 0/1 and 3/4/5 and over-prediction are the remaining limits.
- Test was not evaluated; no test-driven selection.
- The training-time val_history is a trend only; the corrected stage-5 table
  is the authoritative val result. A post-baseline classification fine-tune
  is running separately and will be reported on its own.
