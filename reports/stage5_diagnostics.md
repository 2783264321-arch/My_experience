# Stage-5 diagnostics: three-level loss decomposition

This report answers the question "where does the performance go?" with three
levels:

1. **R_agn** - class-agnostic localization recall (IoU=0.5, final 100 boxes).
2. **AP_agn** - class-agnostic AP@0.5 (adds score ranking/precision).
3. **AP_cls** - strict class-aware mAP@0.5 (the actual metric).
4. **Oracle-class AP** - diagnostic: relabel each class-agnostic-matched
   prediction with its GT class, then recompute class-aware AP. This is the
   "perfect classification of matched boxes" ceiling under the current
   detection/ranking.

Also included: stage-wise recall (pre-boundary / post-boundary / post-NMS /
caps), and AP50/AP75/AP50:95.

Artifacts:
- `stage5_val/signal_characteristics.json` - R_agn/R_cls/AP_agn/AP_cls per group
- `stage5_val/oracle_class_ap.json` - actual vs oracle class AP
- `stage5_val/stage_recall.json` - pre/post/NMS/cap recalls
- `stage5_val/iou_sweep.json` - AP50/AP75/AP50:95

---

## 1. Three-level table by signal group

`R_agn` = class-agnostic recall; `R_cls` = same-class matched recall;
`AP_agn` = class-agnostic AP; `AP_cls` = class-aware AP; `oracle` = oracle-class
AP; `cls gap` = oracle - actual; `rank gap` = R_agn - AP_agn.

| group | n_gt | R_agn | R_cls | AP_agn | AP_cls | oracle | cls gap | rank gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 7162 | 0.958 | 0.906 | 0.905 | **0.643** | 0.864 | 0.221 | 0.053 |
| narrow ≤0.25 MHz | 2160 | 0.916 | 0.910 | 0.798 | 0.780 | 0.786 | **0.006** | 0.118 |
| medium 0.3–2 MHz | 3041 | 0.981 | 0.964 | 0.940 | 0.839 | 0.920 | 0.081 | 0.041 |
| wide 2.1–10 MHz | 522 | 0.967 | 0.862 | 0.673 | 0.470 | 0.846 | **0.375** | 0.294 |
| very wide 10–20 MHz | 997 | 0.970 | 0.807 | 0.578 | 0.402 | 0.827 | **0.425** | 0.392 |
| ultra >20 MHz | 442 | 0.966 | 0.762 | 0.531 | 0.441 | 0.865 | **0.424** | 0.435 |
| relative >20% | 1978 | 0.969 | 0.819 | 0.761 | 0.497 | 0.841 | 0.344 | 0.208 |
| short ≤5 ms | 514 | 0.881 | 0.811 | 0.454 | 0.242 | 0.396 | 0.153 | 0.427 |

Interpretation:

- **Recall is not the main bottleneck** for wide/ultra: R_agn is ~0.97, while
  AP_agn is only 0.53–0.67. The model finds the boxes; the loss is
  precision/score ranking (too many FPs / duplicate boxes / low-ranked TPs).
- **Classification is the dominant loss for wide/ultra**: actual AP 0.40–0.47
  vs oracle 0.83–0.87. Classes 0/1/2, 3/4/5 and 10/11 gain 0.19–0.49 AP when
  classification is oracle.
- **Narrow ≤0.25 MHz**: R_agn 0.916, AP_agn 0.798, AP_cls 0.780, oracle 0.786.
  Classification gap is only 0.006; the loss is tiny-box localization/ranking
  and duplicate handling. Classes 9 (0.948), 12 (0.739), 13 (0.654).
- **Medium 1–2 MHz**: R_agn 0.981, AP_agn 0.940, AP_cls 0.839, oracle 0.920.
  Localization is near ceiling; an 8-point classification gap remains.
- **Short ≤5 ms**: R_agn 0.881 but AP_agn only 0.454, oracle 0.396. The first
  gap (0.427) and the oracle ceiling both show a time-representation/ranking
  problem; classification is secondary.

---

## 2. Stage-wise recall: boundary/NMS/max-100 are not the bottleneck

Overall, 7162 GT:

| stage | matched | recall |
| --- | ---: | ---: |
| pre-boundary (per-view NMS) | 6908 | 0.9645 |
| post-boundary | 6880 | 0.9606 |
| post global NMS | 6863 | 0.9583 |
| final @100 | 6860 | **0.9578** |
| final @10 | 6648 | 0.9282 |

Losses:
- boundary rule: 0.9645 -> 0.9606 = **0.39 pp**
- global class-wise NMS: 0.9606 -> 0.9583 = **0.23 pp**
- 100-box cap: 0.9583 -> 0.9578 = **0.05 pp**
- final recall vs evaluator `localization_recall` 0.9580: matches within rounding.

Per group (final@100 recall):
- narrow 0.916, medium 0.981, wide 0.967, very wide 0.970, ultra 0.966,
  short 0.879, relative>20% 0.968.

So the **first level is nearly saturated**; boundary/NMS/max-det tuning cannot
recover the 0.96 -> 0.53 AP_agn gap. The real loss is ranking/precision and
classification.

---

## 3. Oracle-class AP: where the classification loss is

Overall: actual 0.6427 -> oracle **0.8639**, gap **22.1 pp**.

Per class (actual -> oracle, gap):

| class | actual AP50 | oracle AP50 | gap | median BW |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.354 | 0.818 | 0.464 | 20 MHz |
| 1 | 0.329 | 0.814 | 0.485 | 20 MHz |
| 2 | 0.523 | 0.850 | 0.327 | 20 MHz |
| 3 | 0.492 | 0.879 | 0.387 | 40 MHz |
| 4 | 0.427 | 0.863 | 0.436 | 40 MHz |
| 5 | 0.405 | 0.853 | 0.448 | 40 MHz |
| 6 | 0.959 | 0.964 | 0.006 | 1 MHz |
| 7 | 0.947 | 0.959 | 0.011 | 2 MHz |
| 8 | 0.941 | 0.953 | 0.012 | 2 MHz |
| 9 | 0.948 | 0.953 | 0.004 | 0.25 MHz |
| 10 | 0.708 | 0.901 | 0.193 | 10 MHz |
| 11 | 0.571 | 0.881 | 0.310 | 10 MHz |
| 12 | 0.739 | 0.742 | 0.003 | 0.2 MHz |
| 13 | 0.654 | 0.664 | 0.009 | 0.12 MHz |

Per group:
- wide 0.470 -> 0.846; very wide 0.402 -> 0.827; ultra 0.441 -> 0.865;
  relative>20% 0.497 -> 0.841.
- narrow 0.780 -> 0.786; medium 0.839 -> 0.920; short 0.242 -> 0.396.

Conclusion: **classes 6/7/8/9/12/13 are already at their oracle ceiling** under
the current detection/ranking; the big classification losses are
0/1/2, 3/4/5, 10/11. The previous global class-weighted fine-tune was too
blunt and failed; targeted classification/verification is the lever.

Note: oracle class-aware AP can numerically exceed class-agnostic AP for a
group (e.g. ultra: oracle 0.865 vs AP_agn 0.531) because the oracle relabeling
makes per-class APs high while class-agnostic AP is dragged by cross-class FPs
and macro averaging. The hierarchy R_agn -> AP_agn -> AP_cls is a diagnostic
ladder, not a strict numeric chain for every group.

---

## 4. AP50:95 / AP75

| metric | class-aware | class-agnostic |
| --- | ---: | ---: |
| AP50 | 0.6427 | 0.9046 |
| AP75 | 0.5945 | 0.8543 |
| AP50:95 | **0.5764** | 0.8203 |

The drop AP50 -> AP75 is 4.8 pp, AP50 -> AP50:95 is 6.6 pp. Boxes are
reasonably tight; the high-IoU metric does not change the weak/strong class
ranking. Per-class AP50:95: 0=0.310, 1=0.288, 2=0.463, 3=0.414, 4=0.382,
5=0.365, 6=0.918, 7=0.888, 8=0.883, 9=0.830, 10=0.653, 11=0.509, 12=0.616,
13=0.550.

Recommendation:
- Keep **AP50 as the primary/checkpoint-selection metric** (matches the frozen
  plan; time-frequency box tolerance; macro classes).
- Add **AP75 and AP50:95 as secondary localization diagnostics**, per class and
  per bandwidth/duration group. They are cheap (10 IoU thresholds, ~1 min).
- For tiny 12/13 boxes, high-IoU AP is scale-sensitive; report with group
  context, do not compare across extreme box sizes without a caveat.
- Do not use AP50:95 to replace AP50 unless the plan is explicitly revised.

---

## 5. What this implies for the next experiment

Priority 1: **precision/ranking + classification for the wide families**
{0,1,2}, {3,4,5}, {10,11}. The boxes are found (R_agn ~0.97); the AP is lost to
FPs/ranking and wrong class. A targeted classifier/verifier on proposals, a
larger model, hard-negative mining, or a pairwise confusion loss is more
promising than more of the same training.

Priority 2: **short signals ≤5 ms** - R_agn 0.881 but AP_agn 0.454; time
representation/ranking. Multi-resolution time / shorter windows / time crops.

Priority 3: **tiny 12/13** - at their oracle ceiling; improve tiny-box
localization (frequency zoom, IoU-aware assignment) rather than classification.

Not a priority: boundary rule / global NMS / 100-box cap (total recall loss
<0.7 pp).

---

## 6. 8-epoch convergence continuation (completed)

Pure continuation from `last.pt` (EMA), lr=2e-5, cosine to 2e-7, warmup=0,
same loss/views; corrected val every epoch. Run:
`runs/yolo11s_c1_directreg_2048x512_conv8/` (0.797 h wall).

| epoch | mAP50 | AP50 agnostic | loc recall | pred |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6384 | 0.9017 | 0.9563 | 66108 |
| 1 | 0.6415 | 0.9018 | 0.9580 | 67430 |
| 2 | 0.6406 | 0.9019 | 0.9573 | 64809 |
| 3 | **0.6423** | 0.9027 | 0.9576 | 64121 |
| 4 | 0.6414 | 0.9021 | 0.9581 | 68011 |
| 5 | 0.6398 | 0.9011 | 0.9581 | 68661 |
| 6 | 0.6403 | 0.9020 | 0.9576 | 66133 |
| 7 | 0.6405 | 0.9014 | 0.9584 | 67762 |

Baseline `best_corrected.pt`: 0.6427. Continuation best: **0.6423** at epoch 3
(-0.0004, within noise). Final epoch 7: 0.6405. Training loss stayed at
~14-20; LR ended at 2e-7.

**Verdict: no gain; the model is effectively converged for this configuration.**
Further same-loss/same-representation training is not worthwhile. The
baseline `best_corrected.pt` remains the deliverable; the next work should be
targeted (classification/verification for wide classes, short-signal time
representation, tiny-box localization), not more epochs.
