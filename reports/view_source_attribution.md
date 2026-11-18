# View-source attribution (frozen `best_corrected.pt`)

Protocol: `docs/collaboration/README.md`, commit `40e94aa`.
Run: `yolo11s_c1_directreg_2048x512`, checkpoint `best_corrected.pt`.
Data: fixed val, 1543 samples / 24,975 views; test sealed.

One all-view inference; per-view NMS (`conf=0.001`, `iou=0.5`, `max_det=300`),
then the internal-boundary filter. Before the global merge, predictions are
split into **coarse-only**, **fine-only** and **both**. Each group gets the same
class-wise global NMS (`iou=0.5`) and per-sample 100-box cap.

Artifacts:
- `runs/yolo11s_c1_directreg_2048x512/stage5_val/view_source_attribution.json`
  (full metrics, FP/FN taxonomy, top examples, boundary counts; server-local)
- `docs/collaboration/opencode-view-attribution.md` (shared summary)
- Entry alignment: `predict_sample` now takes `max_det_view=300` (same as the
  full validator), instead of hard-coded 1000.

## Reproduction

| metric | baseline `stage5_val` | both group | delta |
| --- | ---: | ---: | ---: |
| mAP50 | 0.6427009 | 0.6427025 | +0.0000017 |
| n_pred | 65025 | 65027 | +2 |
| boundary dropped | 202142 | 202131 | -11 |

The per-class APs match the baseline to 3 decimals. The tiny delta is numerical
nondeterminism from chunk composition: the original validator flushed a global
buffer of 8 views across sample boundaries; this run processes each sample's
views in per-sample chunks. The source conclusions below are unaffected.

## Overall source metrics

| source | mAP50 | AP50 agnostic | R_agn | n_pred | cls accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| coarse only | 0.5794 | 0.8178 | 0.8773 | 25,409 | 0.8278 |
| fine only | 0.3165 | 0.5013 | 0.5564 | 45,629 | 0.8949 |
| both | **0.6427** | 0.9046 | 0.9578 | 65,027 | 0.8433 |

The union is best overall, but the union is a **view-dependent** mixture:
coarse supplies the wide classes; fine supplies narrow classes.

## Wide groups

`AP` is the mean of the per-class APs in the group; `FP@0.5(m/d/b)` counts
false positives with score >=0.5 as misclass / duplicate / background;
`GT(c/w/m)` is GT covered with the correct class / wrong class / missed.

### 0/1/2 (20 MHz)

| source | AP | R_agn | TP/FP/FN | FP(m/d/b) | FP@0.5(m/d/b) | GT(c/w/m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse | 0.4212 | 0.9699 | 806/2783/191 | 1509/32/1242 | 464/2/22 | 437/530/30 |
| fine | 0.0000 | 0.0000 | 0/13503/997 | 293/0/13210 | 0/0/0 | 0/0/997 |
| both | 0.4020 | 0.9699 | 805/15970/192 | 1795/31/14144 | 464/2/22 | 437/530/30 |

Per-class AP (coarse / fine / both): 0 = 0.368/0/0.354,
1 = 0.347/0/0.329, 2 = 0.548/0/0.523.

### 3/4/5 (40 MHz)

| source | AP | R_agn | TP/FP/FN | FP(m/d/b) | FP@0.5(m/d/b) | GT(c/w/m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse | 0.4550 | 0.9661 | 337/926/105 | 516/14/396 | 183/1/18 | 224/203/15 |
| fine | 0.0000 | 0.0000 | 0/4993/442 | 110/0/4883 | 0/0/0 | 0/0/442 |
| both | 0.4415 | 0.9661 | 337/5787/105 | 624/14/5149 | 183/1/18 | 224/203/15 |

Per-class AP (coarse / fine / both): 3 = 0.503/0/0.492,
4 = 0.438/0/0.427, 5 = 0.424/0/0.405.

### 10/11 (mixed 0.3/1.6/10 MHz)

| source | AP | R_agn | TP/FP/FN | FP(m/d/b) | FP@0.5(m/d/b) | GT(c/w/m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse | 0.5908 | 0.9679 | 630/1249/86 | 575/21/653 | 273/0/15 | 452/241/23 |
| fine | 0.2157 | 0.2430 | 165/5855/551 | 459/0/5396 | 0/0/42 | 161/13/542 |
| both | 0.6395 | 0.9721 | 634/6934/82 | 1021/14/5899 | 273/0/51 | 456/240/20 |

Per-class AP (coarse / fine / both): 10 = 0.572/0.340/0.708,
11 = 0.610/0.091/0.571.

### Narrow 9/12/13 (context)

| source | AP | R_agn | TP/FP/FN | FP(m/d/b) | FP@0.5(m/d/b) | GT(c/w/m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse | 0.4838 | 0.6671 | 1401/12506/759 | 378/311/11817 | 6/1/31 | 1372/69/719 |
| fine | 0.7821 | 0.9032 | 1942/10329/218 | 425/90/9814 | 4/1/73 | 1925/26/209 |
| both | 0.7805 | 0.9162 | 1965/21039/195 | 723/165/20151 | 10/1/94 | 1946/33/181 |

Per-class AP (coarse / fine / both): 9 = 0.631/0.952/0.948,
12 = 0.509/0.738/0.739, 13 = 0.312/0.657/0.654.

## Key findings

1. **Wide classes are coarse-view classes.** For 0/1/2 and 3/4/5,
   coarse-only R_agn is 0.966-0.970; fine-only R_agn/AP is 0.000. Fine views
   cannot contain a complete 20/40 MHz box. The both group has the same GT
   coverage as coarse-only for these groups, but **13,210 + 4,883 additional
   background FPs** from fine. Both AP is slightly lower than coarse-only
   (0.402 vs 0.421; 0.442 vs 0.455).
2. **High-score errors are coarse-view misclassification, not fine
   background.** FP@0.5 for 0/1/2 is 464 misclass / 2 duplicate / 22 background
   from coarse and 0/0/0 from fine; for 3/4/5 it is 183/1/18 from coarse and
   0/0/0 from fine. Fine's FPs are mostly low-score background.
3. **The correct box is often assigned the wrong class.** GT covered with the
   wrong class: 530/997 (0/1/2), 203/442 (3/4/5), 240/716 (10/11). Examples
   below are high-confidence and nearly exact in time-frequency extent. This is
   the classification gap already measured by oracle-class AP (0.83-0.87
   ceiling vs 0.40-0.47 actual for these groups).
4. **10/11 is mixed.** Fine-only finds the narrow 10/11 instances (R_agn 0.243)
   but adds 5,396 background FPs and 42 high-score background FPs; both AP is
   0.640, higher than coarse 0.591, mainly via class 10. A blanket "drop fine"
   for 10/11 would lose useful narrow-instance TPs; use bandwidth-aware gating.
5. **Narrow 9/12/13 is fine-view territory.** Fine-only R_agn 0.903,
   AP 0.782; coarse-only R_agn 0.667, AP 0.484; both is essentially fine
   (0.7805 vs 0.7821). Coarse adds ~10.7k FPs and only ~23 extra TPs.
6. **Boundary/near-duplicate are secondary.** Boundary candidates: 202,131
   total; wide groups 81,421/57,866/30,809. Near-duplicate GT FN counts:
   27/11/3 for the wide groups vs 139/86/59 wrong-class-covered and 26/8/20
   missed. It is not the dominant wide-class error.

## Representative errors

High-confidence, well-localized, wrong class (source both/coarse):

| sample | source | pred | score | GT | IoU | box (ms / MHz) | GT (ms / MHz) |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| train/3254 | both | 2 | 0.981 | 1 | 0.991 | 0.1-16.0 / 2447.0-2466.9 | 0.1-16.0 / 2447.0-2467.0 |
| train/520 | both | 2 | 0.977 | 1 | 0.999 | 11.0-40.0 / 2452.0-2472.0 | 11.0-40.0 / 2452.0-2472.0 |
| train/3126 | both | 5 | 0.984 | 4 | 0.992 | 14.2-40.0 / 2442.0-2482.0 | 14.2-40.0 / 2442.0-2482.0 |
| train/2777 | both | 10 | 0.974 | 11 | 0.983 | 0.5-40.0 / 2447.0-2457.0 | 0.5-40.0 / 2447.0-2457.0 |
| train/383 | both | 10 | 0.973 | 11 | 0.981 | 0.0-18.7 / 2447.0-2457.0 | 0.0-18.7 / 2447.0-2457.0 |
| train/5654 | both | 3 | 0.762 | 4 | 0.998 | 0.0-20.0 / 2422.0-2462.0 | 0.0-20.0 / 2422.0-2462.0 |
| train/6901 | both | 0 | 0.468 | 1 | 0.998 | 0.0-20.0 / 2447.0-2467.0 | 0.0-20.0 / 2447.0-2467.0 |
| train/5531 | both | 10 | 0.617 | 11 | 0.912 | 0.0-20.0 / 2437.0-2447.0 | 0.0-20.0 / 2437.0-2447.0 |

High-score background from fine (10/11): train/5385 pred 11, score 0.945,
best GT class 11 IoU 0.159; train/5958 pred 11, 0.941, best IoU 0.157. These
are fine-view fragments/background, not complete 10 MHz events.

Near-duplicate GT examples: train/1695 GT 1, same-class pred 0 at IoU 0.997;
train/5700 GT 3, pred 3 at IoU 0.996; train/5531 GT 11, pred 10 at IoU 0.912.
The one-to-one rule keeps one match; the duplicate GT is an FN, as required by
the no-dedup policy.

## Recommended first change

**Coarse-proposal classifier / verification for the wide families
`0/1/2`, `3/4/5`, `10/11`.**

Evidence:

- Correct boxes are found: wide-group R_agn 0.966-0.970; GT correct+wrong
  covers 967/997, 427/442, 696/716 (the remaining are misses).
- Classes are often wrong: 530/203/240 GT covered by the wrong class.
- High-score errors are coarse-view misclass: 464/183/273 FP@0.5, while
  fine-view high-score misclass is 0 for 0-5.
- Oracle-class AP for these groups is 0.83-0.87 vs 0.40-0.47 actual.

Concrete next step: freeze the detector; take coarse-view proposals (or
GT-matched coarse proposals) on fit; train a lightweight 14-class (or
family-aware) classifier on coarse crops/features with hard negatives; evaluate
on the same val with the frozen detector. A proposal-level reclassification
head is enough for the first test; no detector retraining.

**Cheap secondary change (after/with the classifier): view-aware merge.**

- For `0/1/2` and `3/4/5`, drop fine-view predictions: fine-only R_agn=AP=0,
  and fine adds 13,210/4,883 background FPs with no TPs; both AP is slightly
  below coarse-only.
- For `10/11`, keep fine but gate by bandwidth/source: a fine view covers
  2.93 MHz, so a 10 MHz signal cannot be complete in it; use coarse for the
  wide instances, fine for the narrow 10/11 instances. Do not blanket-drop.
- For `9/12/13`, prefer fine; both is already close to fine-only.

Do **not** prioritize more training, boundary/NMS/max-det tuning, or a global
class-weight fine-tune (the latter was also affected by the verified `w²`
implementation).

## Notes

- Both-group reproduction: mAP +1.7e-6, n_pred +2, boundary -11; per-class APs
  match to 3 decimals. See the reproduction section.
- Full per-class APs, all FP/FN examples, and boundary counts are in
  `view_source_attribution.json` (server run directory).
- Test remains sealed.
