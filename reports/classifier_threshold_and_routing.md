# Threshold and routing analysis for the coarse proposal classifiers

Protocol: `docs/collaboration/README.md` (`67a453a`), after the merge-filter
round. This is the fit-holdout gate for the second full-val classifier run.

Control: shared fine-0-5 filter on the full val (`stage5_val_filtered_shared`):
**mAP50 0.649725**, AP50 agnostic 0.905977, R_agn 0.957833, n_pred 47,784,
0/1/2 AP 0.4212, 3/4/5 AP 0.4550, 10/11 0.6395, narrow 0.7805. Full-val time
177 s.

> **Correction (2026-11-18).** The first version of this report and
> `classifier_threshold_sweep.json` had two accounting errors:
>
> 1. `net` was `corrected - all_wrong_after_change`, so a wrong class changed to
>    another wrong class was counted as a new error. This conflated
>    wrong→another-wrong with correct→wrong and understated the net.
> 2. `acc_after` used the raw classifier label at every threshold instead of
>    the adopted label (classifier only when the gate passes).
>
> This version computes accuracy from the adopted labels and separates the
> transitions. The requested predicted-routed `t=0` check now gives **-48**
> (the old report gave -172); the old `t=0.70` net of -2 becomes **+4**. The
> decision does not change: coverage is too small and unstable to lock a
> threshold.

## 1. Holdout setup

Reused the existing checkpoints and the same fit crops/sample-level holdout as
`train_proposal_classifiers.py` (10% samples, seed 1729):

- holdout crops: **761** (GT 0/1/2 457; GT 3/4/5 304)
- detector predicted family: 0/1/2 471, 3/4/5 279, outside 0-5 11
- same GT/pred family: **701**; cross-family: **60 (7.9%)**

Training sampled by **GT class family**; deployment routes by **predicted
class family**. Cross-family cases are therefore evaluated by the wrong family
classifier (or not changed if predicted outside 0-5). This is the routing
mismatch requested.

The holdout is a fit holdout, not the val split. Full val has **4,844** boxes
in classes 0-5 (vs 47,784 predictions over all classes), so holdout coverage
rates below are scaled to that 0-5 population only.

## 2. Definitions and accounting checks

For each gate and threshold `t`:

- `adopted = classifier_label if gate >= t else detector_label`
- `changed = adopted != detector_label`
- `wrong_to_correct (W2C) = (old != GT) & (adopted == GT)`
- `correct_to_wrong (C2W) = (old == GT) & (adopted != GT)`
- `wrong_to_other_wrong (W2W) = (old != GT) & (adopted != GT) & (adopted != old)`
- `net_correct = W2C - C2W`, which must equal `correct_after - correct_before`
- `n_changed_samples` counts unique source samples, not crops

The script asserts both identities for every row:

```
changed == W2C + C2W + W2W
net_correct == correct_after - correct_before
```

All rows in the regenerated JSON pass these assertions. The two requested
checks also pass:

- predicted-routed all, `t=0`: **net -48**
- predicted-routed same-family, `t=0`: **net -48** (cross-family changes at
  `t=0` are all W2W, so they do not affect accuracy)

## 3. Confidence threshold sweep

`coverage = changed / n`; `precision = W2C / changed`.
`acc_before -> acc_after` uses the adopted labels.

| routing | t | changed | W2C | C2W | W2W | net | coverage | precision | acc before -> after | samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT-routed (oracle) | 0.00 | 452 | 150 | 175 | 127 | **-25** | 0.594 | 0.332 | 0.4717 -> 0.4389 | 239 |
| GT-routed | 0.50 | 132 | 44 | 50 | 38 | **-6** | 0.173 | 0.333 | 0.4717 -> 0.4639 | 93 |
| GT-routed | 0.65 | 28 | 10 | 8 | 10 | **+2** | 0.037 | 0.357 | 0.4717 -> 0.4744 | 22 |
| GT-routed | 0.70 | 16 | 6 | 2 | 8 | **+4** | 0.021 | 0.375 | 0.4717 -> 0.4770 | 12 |
| GT-routed | 0.75 | 9 | 1 | 1 | 7 | **0** | 0.012 | 0.111 | 0.4717 -> 0.4717 | 8 |
| pred-routed same | 0.00 | 392 | 127 | 175 | 90 | **-48** | 0.559 | 0.324 | 0.5121 -> 0.4437 | 224 |
| pred-routed same | 0.50 | 104 | 32 | 50 | 22 | **-18** | 0.148 | 0.308 | 0.5121 -> 0.4864 | 76 |
| pred-routed same | 0.65 | 19 | 8 | 8 | 3 | **0** | 0.027 | 0.421 | 0.5121 -> 0.5121 | 16 |
| **pred-routed same** | **0.70** | **10** | **6** | **2** | **2** | **+4** | **0.014** | **0.600** | **0.5121 -> 0.5178** | **8** |
| pred-routed same | 0.75 | 4 | 1 | 1 | 2 | **0** | 0.006 | 0.250 | 0.5121 -> 0.5121 | 4 |
| pred-routed all | 0.00 | 426 | 127 | 175 | 124 | **-48** | 0.560 | 0.298 | 0.4717 -> 0.4087 | 234 |
| pred-routed all | 0.50 | 117 | 32 | 50 | 35 | **-18** | 0.154 | 0.274 | 0.4717 -> 0.4481 | 82 |
| pred-routed all | 0.65 | 24 | 8 | 8 | 8 | **0** | 0.032 | 0.333 | 0.4717 -> 0.4717 | 20 |
| **pred-routed all** | **0.70** | **14** | **6** | **2** | **6** | **+4** | **0.018** | **0.429** | **0.4717 -> 0.4770** | **12** |
| pred-routed all | 0.75 | 8 | 1 | 1 | 6 | **0** | 0.011 | 0.125 | 0.4717 -> 0.4717 | 8 |
| cross-family only | 0.00 | 34 | 0 | 0 | 34 | **0** | 0.567 | 0.000 | 0.0000 -> 0.0000 | 32 |
| cross-family only | 0.70 | 4 | 0 | 0 | 4 | **0** | 0.067 | 0.000 | 0.0000 -> 0.0000 | 4 |

Full per-threshold rows for every routing are in
`reports/classifier_threshold_sweep.json`.

Exact predicted-routed rows requested by the protocol:

- **`t=0` (all)**: 426 changes on 234 samples; W2C 127, C2W 175, W2W 124;
  net **-48**; accuracy 0.4717 -> 0.4087.
- **`t=0.70` (all)**: 14 changes on 12 samples; W2C 6, C2W 2, W2W 6;
  net **+4**; accuracy 0.4717 -> 0.4770; coverage 1.8%.
- **`t=0.70` (same-family)**: 10 changes on 8 samples; W2C 6, C2W 2, W2W 2;
  net **+4**; accuracy 0.5121 -> 0.5178; coverage 1.4%.
- **`t=0.70` (cross-family)**: 4 changes on 4 samples; all W2W; net 0. These
  cannot become correct because the routed family head cannot emit the GT
  family.
- The 12 `t=0.70` all-routing sample IDs are stored in the JSON at
  `checks.t0_70.pred_routed_all.changed_sample_ids`; the same-family and
  cross-family ID lists are alongside it.

Margin sweep: no positive deployable point. GT-routed best is net 0 at
margin 0.15 (177 changes, 61 W2C / 61 C2W / 55 W2W); pred-routed all is
negative at every margin (`0.0 -48`, `0.1 -30`, `0.2 -19`, `0.3 -10`).
Full margin rows are in the JSON.

## 4. Stability: the +4 is a single-grid-point fluctuation

The only positive deployable point is `t=0.70`. It is not stable:

- **Neighboring thresholds are zero**: `t=0.65` gives 24 changes / 8 W2C /
  8 C2W / net 0; `t=0.75` gives 8 changes / 1 W2C / 1 C2W / net 0. The
  positive result exists at one point of a 12-point grid, inspected under four
  routing views.
- **Not statistically distinguishable from zero**: McNemar exact test on the
  6 W2C vs 2 C2W pairs gives **p = 0.289**. A sample-level cluster bootstrap
  (10,000 resamples over the 761 crops) gives a 95% net interval of
  **[-2, +11]** and `P(net > 0) = 0.879`; the interval includes zero.
- **Driven by a few samples**: the 6 corrections come from 5 unique samples
  (`train/5463` contributes two); the 2 regressions come from 2 samples.
  This is sample-level noise, not a population-level effect.
- **Coverage is tiny**: 1.8% of holdout 0-5 proposals (1.4% same-family). On
  the full val's 4,844 0-5 boxes this is order 10^2 boxes, far too few to
  support an AP claim.
- **The classifier itself is still near chance** on the holdout (0/1/2 0.431,
  3/4/5 0.451). A gate selected on this holdout has no reliable signal to
  gate on.

For this gate we require (i) coverage large enough to affect a non-trivial
fraction of the 4,844 val 0-5 boxes, (ii) a positive net whose 95% interval
excludes zero, and (iii) the same sign at adjacent thresholds. The best point
fails (ii) and (iii), and its 1.8% coverage is only order 90 boxes on val.
This fails the reliability gate: coverage is not sufficient and the positive
net is not stable.

## 5. Routing discrepancy

- 60/761 = **7.9%** of GT 0-5 holdout crops have a detector predicted class in
  the other family (or outside 0-5: 11 crops).
- Every cross-family change is wrong by construction: 34/34 at `t=0`, 4/4 at
  `t=0.70`. Family-specific routing cannot correct these cases; it can only
  change one wrong label to another wrong label.
- Even with **oracle GT routing**, the negative `t=0` net (-25) and the
  near-zero positive points show the classifier, not only routing, is weak.
- The family-specific holdout also makes cross-family pred-routed numbers
  indicative, not a clean deployment estimate: the other family classifier may
  have seen those samples in its own training split. A clean estimate requires
  a shared sample-level holdout for both heads.

## 6. Decision

**No threshold passes the gate.** We did not lock a threshold and did not run
the second full-val group; the control remains **0.649725**. The small
coarse-crop classifiers are not usable with the current near-chance holdout
accuracy and no reliable net correction.

The corrected accounting is important for the record: the early `t=0` penalty
was -172 under the old definition, but the true accuracy net is -48. The
classifier is less harmful than the first report said, but still harmful
without a gate, and the only positive gate is too small and too fragile to
adopt.

## 7. Next step: shared-holdout unified 0-5 coarse model

Recommendation: **retrain one unified 0-5 coarse-image classifier on a shared
sample-level holdout**, instead of tuning the two family heads further.

Why this is the right next comparison:

- **Cross-family routing is unfixable with two heads.** 7.9% of holdout crops
  are cross-family; every cross-family change is W2W. A unified 6-class head
  (0-5, optionally + background) can emit 2<->3 and cross-family labels
  directly.
- **Shared holdout removes the leakage caveat.** One sample-level split used by
  all classes gives a clean deployment estimate; no crop is scored by a head
  that saw it in training.
- **The current holdout signal is too weak to tune a gate.** The best
  deployable point is +4 on 761 crops with CI including zero. A stronger model
  should first show a stable W2C > C2W margin before any full-val run.

Concrete next round:

- Same frozen detector, same pre-global-merge candidate boxes, same coarse
  crops and physical context. Only the classifier changes.
- Unified 6-class head over 0-5 (+ background if hard negatives are included),
  shared sample-level holdout (e.g. 10-15% of samples), class-balanced sampling
  and hard negatives.
- Stronger backbone: ResNet18 / ConvNeXt-Tiny with 1-channel input, or detector
  P5/ROI features; light augmentation, longer training, early stopping on the
  shared holdout. Keep the current small CNN as the architecture control.
- Gate selection only on the shared fit holdout. Require W2C > C2W with a
  confidence interval excluding zero and meaningful coverage before touching
  full val.
- If the gate passes, run one full-val group with fixed boxes/scores, class
  changes only, then class-wise NMS + 100, and compare against the filtered
  **0.649725** control. Each group merged once. Do not tune on val.
- If the unified coarse model still fails, then run the same-proposal-box
  coarse vs native STFT crop comparison with the same architecture, physical
  context, split and fusion rule. Do not naive-resize the 16.8k-33.6k-row
  native crop to 128x256.

10/11 remains separate (mixed bandwidth; fine helps narrow instances).
6/7/8/9/12/13 are not classifier targets. Test remains sealed.

Artifacts:
- `reports/classifier_threshold_sweep.json`
- `scripts/classifier_threshold_sweep.py`
- control: `runs/yolo11s_c1_directreg_2048x512/stage5_val_filtered_shared/` (server)
- previous round: `reports/proposal_classifier_experiment.md`
