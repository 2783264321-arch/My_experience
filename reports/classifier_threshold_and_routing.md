# Threshold and routing analysis for the coarse proposal classifiers

Protocol: `docs/collaboration/README.md` (`67a453a`), after the merge-filter
round. This is the fit-holdout gate for the second full-val classifier run.

Control: shared fine-0-5 filter on the full val (`stage5_val_filtered_shared`):
**mAP50 0.649725**, AP50 agnostic 0.905977, R_agn 0.957833, n_pred 47,784,
0/1/2 AP 0.4212, 3/4/5 AP 0.4550, 10/11 0.6395, narrow 0.7805. Full-val time
177 s.

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

## 2. Confidence threshold sweep

`changed/corrected/wrong/net/ratio/precision`, `acc_before -> acc_after`.
`GT-routed` = classifier chosen by GT family (clean, as training; not
deployable). `pred-routed same` = deployment routing restricted to clean
same-family cases. `pred-routed all` includes cross-family cases (indicative;
the other classifier may have seen those samples in its own training split).

| routing | t | changed | corrected | wrong | net | ratio | precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT-routed | 0.0 | 452 | 150 | 302 | -152 | 0.594 | 0.332 |
| GT-routed | 0.5 | 132 | 44 | 88 | -44 | 0.173 | 0.333 |
| GT-routed | 0.65 | 28 | 10 | 18 | -8 | 0.037 | 0.357 |
| GT-routed | 0.70 | 16 | 6 | 10 | -4 | 0.021 | 0.375 |
| pred-routed same | 0.0 | 392 | 127 | 265 | -138 | 0.559 | 0.324 |
| pred-routed same | 0.5 | 104 | 32 | 72 | -40 | 0.148 | 0.308 |
| pred-routed same | 0.65 | 19 | 8 | 11 | -3 | 0.027 | 0.421 |
| **pred-routed same** | **0.70** | **10** | **6** | **4** | **+2** | **0.014** | **0.600** |
| pred-routed same | 0.75 | 4 | 1 | 3 | -2 | 0.006 | 0.250 |
| pred-routed all | 0.0 | 426 | 127 | 299 | -172 | 0.560 | 0.298 |
| pred-routed all | 0.5 | 117 | 32 | 85 | -53 | 0.154 | 0.274 |
| pred-routed all | 0.70 | 14 | 6 | 8 | -2 | 0.018 | 0.429 |
| cross-family only | 0.0 | 34 | 0 | 34 | -34 | 0.567 | 0.000 |
| cross-family only | 0.5 | 13 | 0 | 13 | -13 | 0.217 | 0.000 |

Accuracy: GT-routed before/after 0.472 -> 0.439; pred-routed same 0.512 ->
0.444; pred-routed all 0.472 -> 0.409.

Margin sweep (GT-routed) is also negative at every margin:
0.0: -152; 0.1: -83; 0.2: -50; 0.3: -30.
Same-family margin sweep: 0.0 -138; 0.1 -73; 0.3 -24.

## 3. Routing discrepancy

- 60/761 = **7.9%** of GT 0-5 holdout crops have a detector predicted class in
  the other family (or outside 0-5: 11 crops).
- Every cross-family change is wrong (34/34 at t=0; 13/13 at t=0.5;
  4/4 at t=0.7). The training-by-GT set contains these boxes, but deployment
  routes them to the wrong classifier, so this scheme cannot correct them.
- Even with **oracle GT routing**, net correction is negative at all
  thresholds; the classifier itself, not only routing, is the problem.
- The best deployable point is t=0.70 same-family: 10 changed / 6 corrected /
  4 wrong, net +2 on 10/701 (1.4% coverage). This is not a reliable net
  correction.

Caveat: the family-specific holdout means cross-family pred-routed numbers are
indicative, not a clean held-out estimate (the other family classifier may have
seen those samples). A clean deployment estimate requires a shared
sample-level holdout for both classifiers.

## 4. Decision

**No threshold passes the reliability gate.** We did not lock a threshold and
did not run the second full-val group; the control remains 0.649725. The small
coarse-crop classifiers are not usable with the current near-chance holdout
accuracy (0/1/2 0.431, 3/4/5 0.451) and no reliable net correction.

## 5. Next step: stronger coarse model, then native STFT crop if needed

Recommendation: **try a stronger coarse model first**, not the native STFT
crop.

Why:

- The current small CNN is the primary confound: it is underfit (training loss
  ~0.98/0.90), near chance on holdout, and even oracle GT routing is negative.
- Routing mismatch is only 7.9%; fixing routing alone cannot turn the
  classifier around. A stronger coarse classifier tests whether the coarse
  representation carries the class information.
- Native crops are more expensive/design-sensitive (7-27x taller than coarse;
  naive resize loses detail). Use them only if a stronger coarse classifier
  still fails.

Concrete next classifier round:

- Shared sample-level holdout for both families; 6-class + background head for
  0-5 (handles 2<->3 and cross-family) or family heads with a gate.
- Stronger backbone: ResNet18/ConvNeXt-Tiny (1-channel input), or ROI features
  from detector P5/coarse features; more hard negatives, light augmentation,
  longer training.
- Confidence/margin gating: only change when the classifier is confident;
  threshold chosen on the shared fit holdout, not val.
- Evaluate on full val with the same pre-global-merge candidate boxes, boxes
  and scores fixed, each group merged once.

If the stronger coarse model still gives no reliable net:

- Run the **same-proposal-box coarse vs native STFT crop** comparison with the
  same architecture, physical context, fit/val split and fusion rule.
- Use 256x256/512x512 or a 1D frequency profile / two-branch
  coarse+native-profile; do not naive-resize the 16.8k-33.6k-row native crop
  to 128x256.
- Measured cost: crop+resize 0.15-0.27 ms; ResNet18 0.07-0.15 ms/proposal;
  4.8k boxes ~1-5 s extra, small.

10/11 remains separate (mixed bandwidth; fine helps narrow instances).
6/7/8/9/12/13 are not classifier targets. Test remains sealed.

Artifacts:
- `reports/classifier_threshold_sweep.json`
- `scripts/classifier_threshold_sweep.py`
- control: `runs/yolo11s_c1_directreg_2048x512/stage5_val_filtered_shared/` (server)
- previous round: `reports/proposal_classifier_experiment.md`
