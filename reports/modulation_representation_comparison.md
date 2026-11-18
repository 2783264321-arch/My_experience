# Modulation separability of local time-frequency representations (0/1/2 and 3/4/5)

Protocol: `docs/collaboration/README.md`, plus the 2026-11-18 request to move
from threshold/routing to modulation separability. Control remains the shared
fine-0-5 filtered full val: **mAP50 0.649725** (AP_agn 0.905977, R_agn
0.957833, n_pred 47,784). Test is sealed.

## 1. Label audit: what 0-5 actually are

The downloaded AutoDL package has no class-name table; JSON only stores numeric
`class`. The mapping was cross-checked against the public SpaceNet advanced
schema and the SpaceNet/EMSpec vocabulary (details:
`reports/spaceNet_class_ids_and_wifi_labels.md`):

| id | class | design bandwidth |
| ---: | --- | ---: |
| 0 | WIFI 20MHz QPSK | 20 MHz |
| 1 | WIFI 20MHz 16QAM | 20 MHz |
| 2 | WIFI 20MHz 64QAM | 20 MHz |
| 3 | WIFI 40MHz QPSK | 40 MHz |
| 4 | WIFI 40MHz 16QAM | 40 MHz |
| 5 | WIFI 40MHz 64QAM | 40 MHz |

So 0/1/2 and 3/4/5 are **within-bandwidth three-class modulation problems**.
The six WiFi6 classes are the Cartesian product of 20/40 MHz and
QPSK/16QAM/64QAM; they are not six protocols. Local label bandwidths confirm
the split: 0-2 are exactly 20.000 MHz, 3-5 exactly 40.000 MHz. The exact
numeric order within each group is the verified public mapping; the three-class
comparison itself is invariant to a permutation of the group labels.

## 2. Proposals and shared holdout

Frozen detector `best_corrected.pt`; proposals from the existing fit coarse-view
collection (`reports/proposal_crops_fit.npz`). We kept one highest-score
proposal per `(sample_id, GT box)` and required IoU >= 0.75:

- **3,754 crops / 2,503 samples**;
- IoU 0.05 / 0.50 / 0.95 = 0.908 / 0.984 / 0.996;
- classes 0/1/2/3/4/5 = 705 / 716 / 732 / 530 / 517 / 554.

A single shared sample-level split over all 0-5 crops (20% of samples, seed
1729) gives **2,996 train crops / 2,002 samples** and **758 holdout crops /
501 samples**; the same split is used for every representation, bandwidth
group and seed. Holdout class counts are 143/148/148/92/115/112.

Overlap flags are computed from GT time-frequency intersections:

- WiFi-clean (no other 0-5 signal): **1,913**; WiFi overlap: **1,841**;
- cross-class WiFi overlap: **1,604**; same-class WiFi overlap: **405**;
- strong WiFi overlap (>=25% of the instance area): **1,513**;
- any-signal overlap: 3,393; any-clean: 361.

## 3. Representations and window choice

All three use the same proposal boxes and 10% physical context.

- **coarse**: existing 0.7 ms coarse power crop, full-band 2048 rows -> 128x256.
- **local_power**: local complex STFT at **0.1 ms** (main choice), local band,
  log power -> 128x256. 0.1 ms gives 10 kHz bins and still spans ~8 WiFi OFDM
  symbol periods; it is a deliberate compromise between frequency detail and
  time localization.
- **local_complex**: real/imag of the **same** local complex STFT coefficients
  and the same 128x256 grid. Global phase is removed at the strongest
  coefficient and each crop is jointly RMS-normalized. This is the requested
  dual-channel real/imag representation.

Power and complex therefore share one set of complex coefficients and one
output grid; the only difference is `|z|^2` vs `Re(z)/Im(z)`. Coarse and local
power are standardized per crop; complex is jointly RMS-normalized then
standardized; all are clipped to [-10, 10].

Window sensitivity: 0.05, 0.1, 0.2, 0.35 and 0.7 ms were collected in one IQ
pass. The main comparison uses 0.1 ms; the one-seed 100-epoch sweep is in
Section 6.

## 4. Training budget

A 20-epoch pilot was clearly **underfit** (coarse 012 train acc ~0.55, holdout
0.403; local power 012 train acc ~0.51). The main comparison therefore uses
**100 epochs** for every representation, with the same architecture depth/width,
AdamW lr 1e-3, weight decay 1e-4, cosine schedule, batch 64, class weights,
and seed list. Coarse and local power use 3 seeds (1729/1730/1731); the
real/imag model uses 1 seed because it overfits the training set to ~99.9% and
is at chance on the holdout, so additional seeds do not change the conclusion.
The 20-epoch pilot is kept only as a budget diagnostic.

Caveat: the same fit holdout was used to reject the 20-epoch budget and to
choose the 0.1 ms window. The main 3-seed numbers are therefore
selection-aware, not a pristine final test. They are sufficient for a
go/no-go on representation direction; the full val remains the unseen check.

## 5. Main result: 100 epochs

| representation | group | acc mean ± std | macro-F1 | confusion (sum over seeds) |
| --- | --- | ---: | ---: | --- |
| coarse | 012 | 0.4085 ± 0.0172 | 0.4081 | `[[167,160,102],[171,153,120],[116,110,218]]` |
| **local_power** | **012** | **0.4313 ± 0.0173** | **0.4325** | `[[197,141,91],[180,170,94],[108,135,201]]` |
| local_complex | 012 | 0.3485 (1 seed) | 0.3486 | `[[53,48,42],[50,50,48],[47,51,50]]` |
| coarse | 345 | 0.5528 ± 0.0015 | 0.5513 | `[[199,39,38],[91,168,86],[118,56,162]]` |
| **local_power** | **345** | **0.5799 ± 0.0210** | **0.5800** | `[[188,38,50],[101,199,45],[113,55,168]]` |
| local_complex | 345 | 0.4107 (1 seed) | 0.3954 | `[[39,9,44],[25,24,66],[27,17,68]]` |

Rows are GT, columns are the group's class IDs. Per-class recall:

- coarse 012: 0.389 / 0.345 / 0.491; local power 012: 0.459 / 0.383 / 0.453;
  complex 012: 0.371 / 0.338 / 0.338.
- coarse 345: 0.721 / 0.487 / 0.482; local power 345: 0.681 / 0.577 / 0.500;
  complex 345: 0.424 / 0.209 / 0.607.

Local fine power is the only positive representation: **+2.28 pp** on 012 and
**+2.72 pp** on 345 over coarse. The paired seed differences are +2.3 ± 1.6 pp
(012) and +2.7 ± 2.8 pp (345); with three seeds this is suggestive, not a
locked effect. The real/imag dual-channel model is far below coarse
(012 -6.0 pp; 345 -14.2 pp) and its training accuracy is ~99.9%, i.e. it
memorizes the training phase patterns instead of learning modulation.

## 6. Window sensitivity (local power, 1 seed, 100 epochs)

One seed (1729), 100 epochs, same split; local power only. These are directly
comparable to each other; the 0.1 ms main result uses three seeds in Section 5.

| local window (ms) | 012 acc | 345 acc | mean |
| ---: | ---: | ---: | ---: |
| 0.05 | 0.4214 | 0.5893 | 0.5054 |
| **0.10** | **0.4487** | **0.5862** | **0.5175** |
| 0.20 | 0.4442 | 0.5172 | 0.4807 |
| 0.35 | 0.4510 | 0.5204 | 0.4857 |
| 0.70 | 0.4032 | 0.5047 | 0.4540 |

0.1 ms is the best balanced window. 0.35 ms is marginally better on 012
(+0.23 pp) but loses 6.6 pp on 345; 0.05 ms is slightly better on 345
(+0.31 pp) but loses 2.7 pp on 012; 0.2/0.7 ms are worse on 345. The 0.1 ms
choice is therefore not a cherry-pick of a single group, and it is the window
used in the main 3-seed comparison.

## 7. Overlap-condition performance

Mean accuracy over seeds; `n` is the number of holdout crops in the condition.

| representation | group | clean WiFi | WiFi overlap | cross-class WiFi | strong WiFi >=25% |
| --- | --- | ---: | ---: | ---: | ---: |
| coarse | 012 | 0.4354 (222) | 0.3810 (217) | 0.3979 (191) | 0.3752 (183) |
| **local_power** | **012** | 0.4144 (222) | **0.4485 (217)** | **0.4520 (191)** | **0.4444 (183)** |
| local_complex | 012 | 0.3514 (222) | 0.3456 (217) | 0.3351 (191) | 0.3443 (183) |
| coarse | 345 | 0.5915 (173) | 0.5068 (146) | 0.5076 (132) | 0.4720 (113) |
| **local_power** | **345** | **0.6262 (173)** | **0.5251 (146)** | **0.5202 (132)** | **0.5162 (113)** |
| local_complex | 345 | 0.4277 (173) | 0.3904 (146) | 0.3636 (132) | 0.3717 (113) |

The local-power gain is not a clean-sample artifact:

- on 012 it loses 2.1 pp on clean WiFi but gains **+6.8 pp** under WiFi overlap
  and **+6.9 pp** under strong overlap;
- on 345 it gains on clean (+3.5 pp), overlap (+1.8 pp), cross-class (+1.3 pp)
  and strong overlap (+4.4 pp).

This is the pattern expected if local fine power preserves modulation/edge
detail that the coarse frequency aggregation smears, and if that detail helps
most when another WiFi signal is present.

## 8. Interpretation and decision

**Do not advance the real/imag dual-channel representation.** The controlled
comparison is decisive: with the same boxes, same coefficients, same grid and
same training budget, real/imag overfits to ~99.9% train accuracy and stays at
chance on the shared holdout (0.349 / 0.411). Its failure is not a coverage or
routing artifact. A complex-valued network, phase augmentation or a
phase-invariant feature may change this, but plain two-channel real/imag is not
a usable next step.

**The positive signal is local fine power, not dual-channel complex.** It beats
coarse by ~2.3-2.7 pp and improves overlap accuracy, which is the relevant
condition for this dataset. The gain is modest and seed variance is ~1.7-2.1 pp,
so it is a candidate, not a locked improvement. It also overfits training
(~99.9%), so the small CNN is still a confound.

**Recommendation: prioritize the fixed-input stronger classifier/detector
control before building a dual-channel branch.** Use the same proposals,
shared sample split and 128x256 grid; compare a stronger fixed-input model
(ResNet18/ConvNeXt-Tiny 1-channel, or detector P5/ROI features) on **coarse vs
local fine power**. This separates "representation detail" from "classifier
capacity":

- if the stronger model on coarse closes the gap, the bottleneck is capacity;
  keep the current input and invest in the detector/classifier;
- if local fine power remains ahead, integrate a **single-channel local
  fine-power** branch (not real/imag) and then run one full-val group against
  the 0.649725 control.

A local IQ/constellation check was not added yet: the short-window sweep and
the real/imag test already show that simply exposing more local phase does not
help this small CNN. If the stronger fixed-input control still fails, the next
diagnostic should be a local IQ/constellation or cyclic-feature classifier,
not a wider STFT crop.

Do not run a full-val classifier group yet; no representation has demonstrated
a reliable AP-level gain. The control remains 0.649725. Test remains sealed.

## 9. Artifacts

- `reports/spaceNet_class_ids_and_wifi_labels.md` — class-ID and WiFi6 label audit.
- `reports/local_rep/README.md` — protocol, commands, file inventory.
- `reports/local_rep/representation_comparison_100ep.json` — merged main result.
- `reports/local_rep/representation_summary_100ep.md` — generated tables.
- `reports/local_rep/window_*_012.json`, `window_*_345.json` — window sweep.
- `scripts/collect_local_proposal_crops.py`,
  `scripts/train_local_representation_classifiers.py`,
  `scripts/summarize_local_representation.py`.
- Large `.npz` representation files are server-local and gitignored.
