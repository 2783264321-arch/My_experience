# Paired C1/C3 complex pilot: first end-to-end comparison

Question: with the same power inputs already present, does adding Re/Im to both
the overlapping coarse view and the native fine slice improve broadband
modulation recognition while preserving narrowband detection?

Scope: **short pilot**, 256 fit / 128 val samples, one fixed fine slice per
sample, 50 epochs, two seeds. This is a screening experiment, not a full-val
result. The 0.649725 control is unchanged; test remains sealed.

**Bottom line: no stable C3 gain. 0/1/2 and 3/4/5 AP deltas flip sign by seed;
10/11 AP and localization recall fall in both seeds. C3 is not adopted; the next
control is coarse-only vs fine-only complex attribution.**

## 1. Paired-design review

The design is sound in the important places:

- C1 uses cache channel 0 (power) for both coarse and fine.
- C3 uses the same channel 0 and adds Re/Im channels 1/2. The power tensors are
  **bitwise identical**; the pairing check reports `power_identical=True`.
- C1/C3 start from paired initialization checkpoints: C3's power stem equals
  C1's summed RGB stem and its Re/Im stem channels are zero. Initial logits and
  loss are **exactly identical** (`max_abs_diff=0`, loss difference `0`).
- Same subset, fixed views, labels, model body, optimizer, steps and seed list.
- Coarse power/complex come from the same 0.7 ms STFT and the same 50%-overlap
  linear-Hann weights; fine power/complex come from the same native complex
  coefficients.
- Complex calibration: coarse scale 0.005885, fine scale 0.010778; Re/Im std
  about 0.10-0.12; power clip 0.66%/1.94%, complex clip 0.23%/0.20%.

Caveats that limit the conclusion:

- only 128 val samples (128 samples × 2 views), so group AP is noisy;
- one fixed fine slice per sample, not all Baseline-1 fine views;
- 50 epochs on 256 sources is far from the baseline's full training budget;
- two seeds show large swings, especially 0/1/2;
- absolute mAP is not comparable to 0.6497; only the paired C1/C3 difference is
  meaningful.

## 2. Results at 50 epochs

| seed | arm | mAP50 | AP_agn | loc recall | class acc | n_pred |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1729 | C1 | 0.2905 | 0.5912 | 0.8184 | 0.5916 | 8436 |
| 1729 | C3 | 0.2776 | 0.5810 | 0.8157 | 0.6071 | 8091 |
| 1730 | C1 | 0.2969 | 0.5956 | 0.8267 | 0.6290 | 8158 |
| 1730 | C3 | 0.3048 | 0.5862 | 0.8102 | 0.6265 | 7894 |
| **mean** | **C1** | **0.2937** | **0.5934** | **0.8226** | **0.6103** | **8297** |
| **mean** | **C3** | **0.2912** | **0.5836** | **0.8129** | **0.6168** | **7993** |
| **mean Δ** | C3−C1 | **−0.25 pp** | **−0.98 pp** | **−0.96 pp** | +0.65 pp | −305 |

Group class-AP mean (same fixed fine slice; not full Baseline-1):

| seed | arm | 0/1/2 | 3/4/5 | 10/11 | narrow 9/12/13 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1729 | C1 | 0.2176 | 0.1456 | 0.2355 | 0.1686 |
| 1729 | C3 | 0.1457 | 0.1448 | 0.2211 | 0.1992 |
| 1730 | C1 | 0.1393 | 0.1499 | 0.2423 | 0.2487 |
| 1730 | C3 | 0.2055 | 0.1910 | 0.2223 | 0.2323 |
| **mean** | **C1** | **0.1784** | **0.1477** | **0.2389** | **0.2086** |
| **mean** | **C3** | **0.1756** | **0.1679** | **0.2217** | **0.2157** |
| **mean Δ** | C3−C1 | **−0.28 pp** | **+2.02 pp** | **−1.72 pp** | **+0.71 pp** |

Per-seed group deltas show why the mean is not a conclusion:

| group | seed 1729 Δ | seed 1730 Δ | mean Δ |
| --- | ---: | ---: | ---: |
| 0/1/2 AP | −7.19 pp | +6.62 pp | −0.28 pp |
| 3/4/5 AP | −0.08 pp | +4.11 pp | +2.02 pp |
| 10/11 AP | −1.44 pp | −2.00 pp | −1.72 pp |
| narrow AP | +3.06 pp | −1.64 pp | +0.71 pp |

The 0/1/2 and 3/4/5 signs flip with seed; 10/11 is the only consistently
negative group. Localization recall is also consistently slightly lower
(seed 1729 −0.27 pp, seed 1730 −1.65 pp). Narrowband AP improves in one seed
and degrades in the other; its recall moves −6.6 pp in seed 1730.

Broadband matched-instance counts (IoU 0.5, 100 cap):

| seed | arm | 0/1/2 correct/wrong | 3/4/5 correct/wrong |
| ---: | --- | ---: | ---: |
| 1729 | C1 | 24 / 63 | 17 / 34 |
| 1729 | C3 | 19 / 64 | 9 / 42 |
| 1730 | C1 | 15 / 68 | 12 / 40 |
| 1730 | C3 | 27 / 57 | 17 / 37 |

Again, the direction flips by seed. C3 can improve matched wide-class counts in
one seed and hurt them in the other; the AP mean does not establish a stable
broadband gain.

## 3. Training-length sensitivity

Seed 1729 early curves (same data/init, 10/30/50 epochs):

| epochs | C1 mAP | C3 mAP | 0/1/2 C1→C3 | 3/4/5 C1→C3 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.1171 | 0.0953 | 0.0647→0.0441 | 0.0000→0.0000 |
| 30 | 0.2526 | 0.2696 | 0.1603→0.1272 | 0.1209→0.1863 |
| 50 | 0.2905 | 0.2776 | 0.2176→0.1457 | 0.1456→0.1448 |

At 30 epochs C3 looked better overall and on 3/4/5; at 50 epochs C1 overtook it
and 0/1/2 heavily favored C1. Both models are still improving at 50 epochs
(C1 train loss 21.17, C3 18.13). C3 fits the training set better, but its validation gain is not stable
(seed 1729 lower, seed 1730 slightly higher); the extra channels behave like an
overfit path, not a stable representation gain.

## 4. Cost

Two-seed mean: C1 **257.3 s**, C3 **287.8 s** for the same 50 epochs / 1600
steps (+11.9%); peak GPU memory 4.39 vs 4.48 GiB (+2.1%). C3 produces fewer
predictions (7993 vs 8297). The cache build is one-time (~5 min for 384
samples) and is not included per arm.

## 5. Decision

**No stable broadband gain; do not adopt full C3 and do not run a full-val
training group yet.** The paired power/initialization is clean, so the negative
result is about the extra Re/Im channels, not a data mismatch. Two seeds are
enough to show that the C3 effect is smaller than seed variance on 0/1/2 and
narrowband groups; 10/11 AP and localization recall move consistently in the
wrong direction. The 30-vs-50 epoch reversal also rules out a short-run
adoption.

This does not prove complex information is useless; it shows that adding Re/Im
to **both** coarse and fine at this capacity/budget is not a reliable win.

## 6. Next best control

**Attribute the complex channels: C3-coarse-only vs C3-fine-only**, both with
the same 3-channel model and power inputs:

- coarse-only: coarse power+Re/Im active, fine Re/Im zeroed;
- fine-only: coarse Re/Im zeroed, fine power+Re/Im active.

Use the same paired init (C3 init), 2-3 seeds, 50 epochs, and a larger val
subset (256-384 samples) because the current 0/1/2 seed swing is already ~14 pp.
This is the minimal experiment that distinguishes:

- coarse complex helping 3/4/5 full-band context while fine power preserves
  narrowband;
- fine complex helping narrowband but injecting phase noise into wide classes;
- both being harmful, in which case keep C1 and move to phase augmentation /
  channel dropout or a phase-invariant complex feature.

If only one follow-up arm is affordable, run **C3-coarse-only** first: 0/1/2 is
the group the project cares most about, and the coarse view is where complex
full-band context could plausibly help 3/4/5 without disturbing the native fine
power. Add phase augmentation/channel dropout only after the attribution arm.

Artifacts: `reports/complex_pilot_c1_c3_summary.json`,
`reports/complex_pilot_pairing_check.json`,
`scripts/build_complex_pilot_cache.py`,
`scripts/train_complex_pilot.py`,
`scripts/prepare_complex_pilot_init.py`.
