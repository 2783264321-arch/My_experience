# C3 replication and zero-complex control

Date: 2026-11-19
Starting point: `92e0122`
Subset: fixed 1536 fit / 384 val, 20,000 optimizer steps, effective batch 16,
no early stopping, best and last EMA both evaluated. Test remains sealed.

## What was added

- `dense` seed 1730, same dense operator/cache/scale/budget as seed 1729.
- `zero3` control: 3-channel `power + 0 + 0`, trained from scratch, same
  3-channel architecture, same power stem/head initialization seed and budget,
  seeds 1729 and 1730.
- `best` and `last` checkpoints evaluated for dense and zero3.
- Matched-box analysis: a GT 3/4/5 box is *common* only if both models have a
  class-agnostic IoU>=0.5 match to it. Classification accuracy/confusion is then
  compared on exactly those boxes.
- Dense estimator comparison on 6 real samples + synthetic tone/noise:
  current per-pair normalize-then-average vs aggregate-numerator/energy-first.

`zero3` is a **no-complex-information control within the same 3-channel
architecture/head**, not a live-capacity control: its two extra input channels
are identically zero and their stem weights remain zero. It isolates the value
of live, IQ-aligned complex channels from architecture/head/seed effects. A
separate live-but-uninformative control would be needed to test raw capacity.

## Best / last per seed

| model | checkpoint | mAP50 | 0/1/2 | 3/4/5 | 10/11 | 9/12/13 | AP_agn | R_agn |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense s1729 | best | 0.5071 | 0.2452 | 0.4646 | 0.4464 | 0.4288 | 0.7850 | 0.8493 |
| dense s1729 | last | 0.4765 | 0.2340 | 0.3825 | 0.4117 | 0.4146 | 0.7555 | 0.8242 |
| dense s1730 | best | 0.5174 | 0.2786 | 0.4439 | 0.4864 | 0.4314 | 0.7867 | 0.8497 |
| dense s1730 | last | 0.4678 | 0.2237 | 0.3821 | 0.3993 | 0.3857 | 0.7495 | 0.8160 |
| zero3 s1729 | best | 0.5319 | 0.3135 | 0.4344 | 0.5704 | 0.4150 | 0.7836 | 0.8638 |
| zero3 s1729 | last | 0.4827 | 0.2727 | 0.3671 | 0.4130 | 0.4075 | 0.7614 | 0.8229 |
| zero3 s1730 | best | 0.4988 | 0.2532 | 0.4052 | 0.4772 | 0.4190 | 0.7836 | 0.8470 |
| zero3 s1730 | last | 0.4666 | 0.1984 | 0.3595 | 0.4585 | 0.3965 | 0.7541 | 0.8174 |
| dense mean | best | 0.5123 | 0.2619 | **0.4543** | 0.4664 | 0.4301 | 0.7859 | 0.8495 |
| zero3 mean | best | **0.5156** | **0.2834** | 0.4198 | **0.5253** | 0.4170 | 0.7836 | **0.8554** |
| dense mean | last | 0.4722 | 0.2289 | 0.3823 | 0.4055 | 0.4002 | 0.7525 | 0.8201 |
| zero3 mean | last | **0.4747** | **0.2356** | 0.3633 | **0.4358** | 0.4020 | **0.7578** | **0.8202** |

**Dense vs zero3 (same 3ch architecture/head/seed):**

- best mean: overall −0.31 pp, 0/1/2 −2.15 pp, **3/4/5 +3.45 pp**,
  10/11 −5.88 pp, narrow +1.31 pp.
- last mean: overall −0.25 pp, 0/1/2 −0.67 pp, **3/4/5 +1.90 pp**,
  10/11 −3.03 pp, narrow −0.19 pp.
- The 3/4/5 residual is positive for both seeds (s1729 +3.02, s1730 +3.87);
  overall/0/1/2/10-11 do not favour dense.

## Common-box 3/4/5 classification

Only GT boxes matched by both models at IoU>=0.5 are included.

| pair | common n | dense acc | zero3 acc | Δ | dense recall | zero3 recall | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1729 best | 135 | 0.5037 | 0.4593 | **+4.44 pp** | 0.901 | 0.934 | 0.461 |
| s1729 last | 118 | 0.4915 | 0.4576 | **+3.39 pp** | 0.816 | 0.849 | 0.608 |
| s1730 best | 134 | 0.5224 | 0.4478 | **+7.46 pp** | 0.908 | 0.908 | 0.220 |
| s1730 last | 120 | 0.4750 | 0.4500 | **+2.50 pp** | 0.842 | 0.868 | 0.771 |

- Dense wins the common-box classification accuracy in all four comparisons
  (+2.50 to +7.46 pp; best-mean +5.95, last-mean +2.95), but no individual
  McNemar test reaches significance (p 0.22–0.77).
- Dense localization recall is equal or lower, so the 3/4/5 AP gain is **not**
  from finding more boxes; the evidence points to modulation discrimination
  and/or precision/ranking on already localized boxes.
- The effect is not uniform by class: on s1729 best, zero3 is much better on
  class 3 (32/43 vs 21/43) while dense is better on 4 and 5; dense has a more
  balanced 3/4/5 confusion. Full 3×3 + other tables are in
  `reports/c3_matched_*.json`.

## Dense estimator comparison

Current production: per-pair normalized correlation, then average per column.
Alternative: sum phase-corrected correlation and pair energy per column, then
divide once. Six real samples and synthetic tone+noise:

- Synthetic noise region mean |corr|: current **0.528**, aggregate **0.619**;
  target-tone magnitude is ~1 in both.
- Real GT 3/4/5 and outside-GT regions move together: e.g. train/29 GT
  0.314→0.453, noise 0.316→0.456; train/14 GT 0.687→0.864, noise 0.689→0.860.
- Aggregate-first increases both signal and noise magnitudes; it does not
  improve the GT/noise separation or weak-signal ranking in these samples.
- **Keep the current per-pair normalized operator.** The alternative is recorded
  in `reports/c3_dense_correlation_comparison.json` but was not used for cache
  training.

## Judgment

1. **No complex front-end beats the no-complex 3ch control overall.**
   zero3 mean 0.5156 vs dense 0.5123, overlap 0.5078, sparse 0.4998.
   The C1→complex overall gains are within architecture/seed/training variance;
   zero3's high seed1729 result (0.5319) is a warning against single-seed
   conclusions.
The replicated long-window overlap 3/4/5 gain vs C1 (+4.01 pp) is **not** a
complex-information effect either: zero3 0.4198 > overlap 0.3976. Only dense
exceeds the same-architecture no-complex control on 3/4/5.

2. **0/1/2 is not a complex-info gain.** zero3 is best (0.2834) and dense is
   worse (0.2619); live complex channels do not help 20 MHz Wi-Fi.
3. **40 MHz (3/4/5) has a small, directionally consistent dense residual.**
   Dense is +3.45 pp (best mean) vs zero3, positive in both seeds; common-box
   classification is +2.95 to +5.95 pp in the same direction. It is not
   statistically strong (McNemar p>0.2) and recall is not higher.
4. **Do not run full training yet.** A focused follow-up is justified:
   - dense vs zero3 with 3–5 seeds;
   - 768+ val samples so common-box counts/ McNemar have power;
   - a **live** uninformative capacity control (e.g. duplicate power or random
     channels) to separate raw capacity from complex information;
   - only if the 3/4/5 residual survives, expand to the full 37k/full val.
   Do not expand 0/1/2 or overall; no evidence there.
5. **Cheaper alternatives:** in a later round, a 3ch model with duplicated
   power/random live channels is more informative than zero3 for capacity;
   zero3 is the right no-information control for the same architecture.
   Matched-box analysis reuses one validation pass and is cheaper than more
   training.

Test remains sealed. Absolute mAP is on the 1536/384 screening subset, not the
full 0.6497 baseline.
