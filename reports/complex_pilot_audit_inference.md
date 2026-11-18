# Complex pilot audit and existing-weight inference

Scope: code/cache audit plus inference with the existing C1/C3 50-epoch
weights. No new training, no full val, test sealed. System reference remains
0.649725; absolute pilot mAP is not comparable to it because the val subset is
128 samples and training is 50 epochs on 256 sources.

## 1. Bottom line

- Cache, power/complex channels, fine starts, boxes and 90-degree rotation are
  constructed correctly; C1/C3 power channels are bitwise identical.
- C3's Re/Im stem channels are not ignored: trained norms are 0.46/0.43 and
  0.49/0.52 versus 4.79 for the power channel, with non-zero input gradients.
- On all native fine slices of the fixed val subset, C3 is **flat overall**
  (+0.02 pp mAP without filter, +0.19 pp with fine-0-5 filter), while
  AP_agnostic and localization recall are **lower** (-1.29 / -1.03 pp).
- C3's 3/4/5 AP is higher by +2.21 pp and narrow 9/12/13 by +3.39 pp, but
  0/1/2 is -0.90 pp and 10/11 is -1.10 pp. The fine-0-5 filter improves both
  arms by ~1.7-1.9 pp and does not change the comparison.
- Diagnostics show the complex channels' measurable contribution is
  **narrowband/10-11**, not 0/1/2 or 3/4/5: zeroing all Re/Im reduces mAP by
  2.94 pp, 10/11 by 3.10 pp and narrow by 9.71 pp, but **improves 0/1/2 by
  0.91 pp** and leaves 3/4/5 essentially unchanged (-0.14 pp).
- The C3 3/4/5 gain over C1 is therefore **not causally explained by complex
  information**; it is more consistent with extra-channel capacity/training
  dynamics. A full split-path/full-val program is not justified for the wide
  classes. The only worthwhile narrowband-oriented control is C3-fine-only.

## 2. Cache and channel audit

Four real fit IQ samples were recomputed from raw IQ with the same functions:

| sample | B/T | coarse bitwise | fine bitwise | fine power vs `stft_power_fast` | box max diff | labels |
| --- | --- | --- | --- | ---: | ---: | --- |
| train/1305 | 20/20 | yes | yes | 2.44e-4 | 0.0 | equal |
| train/1205 | 40/40 | yes | yes | 2.44e-4 | 0.0 | equal |
| train/5435 | 20/150 | yes | yes | 2.44e-4 | 0.0 | equal |
| train/5953 | 80/150 | yes | yes | 2.44e-4 | 0.0 | equal |

The 2.44e-4 is float16 cache quantization; float32 recomputation matches the
existing STFT power path. Rotating the IQ by j and rotating the cached Re/Im
by 90 degrees agree exactly (max diff 0), power unchanged; the C1 rotation
control gives exactly identical predictions.

C3 first-conv audit:

| checkpoint | ch0 (power) norm | ch1 (Re) norm | ch2 (Im) norm |
| --- | ---: | ---: | ---: |
| C3 init | 4.787 | 0.000 | 0.000 |
| C3 seed 1729 | 4.787 | 0.460 | 0.429 |
| C3 seed 1730 | 4.784 | 0.490 | 0.519 |

Input-gradient RMS on a fit batch:

| checkpoint | power | Re | Im |
| --- | ---: | ---: | ---: |
| C1 seed 1729 | 0.00460 | - | - |
| C3 seed 1729 | 0.00523 | 0.000765 | 0.001324 |
| C3 seed 1730 | 0.00322 | 0.000784 | 0.000561 |

The complex channels update and receive gradients; the negative result is not
"unused zero channels".

## 3. Full fine-slice validation

All native fine slices of the 128-sample val subset; standard mapping,
internal-boundary drop, class-wise global NMS, max 100. Mean over two seeds:

| condition | mAP50 | AP_agn | loc recall | class acc | 0/1/2 | 3/4/5 | 10/11 | narrow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 no filter | 0.3030 | 0.5865 | 0.8700 | 0.6498 | 0.1424 | 0.1098 | 0.1750 | 0.3894 |
| C3 no filter | 0.3032 | 0.5737 | 0.8597 | 0.6528 | 0.1335 | 0.1319 | 0.1640 | 0.4234 |
| **Δ no filter** | **+0.02 pp** | **−1.29 pp** | **−1.03 pp** | +0.30 pp | −0.90 pp | +2.21 pp | −1.10 pp | +3.39 pp |
| C1 + fine 0-5 filter | 0.3199 | 0.5904 | 0.8755 | 0.6489 | 0.1819 | 0.1490 | 0.1749 | 0.3899 |
| C3 + fine 0-5 filter | 0.3218 | 0.5790 | 0.8624 | 0.6538 | 0.1797 | 0.1707 | 0.1653 | 0.4246 |
| **Δ + filter** | **+0.19 pp** | **−1.14 pp** | **−1.31 pp** | +0.50 pp | −0.22 pp | +2.17 pp | −0.95 pp | +3.46 pp |

Per-seed mAP (no filter / filter):

| seed | C1 | C3 |
| ---: | ---: | ---: |
| 1729 | 0.2962 / 0.3166 | 0.2902 / 0.3047 |
| 1730 | 0.3098 / 0.3232 | 0.3162 / 0.3390 |

Group class-agnostic recall (no filter, mean over seeds):

| group | C1 | C3 |
| --- | ---: | ---: |
| 0/1/2 | 0.8535 | 0.8384 |
| 3/4/5 | 0.8621 | 0.8966 |
| 10/11 | 0.8681 | 0.8333 |
| narrow 9/12/13 | 0.7602 | 0.7474 |

C3 improves 3/4/5 AP and recall, but 0/1/2, 10/11 and narrow recall are lower.
The narrow AP gain (+3.39 pp) does not translate to higher narrow recall
(-1.28 pp), so "preserve narrowband detection" is not satisfied by C3 full.

The fine-0-5 filter raises both arms by +1.7-1.9 pp; it is a separate gain, not
a C3 effect. C3 remains flat overall and the group pattern is unchanged.

## 4. C3 inference diagnostics

All native fine slices; mean over the two C3 seeds; no fine-class filter.

| mode | mAP50 | 0/1/2 | 3/4/5 | 10/11 | narrow | loc recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 0.3032 | 0.1335 | 0.1319 | 0.1640 | 0.4234 | 0.8597 |
| all Re/Im zero | **0.2738** | **0.1426** | 0.1305 | 0.1330 | **0.3263** | 0.8549 |
| coarse Re/Im zero | 0.2983 | 0.1425 | 0.1370 | 0.1444 | 0.4160 | 0.8590 |
| fine Re/Im zero | 0.2814 | 0.1316 | 0.1316 | 0.1472 | **0.3406** | 0.8535 |
| coarse view zero | 0.1537 | 0.0000 | 0.0000 | 0.0165 | 0.3971 | 0.5186 |
| fine view zero | 0.2715 | **0.1751** | **0.1651** | **0.2141** | 0.1324 | 0.7607 |
| global 90-degree rotation | 0.3109 | 0.1387 | 0.1294 | 0.1810 | 0.4345 | 0.8624 |

Per-seed normal → all-Re/Im-zero:

- seed 1729: 0.2902 → 0.2650; 0/1/2 0.1123 → 0.1120; 3/4/5 0.1206 →
  0.1193; narrow 0.3950 → 0.3245.
- seed 1730: 0.3162 → 0.2826; 0/1/2 0.1546 → **0.1731**; 3/4/5 0.1433 →
  0.1417; narrow 0.4517 → 0.3281.

Interpretation:

- **Coarse view is essential for wide classes**: zeroing it sends 0/1/2 and
  3/4/5 AP to 0; fine-only keeps narrow at 0.397 but cannot recover wide boxes.
- **Fine view is essential for narrow classes**: zeroing it drops narrow AP
  from ~0.42 to 0.13, while wide AP actually rises (fine-view wide predictions
  compete with coarse).
- **All complex zero improves 0/1/2 (+0.91 pp) and leaves 3/4/5 flat**, while
  dropping 10/11 and narrow. Complex info is a narrowband/10-11 feature, not a
  wide-modulation feature.
- Coarse complex zero improves 0/1/2 (+0.90) and 3/4/5 (+0.51), but lowers
  10/11 and narrow slightly; coarse complex is neutral-to-negative for wide.
- Fine complex zero leaves 0/1/2/3/4/5 flat but drops narrow by 8.28 pp and
  10/11 by 1.68 pp; fine complex is the main useful complex path, for narrow.
- 90-degree global rotation changes C3 only modestly (mAP +0.77 pp; wide ~flat;
  narrow +1.11 pp); C1 is exactly invariant (mAP 0.2962 both). The C3 model is
  not strongly phase-locked, but not phase-invariant either.

The C3 3/4/5 full-fine gain over C1 is not reproduced by removing complex
channels; it persists when either or both complex paths are zeroed at inference.
That points to extra-channel capacity/training dynamics rather than complex
information. A trained C3-zero-complex control would be needed to prove this
causally, but the inference evidence already argues against attributing the
wide gain to Re/Im.

Cost: on the same all-fine val cache, C1 takes 15.9/14.6 s per seed (mean
15.25 s) and C3 23.6/23.7 s (mean 23.65 s), about **+55%** for full-fine
inference. Training was +11.9% and peak memory +2.1% in the previous pilot.

## 5. Convergence check

50-epoch validation events (mAP50):

| epoch | C1 1729 | C3 1729 | C1 1730 | C3 1730 |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 0.0792 | 0.0748 | 0.0475 | 0.0661 |
| 19 | 0.1925 | 0.1718 | 0.1367 | 0.1912 |
| 29 | 0.2336 | 0.2430 | 0.2570 | 0.2777 |
| 39 | 0.2813 | 0.2715 | **0.3024** | 0.3022 |
| 49 | 0.2905 | 0.2776 | 0.2969 | **0.3048** |

Both arms are still moving at 50 epochs. C1 seed 1730 peaks at epoch 39
(0.3024) and drops to 0.2969; C3 seed 1730 is still rising. The pilot saved only
`last.pt`, not `best.pt`, so the full-fine evaluation uses final weights only.
Using best-per-seed would change the C1 mean from 0.2937 to 0.2965, while C3
stays 0.2912; the "no stable C3 gain" conclusion remains, but the exact delta is
checkpoint-dependent. A fixed stopping rule and best checkpoint are required in
the next run.

## 6. Implementation biases and what would change the judgment

- **Fixed fine view in training vs all fine in this audit**: the full-fine C3
  is 1.1-1.3 pp above the fixed-view pilot and C1 is 0.6-1.3 pp above; the
  fixed-view pilot understated both, C3 slightly more. The full-fine result is
  the better estimate.
- **Final-only checkpoint**: C1 seed 1730 peaked before 50; no best checkpoint
  was saved. If a best checkpoint were used, C1 would gain ~0.3 pp. The
  wide-group sign flips are robust across epochs, so the main conclusion holds.
- **128 val samples**: 0/1/2 has ~100 GT and 3/4/5 ~48 GT per seed; group AP is
  noisy. The 0/1/2 seed swing remains larger than the C3 effect.
- **Fine 0-5 filter**: improves both arms by ~1.7-1.9 pp; ignoring it lowers
  absolute mAP but does not create a C3 advantage.
- **Masking is an OOD intervention**: zeroing channels or views can change
  activations in ways not equal to "removing information"; use it as a
  dependence diagnostic, not a causal proof.
- **Capacity confound**: C3 has extra input channels and different training
  dynamics. The 3/4/5 gain surviving complex zeroing is consistent with a
  capacity/optimization effect. A trained C3-zero-complex arm is the clean
  control.
- **Absolute mAP**: 0.30-0.32 on 128 val / 50 epochs is not comparable to
  0.649725; only paired differences are meaningful.

## 7. Judgment

**For 0/1/2 and 3/4/5, no evidence that complex information after coarse
aggregation/fine slicing improves end-to-end recognition.** The full-fine
3/4/5 gain is not causally attributable to Re/Im; complex channels mainly help
narrowband/10-11. The wide classes are better served by the power/coarse path.

**Do not start a full split-path/full-val training for the wide classes.** The
only defensible next training control is a **short C3-fine-only arm**
(coarse power only; fine power+Re/Im) with a best checkpoint, 2-3 seeds and a
larger val subset (256-384). It tests the only demonstrated complex benefit
(narrowband) while removing coarse complex that is neutral/negative for wide.
If the project wants to know whether the 3/4/5 gain is purely capacity, add a
**C3-zero-complex** control (3-channel model with Re/Im fixed zero during
training); that is the clean causal test. If neither shows a stable wide gain,
keep C1 and invest in detector capacity/phase-invariant features instead.

Artifacts: `reports/complex_pilot_audit_summary.json`,
`reports/complex_pilot_full_fine_eval.json`,
`reports/complex_pilot_diagnostics.json`,
`reports/complex_pilot_audit.json`,
`scripts/audit_complex_pilot.py`,
`scripts/eval_complex_pilot_full.py`,
`scripts/diagnose_complex_pilot.py`.
