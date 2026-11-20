# Coarse-missed narrowband: intra-band peak/centroid diagnostic

Read-only. Frozen B (coarse-only) and C (all-view) predictions; no
training, no threshold changes. band = GT frequency extent in the native
0.7 ms STFT (time span = GT time). PAPR = peak/mean inside the band;
centroid_offset = energy centroid offset from band centre, / (half band width).

## Groups

| group | n | PAPR med | PAPR p10/p90 | |centroid| med/p90 | peak med | mean med | band bins med |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| both | 1437 | 56.05 | 17.65/255.72 | 0.01/0.12 | 0.001 | 0.000 | 140 |
| c_only | 542 | 65.25 | 12.75/307.42 | 0.06/0.19 | 0.001 | 0.000 | 36 |
| b_only | 4 | 129.97 | 15.25/454.54 | 0.05/0.43 | 0.006 | 0.000 | 32 |
| neither | 177 | 27.09 | 6.05/115.50 | 0.05/0.10 | 0.000 | 0.000 | 84 |
| background | 2289 | 18.75 | 11.31/206.60 | 0.01/0.10 | 0.000 | 0.000 | 840 |
| fp_narrow_B | 2461 | 33.03 | 9.79/188.56 | 0.02/0.21 | 0.000 | 0.000 | 74 |

## Counts by B x T

| B x T | narrow GT | both | C-only | B-only | neither |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20x100 | 57 | 47 | 7 | 0 | 3 |
| 20x150 | 53 | 35 | 11 | 0 | 7 |
| 20x20 | 121 | 96 | 16 | 0 | 9 |
| 20x40 | 68 | 48 | 14 | 0 | 6 |
| 20x60 | 68 | 55 | 8 | 0 | 5 |
| 20x80 | 60 | 41 | 11 | 1 | 7 |
| 30x100 | 56 | 41 | 13 | 0 | 2 |
| 30x150 | 54 | 42 | 11 | 0 | 1 |
| 30x20 | 99 | 70 | 18 | 0 | 11 |
| 30x40 | 59 | 39 | 17 | 0 | 3 |
| 30x60 | 66 | 47 | 14 | 0 | 5 |
| 30x80 | 54 | 37 | 9 | 1 | 7 |
| 40x100 | 69 | 46 | 15 | 0 | 8 |
| 40x150 | 59 | 32 | 21 | 0 | 6 |
| 40x20 | 79 | 58 | 18 | 0 | 3 |
| 40x40 | 69 | 42 | 19 | 0 | 8 |
| 40x60 | 66 | 44 | 16 | 1 | 5 |
| 40x80 | 71 | 44 | 19 | 0 | 8 |
| 50x100 | 62 | 34 | 23 | 0 | 5 |
| 50x150 | 75 | 48 | 21 | 0 | 6 |
| 50x20 | 61 | 39 | 18 | 0 | 4 |
| 50x40 | 62 | 40 | 20 | 0 | 2 |
| 50x60 | 43 | 27 | 11 | 0 | 5 |
| 50x80 | 62 | 40 | 18 | 0 | 4 |
| 5x20 | 197 | 171 | 9 | 1 | 16 |
| 80x100 | 59 | 24 | 29 | 0 | 6 |
| 80x150 | 66 | 28 | 30 | 0 | 8 |
| 80x20 | 60 | 32 | 21 | 0 | 7 |
| 80x40 | 62 | 19 | 39 | 0 | 4 |
| 80x60 | 55 | 30 | 20 | 0 | 5 |
| 80x80 | 68 | 41 | 26 | 0 | 1 |

Raw per-instance records are in the JSON.


## AUC / per-class and per-B breakdown (added post-run)

Separation quality (AUC; 0.5 = no separation, 1 = perfect):

| target vs control | peak AUC | PAPR AUC | |centroid| AUC |
| --- | ---: | ---: | ---: |
| c_only vs background | **0.995** | 0.704 | 0.668 |
| c_only vs B narrow false positives | **0.965** | n/a | 0.611 |

Per class:

| class | n | c_only | both | c_only PAPR med | both PAPR med | c_only |centroid| med | both |centroid| med | peak AUC vs bg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 1,384 | 343 | 1,041 | 72.0 | 53.7 | 0.060 | 0.011 | 0.998 |
| 12 | 334 | 103 | 231 | **10.6** | 40.2 | 0.050 | 0.008 | 0.983 |
| 13 | 261 | 96 | 165 | 77.9 | 198.5 | 0.100 | 0.007 | 0.998 |

Per observation bandwidth:

| B | c_only | bg PAPR med | c_only PAPR med | PAPR AUC vs bg | peak AUC vs bg | peak AUC vs FP_B | |centroid| AUC vs bg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 9 | 12.3 | 33.1 | 0.699 | 1.000 | 0.998 | 0.761 |
| 20 | 67 | 15.4 | 69.0 | 0.765 | 1.000 | 0.989 | 0.843 |
| 30 | 82 | 16.8 | 55.6 | 0.735 | 1.000 | 0.990 | 0.829 |
| 40 | 108 | 22.0 | 68.8 | 0.684 | 1.000 | 0.976 | 0.827 |
| 50 | 111 | 26.1 | 65.8 | 0.651 | 0.999 | 0.974 | 0.786 |
| 80 | 165 | **65.7** | 68.5 | **0.502** | **0.995** | **0.964** | 0.618 |

Interpretation:

1. **Peak, not PAPR, is the signal.** `c_only` peak separates almost perfectly
   from background (AUC 0.995) and from B's narrow false positives
   (AUC 0.965). PAPR only separates moderately (0.704) and collapses at 80 MHz
   (AUC 0.502) because the background there also has large peak/mean ratios.
   PAPR is therefore partly a noise/number-of-bins effect, as suspected.
2. **Class 12 is the main PAPR exception**: its c_only PAPR median (10.6) is
   *lower* than both (40.2) and its PAPR AUC vs background is 0.285. Peak
   still separates at 0.983, so the useful statistic is again the absolute
   peak, not peak/mean.
3. **Centroid offset is a genuine second signal.** c_only |centroid| median
   0.057 vs both 0.010 and background 0.012; per-B AUC 0.62–0.84 (lowest at
   80 MHz). This indicates that coarse-missed narrowband instances are more
   often off the energy centroid/row centre than the ones coarse already
   finds; it is not just noise, but it is weaker than peak and also degrades
   at 80 MHz.
4. **The effect is strongest where the slicing loss is largest (80 MHz)**:
   165/542 c_only instances (30%) are at 80 MHz, and coarse-only recall for
   narrowband is lowest there. But this is also where PAPR is useless
   (background PAPR 65.7) and centroid AUC drops to 0.62; the recoverable
   statistic at 80 MHz is the **absolute peak** (AUC 0.995 vs background,
   0.964 vs B false positives).
5. **Limitations.**
   - Read-only: it shows the information is present in the same STFT, not
     that a detector can use it or that it converts to AP.
   - The c_only set is matched by C's current inference; some of those
     instances may be missed for score-ranking/conf reasons rather than pure
     visibility.
   - Background bands are fixed-width strips, not perfectly matched to each
     instance's bandwidth; this is a conservative control.
   - No threshold sweep or false-positive budget was applied; AUCs are
     rank-based only.

## Verdict

**Peak is a strong candidate; PAPR is not. Centroid is a plausible secondary
statistic.**

- The area-weighted coarse average does discard the intra-band peak, and the
  c_only narrowband instances have a peak that is almost perfectly separable
  from same-B/T background and from B's own narrow false positives.
- PAPR does not add this separation and is confounded with number of bins /
  noise, especially at 80 MHz.
- Centroid offset carries real frequency-shift information (c_only median
  0.057 vs both 0.010), but it is weaker and degrades at 80 MHz.

**Recommendation: one paired training arm is justified, but with narrow
scope.** Add two cheap 2048x512 channels derived from the same 0.7 ms STFT:

1. `band_peak`: per coarse row / time column, the max power inside the row's
   native frequency support (or its log), replacing the area mean.
2. `band_centroid`: per coarse row / time column, the energy-weighted
   frequency offset from the row centre (normalized), 0 at centre.

Do not add a PAPR channel: it is largely the same information as peak but
more noise-sensitive. Do not add complex/short-window channels; the dense C3
ablation is closed. Keep C1 unchanged and compare a frozen-budget paired arm
(power + peak + centroid) against C1 on the same 1536/384 subset with a
pre-registered last-checkpoint rule. If the arm cannot recover a meaningful
part of the c_only narrowband set or reduces wide-class AP, close this
hypothesis too.
