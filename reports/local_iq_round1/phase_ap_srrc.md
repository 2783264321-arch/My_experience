# Phase A' Track S — SRRC single-carrier audit (classes 10/11, bw 10 MHz)

Generated: 2026-11-21 03:33:51  |  runtime: 19 s  |  pipeline: `scripts/baseline3_pa_srrc.py`  |  data: `reports/local_iq_round1/phase_ap_frames.json` (fit split only)

## Method (fixed, label-blind)

- Extraction recipe (fixed): central window `min(10 ms, dur)` of the GT box → `read_segment` → `demod_to_center` (box center) → `band_limit_fft(bw*1.02, edge_frac=0.05)` → trim 0.5 ms at both ends. Frequency convention as audited: FFT bin 0 = observation center `(f_lo + B/2)`; absolute f = `f_lo + B/2 + digital`.

- Synthetic generation (Step A): `gen_srrc_clean` — the same standard Proakis RRC taps (`srrc_pulse_taps`, unmodified) built at 32 samples/symbol with a 16-symbol span, resampled to the capture rate with `resample_poly` at exact rational ratios (3/40, 3/20, 3/10 for alpha=0.2; 27/320, 27/160, 27/80 for alpha=0.35 — no resampler rate error; the 10 MHz band is far inside every output Nyquist). 8-sps/span-8 taps (the shared `gen_srrc` setting) droop the pulse power spectrum up to ~6.5% through the rolloff (measured |H|² vs ideal RC) and would bias the width calibration; at 32 sps/span 16 the pulse fidelity is ~0.6%. Embedded at digital offsets 0 / +6 / −12 MHz for B = 20 / 40 / 80 MHz (band always inside the observation), white capture noise at 20 dB in-box SNR, round-tripped through float16 so the real read path is exercised.

- PSD widths: two-sided Welch PSD (~40 kHz grid, 75%-overlap Hann, ~500 kHz median kernel, plateau reference = median of the top quartile, persistence/hysteresis crossings — the raw max and the naive contiguous-run rule are speckle-broken, as Step A debugging showed). W90/W50/W10 = widths at 90/50/10% of the plateau reference. Raised-cosine inversion (exact for an ideal RC PSD): `W90 = R(1−0.590334a)`, `W50 = R`, `W10 = R(1+0.590334a)`, used as `R_est = W50` (max-slope crossing, most level-robust) and `alpha = (W10−W90)/(1.180668·W50)` (level errors cancel in the difference). The exact-in-theory pair rule `R=(W10+W90)/2` and the task's literal rule (`R_est = W90`, `alpha = W10/W90 − 1`) are also recorded (see estimator note).

- Timing: matched filter → `tone_line_scan` of `|y|²` in `[0.9, 1.1]·R` → direct correlation `D(f) = Σ |y|² e^{−2πifn/fs}` refined on a ±1.5-bin grid (parabolic peak), then `t0 = −angle(D)/(2π)·fs/f mod fs/f`. Evaluating D at the refined frequency avoids the padded-DFT bin-center phase error `πδ(N−1)`; the residual frequency error is <2 Hz over ≤9 ms, i.e. <0.01 symbol of t0 bias.

- Symbols at `t0 + k·fs/R` via `cubic_sample`; `m4`, `m6` (magnitude moments), 8-bin `|s|/RMS` histogram on [0,3]. Reference values: QPSK m4/m6 = 1.00/1.00, 16QAM = 1.32/1.96, Rayleigh noise = 2.0/6.0.

- Class labels are carried for reporting only; all statistics use box geometry. Only `train/` frames are read; `advanced/test` is never touched; no models trained.

## Step A — synthetic calibration (label-free)

| alpha | order | B | R_true (MHz) | R_est (MHz) | err R % | alpha_est | err a % | tone str | peak/med | n_sym | m4 | m6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | QPSK | 20 | 8.333 | 8.333 | -0.00 | 0.222 | +11.17 | 0.2624 | 72.3 | 31657 | 1.021 | 1.064 |
| 0.2 | QPSK | 40 | 8.333 | 8.333 | -0.00 | 0.214 | +7.20 | 0.2668 | 104.0 | 31658 | 1.019 | 1.058 |
| 0.2 | QPSK | 80 | 8.333 | 8.333 | -0.00 | 0.230 | +15.14 | 0.1849 | 102.0 | 31659 | 1.018 | 1.053 |
| 0.2 | 16QAM | 20 | 8.333 | 8.333 | -0.00 | 0.218 | +9.18 | 0.2510 | 69.2 | 31657 | 1.334 | 2.024 |
| 0.2 | 16QAM | 40 | 8.333 | 8.333 | +0.00 | 0.222 | +11.17 | 0.2575 | 100.4 | 31659 | 1.331 | 2.010 |
| 0.2 | 16QAM | 80 | 8.333 | 8.333 | +0.00 | 0.226 | +13.15 | 0.1844 | 101.7 | 31659 | 1.334 | 2.021 |
| 0.35 | QPSK | 20 | 7.407 | 7.407 | +0.00 | 0.375 | +7.20 | 0.4578 | 135.8 | 32583 | 1.022 | 1.067 |
| 0.35 | QPSK | 40 | 7.407 | 7.407 | -0.00 | 0.389 | +11.02 | 0.3458 | 145.1 | 32584 | 1.015 | 1.047 |
| 0.35 | QPSK | 80 | 7.407 | 7.407 | -0.00 | 0.371 | +5.92 | 0.2431 | 144.2 | 32585 | 1.014 | 1.044 |
| 0.35 | 16QAM | 20 | 7.407 | 7.407 | +0.00 | 0.389 | +11.02 | 0.4450 | 132.0 | 32583 | 1.339 | 2.042 |
| 0.35 | 16QAM | 40 | 7.407 | 7.407 | -0.00 | 0.348 | -0.46 | 0.3188 | 133.8 | 32584 | 1.334 | 2.019 |
| 0.35 | 16QAM | 80 | 7.407 | 7.407 | +0.00 | 0.389 | +11.02 | 0.2282 | 135.4 | 32585 | 1.330 | 2.003 |

Mean |R_est error| = 0.00%, mean |alpha_est error| = 9.47%, m4 range QPSK = 1.014–1.022, m4 range 16QAM = 1.330–1.339.

**Estimator note (three width rules, measured errors).** For a raised-cosine PSD the 90%/50%/10% widths are `R(1−0.590334a)` / `R` / `R(1+0.590334a)` (cosine-rolloff algebra), so the literal task rule `R_est = W90, a = W10/W90 − 1` is biased even on a PERFECT noiseless SRRC. Measured over the calibration grid:

| rule | R error range | alpha error range |
|---|---|---|
| literal (`R=W90, a=W10/W90−1`) | -26.2%…-14.2% | +48%…+78% (shape bias) |
| pair (`R=(W10+W90)/2`, `a=(W10−W90)/(1.180668 R)`) | -3.2%…-0.3% | -0.1%…+18.1% |
| USED (`R=W50`, `a=(W10−W90)/(1.180668 W50)`) | -2.4%…+0.3% | +0.4%…+15.3% |

The used pairing is chosen for robustness: the −3 dB (50%) crossing sits at the maximum-slope point of the RC rolloff (least sensitive to reference-level error), and the alpha difference (W10−W90) cancels common level errors. On real frames all three variants are in the JSON (`psd.R_psd_mhz/alpha_psd` = used, `R_pair_mhz/alpha_pair`, `R_lit_mhz/alpha_lit`).

**Timing sensitivity (deliberate t0 offsets).** m4 at the recovered timing vs t0 deliberately offset by ±0.25 and −0.50 symbol:

| order | alpha | m4(t0) | m4(−0.25) | m4(+0.25) | m4(−0.50) |
|---|---|---|---|---|---|
| QPSK | 0.2 | 1.021 | 1.277 | 1.279 | 1.586 |
| QPSK | 0.2 | 1.019 | 1.284 | 1.286 | 1.609 |
| QPSK | 0.2 | 1.018 | 1.282 | 1.281 | 1.611 |
| 16QAM | 0.2 | 1.334 | 1.517 | 1.513 | 1.733 |
| 16QAM | 0.2 | 1.331 | 1.513 | 1.510 | 1.737 |
| 16QAM | 0.2 | 1.334 | 1.512 | 1.521 | 1.741 |
| QPSK | 0.35 | 1.022 | 1.252 | 1.252 | 1.563 |
| QPSK | 0.35 | 1.015 | 1.229 | 1.229 | 1.575 |
| QPSK | 0.35 | 1.014 | 1.235 | 1.235 | 1.574 |
| 16QAM | 0.35 | 1.339 | 1.496 | 1.495 | 1.701 |
| 16QAM | 0.35 | 1.334 | 1.491 | 1.491 | 1.721 |
| 16QAM | 0.35 | 1.330 | 1.477 | 1.481 | 1.708 |

Summary: mean m4 inflation at ±0.25 symbol = 0.484 (QPSK), 0.336 (16QAM); at −0.50 symbol = 0.568 (QPSK), 0.390 (16QAM). Residual timing error therefore shows up as m4 inflation above the constellation reference; the recovered t0 must be used for any m4-based reading.

**Negative controls (pure noise through the same chain).**

| B | R_est (failure mode) | W90 | W10 | alpha_psd | tone str | tone f | m4 | m6 |
|---|---|---|---|---|---|---|---|---|
| 20 | 9.883 MHz (flat PSD → R_psd ≈ full band) | 9.26 | 10.16 | 0.077 | 0.0084 | 8.90 | 2.016 | 6.108 |
| 40 | 9.844 MHz (flat PSD → R_psd ≈ full band) | 9.18 | 10.16 | 0.084 | 0.0098 | 8.89 | 1.985 | 5.847 |
| 80 | 9.883 MHz (flat PSD → R_psd ≈ full band) | 9.41 | 10.16 | 0.064 | 0.0070 | 8.91 | 2.012 | 6.143 |

R_est failure mode on flat noise: the persistence crossings return widths covering essentially the whole analysis band (B20: W90=9.3/W50=9.9/W10=10.2; B40: W90=9.2/W50=9.8/W10=10.2; B80: W90=9.4/W50=9.9/W10=10.2 MHz), so `R_psd = W50 ≈ 9.9 MHz` is the band width, not a symbol rate; `alpha_psd` ≈ 0.06–0.08 is spurious structure on a flat spectrum; the tone scan in [0.9, 1.1]·R_psd ≈ [8.9, 10.9] MHz finds only a noise maximum (strength ≤ 0.0098, raw peak/median ≤ 4.0, same order as the off-band control), and m4 ≈ 2.00 (Rayleigh). Note: the half-split coherence is NOT a discriminator — the line frequency is selected as the maximum of the full-record sum, which biases the two halves into apparent alignment for noise as well — and is recorded in the JSON for completeness only.

Tone-strength threshold for Step B (geometric mean of the worst noise control 0.0098 and the weakest synthetic line 0.1844): **0.0424** (synthetic range 0.1844–0.4578; raw peak/median ≥ 69.2 vs noise ≤ 4.0; off-band specificity: the synthetic in-band peak/median is 20–100x its off-band control, noise ≈ 1x).

## Step B — real frames (10 class-10/11 SRRC frames, same pipeline)

| id | class | stratum | R_est (MHz) | alpha_est | tone str | n_sym | m4 | m6 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| train/1037#1 | 10 | clean | 8.333 | 0.526 | 0.2394 | 74991 | 1.099 | 1.313 | SRRC-like (qpsk_like) |
| train/1058#1 | 10 | clean | 8.333 | 0.262 | 0.2970 | 74991 | 1.012 | 1.035 | SRRC-like (qpsk_like) |
| train/1099#0 | 10 | clean | 8.333 | 0.592 | 0.1842 | 74992 | 1.136 | 1.436 | SRRC-like (qpsk_like) |
| train/1160#1 | 10 | clean | 8.430 | 0.165 | 0.0073 | 75859 | 1.855 | 4.801 | NOT SRRC-like (rayleigh_noise_like) |
| train/1093#0 | 11 | clean | 8.333 | 0.377 | 0.2408 | 74990 | 1.787 | 4.187 | SRRC-like (rayleigh_noise_like) |
| train/123#1 | 11 | clean | 3.958 | 0.980 | 0.0045 | 35617 | 1.858 | 4.833 | NOT SRRC-like (rayleigh_noise_like) |
| train/1471#0 | 11 | clean | 8.333 | 0.640 | 0.2241 | 74992 | 1.390 | 2.272 | SRRC-like (qam16_like) |
| train/1986#0 | 11 | clean | 8.333 | 0.667 | 0.1943 | 74992 | 1.563 | 3.137 | NOT SRRC-like (other) |
| train/1011#0 | 10 | overlap | 8.841 | 0.715 | 0.0077 | 79558 | 1.748 | 4.228 | NOT SRRC-like (other) |
| train/1004#1 | 11 | overlap | 4.824 | 0.980 | 0.0091 | 43403 | 2.068 | 7.490 | NOT SRRC-like (rayleigh_noise_like) |

PSD shape detail (the −3 dB width should equal R_est for an RC band; edge symmetry should be ≈0 for a centered SRRC):

| id | W50 (−3 dB) | W10 (−10 dB) | W90 | center off | asym@10% | edge drop | plateau flat | t0/T | peak/med | spec | SNR dB | n_runs10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train/1037#1 | 7.76 | 9.42 | 4.25 | 0.02 | 0.000 | 16.8 | 0.1 | 0.303 | 160.6 | 41.7 | 13.2 | 1 |
| train/1058#1 | 8.32 | 9.45 | 6.88 | -0.02 | 0.004 | 18.3 | 0.1 | 0.258 | 178.2 | 152.9 | 22.9 | 1 |
| train/1099#0 | 8.05 | 9.49 | 3.67 | 0.00 | -0.004 | 15.1 | 0.1 | 0.698 | 156.3 | 44.7 | 10.5 | 1 |
| train/1160#1 | 9.34 | 10.00 | 8.36 | 0.02 | -0.004 | n/a | 0.1 | 0.470 | 3.1 | 0.8 | 11.8 | 1 |
| train/1093#0 | 8.09 | 9.45 | 5.74 | -0.10 | 0.021 | 17.6 | 0.2 | 0.138 | 102.2 | 31.0 | 18.3 | 1 |
| train/123#1 | 4.14 | 8.87 | 1.88 | -2.27 | 0.427 | 12.7 | 0.1 | 0.016 | 3.8 | 1.0 | 10.4 | 1 |
| train/1471#0 | 8.01 | 9.67 | 3.37 | 0.00 | 0.020 | 18.3 | 0.2 | 0.823 | 150.3 | 38.9 | 14.0 | 1 |
| train/1986#0 | 8.05 | 9.80 | 3.24 | 0.23 | -0.044 | n/a | 0.1 | 0.366 | 116.6 | 29.0 | 7.2 | 1 |
| train/1011#0 | 9.80 | 10.16 | 2.70 | -0.10 | 0.019 | n/a | 0.4 | 0.943 | 3.3 | 0.8 | 3.2 | 1 |
| train/1004#1 | 5.35 | 9.18 | 1.72 | -1.04 | 0.213 | 18.3 | 0.1 | 0.118 | 3.9 | 1.0 | 23.9 | 1 |

Per-frame notes:

- `train/1037#1` (class 10, clean): SRRC-like.
- `train/1058#1` (class 10, clean): SRRC-like.
- `train/1099#0` (class 10, clean): SRRC-like.
- `train/1160#1` (class 10, clean): flat-topped band with no visible edge rolloff (10% width fills the box; edge drop unmeasurable); tone below threshold (strength=0.007297); flat band + near-Rayleigh envelope + no tone: noise-like or OFDM-style content, not single-carrier SRRC; noise-like envelope (m4~2).
- `train/1093#0` (class 11, clean): SRRC-like; note: sync gates pass but envelope is rayleigh_noise_like (m4=1.79) -- noisy or non-QAM constellation.
- `train/123#1` (class 11, clean): no measurable RC transition (alpha_psd=0.980); tone below threshold (strength=0.004491); band off-center by -2.27 MHz; noise-like envelope (m4~2).
- `train/1471#0` (class 11, clean): SRRC-like.
- `train/1986#0` (class 11, clean): flat-topped band with no visible edge rolloff (10% width fills the box; edge drop unmeasurable); symbol-rate tone present (8.333 MHz, strength=0.194) but PSD gates fail: RC-shape gate (edge drop unmeasurable).
- `train/1011#0` (class 10, overlap): flat-topped band with no visible edge rolloff (10% width fills the box; edge drop unmeasurable); tone below threshold (strength=0.007685); flat band + near-Rayleigh envelope + no tone: noise-like or OFDM-style content, not single-carrier SRRC; overlap stratum: box may contain a second signal.
- `train/1004#1` (class 11, overlap): no measurable RC transition (alpha_psd=0.980); tone below threshold (strength=0.009091); band off-center by -1.04 MHz; noise-like envelope (m4~2); overlap stratum: box may contain a second signal.

## Step C — exploratory class-10 vs class-11 comparison (no classification claims)

| group | m4 values | mean | nominal reference |
|---|---|---|---|
| class-10 clean (n=4) | [1.099, 1.012, 1.136, 1.855] | 1.276 | QPSK 1.00 |
| class-11 clean (n=4) | [1.787, 1.858, 1.39, 1.563] | 1.649 | 16QAM 1.32 |
| class-10 clean, SRRC-like only (n=3) | [1.099, 1.012, 1.136] | 1.082 | QPSK 1.00 |
| class-11 clean, SRRC-like only (n=2) | [1.787, 1.39] | 1.588 | 16QAM 1.32 |
| synthetic QPSK (calibration grid, n=6) | [1.021, 1.019, 1.018, 1.022, 1.015, 1.014] | 1.018 | 1.00 |
| synthetic 16QAM (calibration grid, n=6) | [1.334, 1.331, 1.334, 1.339, 1.334, 1.33] | 1.334 | 1.32 |
| noise controls | [2.016, 1.985, 2.012] | 2.004 | 2.0 (Rayleigh) |

**Trend statement:** class-11 clean mean m4 exceeds class-10 (1.65 vs 1.28), matching the QPSK<16QAM ordering, but the absolute values do not separately sit at the references (SRRC-like-only means: class-10 1.08 vs QPSK 1.00 [range 1.01-1.14], class-11 1.59 vs 16QAM 1.32 [range 1.39-1.79]); the class-10 SRRC-like frames cluster near the QPSK reference, while class-11 frames land above the 16QAM reference, consistent with low-SNR pull toward the Rayleigh value 2.0 and/or a non-QAM constellation. Partially consistent at most; exploratory only, no classification claim.

## Facts vs inferences

**Facts (measured / verified here):**

1. The extraction recipe + pipeline recovers synthetic SRRC truth: R within 0.0% (tone-locked, 0.00% on every grid point) and alpha within 15.1% over all 12 grid points; m4 at recovered timing = 1.014–1.022 (QPSK) and 1.330–1.339 (16QAM) vs references 1.00 / 1.32.

2. The literal 90/10 width rule is biased by the RC rolloff shape even on a perfect SRRC (measured +47.5%…+77.6% on alpha); the USED W50/(W10−W90) pairing is what the tables use, and all three variants are recorded per run (see the estimator note).

3. Timing is the fragile part: a deliberate ±0.25-symbol t0 error inflates QPSK m4 to ~1.26 and 16QAM m4 to ~1.50 (mean over the grid); the phase-of-line t0 is what keeps m4 at the reference values.

4. Pure-noise negative controls: R_psd ≈ full band (no symbol rate), tone strength ≤ 0.0098, m4 ≈ 2.00 (Rayleigh).

5. Per-frame measured values for the 10 frames are in the tables above (5 of 10 frames pass the SRRC-like gates: symbol-rate tone above threshold + RC-shaped PSD + W50≈R).

5a. 6 frames (`train/1037#1`, `train/1058#1`, `train/1099#0`, `train/1093#0`, `train/1471#0`, `train/1986#0`) carry a symbol-rate line at 8.333 MHz = 10/1.2 MHz: for a 10 MHz box this pins the box-implied rolloff at alpha = 0.2.

5b. On those frames the 10%-edge widths W10 = 9.42–9.80 MHz (ideal RC at R=8.333, alpha=0.2: 9.32 MHz — edges are RC-consistent), but the 90% plateau widths W90 = 3.24–6.88 MHz fall short of the ideal 7.35 MHz: the measured passbands carry ≥10% ripple/tilt. The width-based alpha_est (0.26–0.67) therefore over-reads the edge-implied 0.2 and is reported with that caveat.

**Inferences (interpretation, weaker than the facts):**

6. Any statement that a given frame *is* an SRRC single carrier rests on the sync-quality gates (tone + RC shape + W50 consistency), not on class labels; m4 alone cannot separate constant-envelope modulations (a pure tone also has m4≈1), so the PSD/tone checks carry the discrimination.

7. class-11 clean mean m4 exceeds class-10 (1.65 vs 1.28), matching the QPSK<16QAM ordering, but the absolute values do not separately sit at the references (SRRC-like-only means: class-10 1.08 vs QPSK 1.00 [range 1.01-1.14], class-11 1.59 vs 16QAM 1.32 [range 1.39-1.79]); the class-10 SRRC-like frames cluster near the QPSK reference, while class-11 frames land above the 16QAM reference, consistent with low-SNR pull toward the Rayleigh value 2.0 and/or a non-QAM constellation. Partially consistent at most; exploratory only, no classification claim.

8. Frames failing the gates split into two label-blind content patterns: flat-topped boxes with no tone above threshold and near-Rayleigh envelopes (`train/1160#1`, `train/1011#0`) — consistent with noise-like or OFDM-style wideband content rather than a single-carrier SRRC — and boxes whose content sits off-center with no coherent tone (`train/123#1`, `train/1004#1`) — mostly empty or interfered boxes. The overlap-stratum rows may additionally contain a second signal inside the GT box; these are content descriptions, not label claims.

9. Nothing here trains or evaluates a classifier; no classification gain is claimed or implied.


Runtime: 19 s total (single process, no training, fit-split frames only).
