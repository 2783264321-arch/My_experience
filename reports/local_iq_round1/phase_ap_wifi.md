# Baseline-3 phase A' · Track W — WiFi CP-OFDM structure audit (30 frames)

2026-11-21 · OpenCode. Pre-registered order: synthetic calibration first (label-free); the threshold derived there is the decision rule applied to the real frames; class labels appear only in summary/interpretation columns. Fit split only (every frame is `train/*`); the test split was not read.

Artifacts: `reports/local_iq_round1/phase_ap_wifi.json` (all numbers), this file. Script: `scripts/baseline3_pa_wifi.py`.

Fixed extraction recipe: central window min(10 ms, box duration) → `read_segment` (complex64) → `demod_to_center` to GT-box center (audited convention: FFT bin 0 = observation center f_lo + B/2) → `band_limit_fft(keep = bw·1.02, edge_frac = 0.05)` → trim 0.5 ms both ends → `cp_scan_complex` over the full pre-registered grid (T_u ∈ {3.2, 6.4, 12.8, 25.6, 51.2} µs × CP ∈ {0.2…6.4} µs).


## 1. STEP A — synthetic calibration and the pre-registered threshold

QPSK CP-OFDM generated at the capture rate (`gen_wifi_ofdm(fs=B·1e6)`, 3000 symbols, unit power, SCS = 1/T_u), up-converted to a mid-band offset that keeps the 20/40 MHz extraction box fully inside the observation (offset 0 when bw = B), demod + band-limited with the recipe, then in-band noise added at {∞, 10, 0, −5} dB. Negatives: pure complex noise, same window/band-limit/trim.


### B = 20 MHz

| bw | off | true T_u/CP | SNR dB | best (T_u,CP) | metric | ratio | best-other-T_u | cfo kHz | match |
|---:|---:|---|---:|---|---:|---:|---:|---:|:--:|
| 20 | +0 | 12.8/0.8 | inf | (12.8,0.8) | 1.003 | 102 | 0.092@51.2 | -0.0 | yes |
| 20 | +0 | 12.8/0.8 | 10 | (12.8,0.8) | 0.913 | 98 | 0.101@51.2 | -0.1 | yes |
| 20 | +0 | 12.8/0.8 | 0 | (12.8,0.8) | 0.495 | 56 | 0.096@51.2 | +0.2 | yes |
| 20 | +0 | 12.8/0.8 | -5 | (12.8,0.8) | 0.227 | 28 | 0.098@51.2 | -0.1 | yes |
| 20 | +0 | 3.2/0.8 | inf | (3.2,0.8) | 0.998 | 229 | 0.087@51.2 | +0.0 | yes |
| 20 | +0 | 3.2/0.8 | 10 | (3.2,0.8) | 0.907 | 185 | 0.094@51.2 | -0.1 | yes |
| 20 | +0 | 3.2/0.8 | 0 | (3.2,0.8) | 0.498 | 69 | 0.097@51.2 | -0.4 | yes |
| 20 | +0 | 3.2/0.8 | -5 | (3.2,0.8) | 0.246 | 47 | 0.105@51.2 | +0.4 | yes |
| 20 | +0 | 25.6/0.8 | inf | (25.6,0.8) | 1.018 | 78 | 0.116@51.2 | +0.0 | yes |
| 20 | +0 | 25.6/0.8 | 10 | (25.6,0.8) | 0.929 | 76 | 0.107@51.2 | +0.0 | yes |
| 20 | +0 | 25.6/0.8 | 0 | (25.6,0.8) | 0.511 | 36 | 0.099@51.2 | -0.1 | yes |
| 20 | +0 | 25.6/0.8 | -5 | (25.6,0.8) | 0.248 | 21 | 0.098@51.2 | +0.1 | yes |

### B = 40 MHz

| bw | off | true T_u/CP | SNR dB | best (T_u,CP) | metric | ratio | best-other-T_u | cfo kHz | match |
|---:|---:|---|---:|---|---:|---:|---:|---:|:--:|
| 20 | +5 | 12.8/0.8 | inf | (12.8,0.8) | 0.980 | 90 | 0.100@51.2 | +0.0 | yes |
| 20 | +5 | 12.8/0.8 | 10 | (12.8,0.8) | 0.889 | 88 | 0.096@51.2 | +0.0 | yes |
| 20 | +5 | 12.8/0.8 | 0 | (12.8,0.8) | 0.488 | 69 | 0.079@51.2 | +0.1 | yes |
| 20 | +5 | 12.8/0.8 | -5 | (12.8,0.8) | 0.239 | 58 | 0.071@51.2 | -0.4 | yes |
| 20 | +5 | 3.2/0.8 | inf | (3.2,0.8) | 0.975 | 157 | 0.114@51.2 | -0.0 | yes |
| 20 | +5 | 3.2/0.8 | 10 | (3.2,0.8) | 0.884 | 162 | 0.117@51.2 | -0.1 | yes |
| 20 | +5 | 3.2/0.8 | 0 | (3.2,0.8) | 0.487 | 110 | 0.074@51.2 | -0.2 | yes |
| 20 | +5 | 3.2/0.8 | -5 | (3.2,0.8) | 0.232 | 47 | 0.090@51.2 | -0.4 | yes |
| 20 | +5 | 25.6/0.8 | inf | (25.6,0.8) | 0.989 | 89 | 0.101@51.2 | -0.0 | yes |
| 20 | +5 | 25.6/0.8 | 10 | (25.6,0.8) | 0.900 | 87 | 0.112@51.2 | -0.0 | yes |
| 20 | +5 | 25.6/0.8 | 0 | (25.6,0.8) | 0.499 | 55 | 0.087@51.2 | +0.1 | yes |
| 20 | +5 | 25.6/0.8 | -5 | (25.6,0.8) | 0.250 | 33 | 0.081@51.2 | +0.1 | yes |
| 40 | +0 | 12.8/0.8 | inf | (12.8,0.8) | 0.982 | 136 | 0.068@51.2 | +0.0 | yes |
| 40 | +0 | 12.8/0.8 | 10 | (12.8,0.8) | 0.892 | 126 | 0.071@51.2 | +0.1 | yes |
| 40 | +0 | 12.8/0.8 | 0 | (12.8,0.8) | 0.497 | 90 | 0.065@51.2 | +0.0 | yes |
| 40 | +0 | 12.8/0.8 | -5 | (12.8,0.8) | 0.247 | 45 | 0.083@51.2 | +0.4 | yes |
| 40 | +0 | 3.2/0.8 | inf | (3.2,0.8) | 0.998 | 230 | 0.076@51.2 | -0.0 | yes |
| 40 | +0 | 3.2/0.8 | 10 | (3.2,0.8) | 0.907 | 151 | 0.079@51.2 | -0.1 | yes |
| 40 | +0 | 3.2/0.8 | 0 | (3.2,0.8) | 0.501 | 119 | 0.089@51.2 | +0.0 | yes |
| 40 | +0 | 3.2/0.8 | -5 | (3.2,0.8) | 0.237 | 58 | 0.075@51.2 | -0.0 | yes |
| 40 | +0 | 25.6/0.8 | inf | (25.6,0.8) | 0.996 | 127 | 0.064@51.2 | +0.0 | yes |
| 40 | +0 | 25.6/0.8 | 10 | (25.6,0.8) | 0.908 | 115 | 0.071@51.2 | -0.0 | yes |
| 40 | +0 | 25.6/0.8 | 0 | (25.6,0.8) | 0.490 | 65 | 0.067@51.2 | -0.0 | yes |
| 40 | +0 | 25.6/0.8 | -5 | (25.6,0.8) | 0.238 | 32 | 0.066@51.2 | -0.1 | yes |

### B = 80 MHz

| bw | off | true T_u/CP | SNR dB | best (T_u,CP) | metric | ratio | best-other-T_u | cfo kHz | match |
|---:|---:|---|---:|---|---:|---:|---:|---:|:--:|
| 20 | +15 | 12.8/0.8 | inf | (12.8,0.8) | 0.971 | 107 | 0.087@51.2 | -0.0 | yes |
| 20 | +15 | 12.8/0.8 | 10 | (12.8,0.8) | 0.884 | 108 | 0.083@51.2 | -0.0 | yes |
| 20 | +15 | 12.8/0.8 | 0 | (12.8,0.8) | 0.489 | 80 | 0.065@51.2 | -0.1 | yes |
| 20 | +15 | 12.8/0.8 | -5 | (12.8,0.8) | 0.236 | 62 | 0.060@51.2 | +0.1 | yes |
| 20 | +15 | 3.2/0.8 | inf | (3.2,0.8) | 0.970 | 225 | 0.092@51.2 | -0.0 | yes |
| 20 | +15 | 3.2/0.8 | 10 | (3.2,0.8) | 0.882 | 238 | 0.088@51.2 | -0.0 | yes |
| 20 | +15 | 3.2/0.8 | 0 | (3.2,0.8) | 0.484 | 141 | 0.057@51.2 | +0.2 | yes |
| 20 | +15 | 3.2/0.8 | -5 | (3.2,0.8) | 0.231 | 60 | 0.076@51.2 | +0.7 | yes |
| 20 | +15 | 25.6/0.8 | inf | (25.6,0.8) | 0.958 | 93 | 0.104@51.2 | -0.0 | yes |
| 20 | +15 | 25.6/0.8 | 10 | (25.6,0.8) | 0.871 | 88 | 0.092@51.2 | -0.0 | yes |
| 20 | +15 | 25.6/0.8 | 0 | (25.6,0.8) | 0.476 | 72 | 0.071@51.2 | +0.0 | yes |
| 20 | +15 | 25.6/0.8 | -5 | (25.6,0.8) | 0.232 | 38 | 0.054@51.2 | -0.3 | yes |
| 40 | +10 | 12.8/0.8 | inf | (12.8,0.8) | 0.991 | 161 | 0.067@51.2 | +0.0 | yes |
| 40 | +10 | 12.8/0.8 | 10 | (12.8,0.8) | 0.902 | 145 | 0.062@51.2 | +0.0 | yes |
| 40 | +10 | 12.8/0.8 | 0 | (12.8,0.8) | 0.491 | 95 | 0.063@51.2 | -0.0 | yes |
| 40 | +10 | 12.8/0.8 | -5 | (12.8,0.8) | 0.237 | 48 | 0.051@51.2 | +0.1 | yes |
| 40 | +10 | 3.2/0.8 | inf | (3.2,0.8) | 0.991 | 244 | 0.065@51.2 | -0.0 | yes |
| 40 | +10 | 3.2/0.8 | 10 | (3.2,0.8) | 0.902 | 204 | 0.062@51.2 | +0.0 | yes |
| 40 | +10 | 3.2/0.8 | 0 | (3.2,0.8) | 0.494 | 123 | 0.055@51.2 | +0.1 | yes |
| 40 | +10 | 3.2/0.8 | -5 | (3.2,0.8) | 0.239 | 60 | 0.052@51.2 | -0.1 | yes |
| 40 | +10 | 25.6/0.8 | inf | (25.6,0.8) | 0.999 | 121 | 0.073@51.2 | -0.0 | yes |
| 40 | +10 | 25.6/0.8 | 10 | (25.6,0.8) | 0.910 | 125 | 0.068@51.2 | -0.0 | yes |
| 40 | +10 | 25.6/0.8 | 0 | (25.6,0.8) | 0.495 | 86 | 0.064@51.2 | +0.0 | yes |
| 40 | +10 | 25.6/0.8 | -5 | (25.6,0.8) | 0.244 | 40 | 0.057@51.2 | +0.1 | yes |

### Noise negatives (pure complex noise, 3 trials per (B, bw))

| B | bw | trial | best (T_u,CP) | metric | ratio |
|---:|---:|---:|---|---:|---:|
| 20 | 20 | 0 | (51.2,0.2) | 0.114 | 3.6 |
| 20 | 20 | 1 | (51.2,0.2) | 0.105 | 3.4 |
| 20 | 20 | 2 | (51.2,0.2) | 0.096 | 2.9 |
| 40 | 20 | 0 | (51.2,0.2) | 0.093 | 3.3 |
| 40 | 20 | 1 | (51.2,0.2) | 0.086 | 3.1 |
| 40 | 20 | 2 | (51.2,0.2) | 0.087 | 3.0 |
| 40 | 40 | 0 | (51.2,0.2) | 0.070 | 3.1 |
| 40 | 40 | 1 | (51.2,0.2) | 0.073 | 3.3 |
| 40 | 40 | 2 | (51.2,0.2) | 0.069 | 3.0 |
| 80 | 20 | 0 | (51.2,0.2) | 0.104 | 3.6 |
| 80 | 20 | 1 | (51.2,0.2) | 0.085 | 2.9 |
| 80 | 20 | 2 | (51.2,0.2) | 0.112 | 3.7 |
| 80 | 40 | 0 | (51.2,0.2) | 0.091 | 4.3 |
| 80 | 40 | 1 | (51.2,0.2) | 0.072 | 3.2 |
| 80 | 40 | 2 | (51.2,0.2) | 0.074 | 3.4 |

### Threshold derivation (pre-registered decision rule)

- noise-best metric = **0.114** (max over 15 trials; by B: B20.0=0.114, B40.0=0.093, B80.0=0.112), noise-best ratio = **4.3**.
- Expected rule `metric > 5×noise-best AND ratio > 15` → metric floor 0.569: positives all-pass only at SNR levels {'inf': True, '10': True, '0': False, '-5': False} — it would drop every marginal (0 dB) positive, so the factor was adjusted.
- Chosen factor = **3.0×** (largest factor in {5,4,3,2,1.5,1.2} whose floor is cleared with >=25% margin by every calibration positive at in-band SNR >= 0 dB (expected 5x kept only if it qualifies)); **metric floor = 0.342**, **ratio_min = 15** (as pre-registered).
- Noise safety: all 15 noise trials fail the final rule → **True**.
- Sensitivity (all cases at a level pass): {'inf': True, '10': True, '0': True, '-5': False}; per-level minima: inf: metric≥0.958, ratio≥78, 10: metric≥0.871, ratio≥76, 0: metric≥0.476, ratio≥36, -5: metric≥0.227, ratio≥21.
- B=30/50 MHz frames were not in the calibration grid; the global (max over calibrated B) metric floor is applied to them, which is conservative.

## 2. STEP B — real frames: CP comb scan over the full grid

| id | sig | cls | stratum | B/bw | best (T_u,CP) µs | metric | ratio | cfo kHz | n_sym | best-other-T_u | in/out dB | verdict |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| train/1024 | 0 | 0 | clean | 80/20 | (3.2,0.8) | 0.662 | 31.9 | -1.1 | 2249 | 0.100@51.2 | 5.8 | **pass** |
| train/1153 | 3 | 0 | clean | 50/20 | (3.2,0.8) | 0.914 | 87.0 | -0.1 | 2249 | 0.105@51.2 | 5.8 | **pass** |
| train/20 | 1 | 0 | clean | 40/20 | (3.2,0.8) | 0.959 | 59.4 | -0.0 | 2249 | 0.100@51.2 | 15.9 | **pass** |
| train/208 | 0 | 0 | clean | 30/20 | (3.2,0.8) | 0.253 | 31.4 | -1.6 | 2249 | 0.094@51.2 | 12.4 | **fail** |
| train/1009 | 0 | 0 | overlap | 40/20 | (3.2,0.8) | 0.204 | 1.7 | +123.1 | 2249 | 0.152@51.2 | 7.9 | **fail** |
| train/123 | 2 | 1 | clean | 80/20 | (3.2,0.8) | 0.308 | 31.8 | -2.2 | 2249 | 0.089@51.2 | 13.8 | **fail** |
| train/1620 | 1 | 1 | clean | 40/20 | (3.2,0.8) | 0.195 | 33.5 | -4.7 | 2249 | 0.093@51.2 | 4.2 | **fail** |
| train/1730 | 1 | 1 | clean | 30/20 | (3.2,0.8) | 0.265 | 62.6 | -2.3 | 2249 | 0.102@51.2 | 11.1 | **fail** |
| train/2945 | 0 | 1 | clean | 50/20 | (3.2,0.8) | 0.837 | 31.6 | -0.1 | 2249 | 0.114@51.2 | 8.6 | **pass** |
| train/1021 | 1 | 1 | overlap | 40/20 | (3.2,0.8) | 0.958 | 129.1 | -0.1 | 2249 | 0.103@51.2 | 2.0 | **pass** |
| train/1128 | 3 | 2 | clean | 40/20 | (3.2,0.8) | 0.929 | 18.4 | +0.0 | 2249 | 0.100@51.2 | 10.3 | **pass** |
| train/1216 | 3 | 2 | clean | 80/20 | (3.2,0.8) | 0.240 | 1.1 | +0.0 | 2249 | 0.104@51.2 | 6.8 | **fail** |
| train/1425 | 2 | 2 | clean | 50/20 | (3.2,0.8) | 0.739 | 13.1 | -0.5 | 2249 | 0.109@51.2 | 1.2 | **fail** |
| train/1556 | 0 | 2 | clean | 30/20 | (3.2,0.8) | 0.962 | 18.9 | +0.0 | 2249 | 0.109@51.2 | 15.9 | **pass** |
| train/101 | 3 | 2 | overlap | 30/20 | (3.2,0.8) | 0.220 | 1.6 | -21.0 | 2249 | 0.191@51.2 | 10.0 | **fail** |
| train/1494 | 2 | 3 | clean | 80/40 | (3.2,0.8) | 0.877 | 204.3 | +0.0 | 2249 | 0.068@51.2 | 8.6 | **pass** |
| train/2982 | 0 | 3 | clean | 50/40 | (3.2,0.8) | 0.921 | 315.9 | -0.0 | 2249 | 0.069@51.2 | 11.8 | **pass** |
| train/1504 | 2 | 3 | clean | 80/40 | (3.2,0.8) | 0.881 | 220.1 | -0.1 | 2249 | 0.104@51.2 | 8.5 | **pass** |
| train/2240 | 5 | 3 | clean | 80/40 | (3.2,0.8) | 0.789 | 157.3 | +0.0 | 2249 | 0.093@51.2 | 6.6 | **pass** |
| train/1021 | 3 | 3 | overlap | 40/40 | (3.2,0.8) | 0.867 | 24.8 | +1.2 | 2249 | 0.071@51.2 | -- | **pass** |
| train/1337 | 4 | 4 | clean | 80/40 | (3.2,0.8) | 0.905 | 240.9 | -0.1 | 2249 | 0.101@51.2 | 10.4 | **pass** |
| train/1550 | 0 | 4 | clean | 50/40 | (3.2,0.8) | 0.903 | 198.0 | -0.1 | 2249 | 0.086@51.2 | 10.3 | **pass** |
| train/1836 | 1 | 4 | clean | 80/40 | (3.2,0.8) | 0.940 | 94.4 | +0.1 | 2249 | 0.101@51.2 | 8.8 | **pass** |
| train/1980 | 3 | 4 | clean | 80/40 | (3.2,0.2) | 0.140 | 1.1 | -0.6 | 2646 | 0.085@51.2 | 4.3 | **fail** |
| train/103 | 2 | 4 | overlap | 80/40 | (3.2,0.8) | 0.341 | 27.4 | +1.0 | 2249 | 0.176@51.2 | 5.7 | **fail** |
| train/10 | 3 | 5 | clean | 80/40 | (3.2,0.2) | 0.208 | 1.1 | -2.4 | 2646 | 0.098@51.2 | 3.8 | **fail** |
| train/1975 | 0 | 5 | clean | 50/40 | (3.2,0.8) | 0.908 | 248.3 | -0.1 | 2249 | 0.080@51.2 | 8.2 | **pass** |
| train/3042 | 0 | 5 | clean | 50/40 | (3.2,0.2) | 0.241 | 1.1 | -4.9 | 2646 | 0.099@51.2 | 13.0 | **fail** |
| train/312 | 0 | 5 | clean | 50/40 | (3.2,0.8) | 0.947 | 373.6 | +0.1 | 2249 | 0.073@51.2 | 13.5 | **pass** |
| train/1016 | 0 | 5 | overlap | 40/40 | (3.2,0.2) | 0.253 | 1.0 | -8.6 | 2646 | 0.128@51.2 | -- | **fail** |

Verdict counts (pass): 17/30; by class/stratum: {'0/clean': 3, '1/clean': 1, '1/overlap': 1, '2/clean': 2, '3/clean': 4, '3/overlap': 1, '4/clean': 3, '5/clean': 2}.


## 3. STEP C — subcarrier statistics (passing frames only)

`cp_strip_fft` at the detected (phi, cfo) with n_sym capped at 600. CFO sign: the helper multiplies by exp(+j2π·cfo·n), so −detected is passed to apply the detected correction (synthetic evidence in the header of the script; for SCS-aligned content cfo ≈ 0 and it is moot). Theory m4: QPSK 1.00, 16QAM 1.32, 64QAM 1.38; consistency = labeled order within ±0.15 of the per-subcarrier m4 median (exploratory, n is tiny).

Across these 17 passes, per-sc m4 medians by labeled order — QPSK: 1.04–1.47; 16QAM: 1.38–2.05; 64QAM: 1.44–1.50. QPSK vs the 16/64QAM orders is partially separated, 16QAM vs 64QAM (Δtheory 0.06) is not; 7 frames fall outside ±0.15 of their labeled order's theory value — per-frame facts, exploratory, n is tiny.

| id | cls (label) | n_sym | n_strong_sc | snr_db_proxy | per-sc m4 median | m4 pool | theory | consistent? |
|---|---|---:|---:|---:|---:|---:|---|:--:|
| train/1024/0 | 0 (QPSK) | 600 | 49 | 36.1 | 1.365 | 1.431 | 1.00 | no (near: 16QAM,64QAM) |
| train/1153/3 | 0 (QPSK) | 600 | 49 | 33.0 | 1.130 | 1.159 | 1.00 | yes (near: QPSK) |
| train/20/1 | 0 (QPSK) | 600 | 49 | 35.3 | 1.043 | 1.089 | 1.00 | yes (near: QPSK) |
| train/2945/0 | 1 (16QAM) | 600 | 49 | 34.6 | 1.461 | 1.484 | 1.32 | yes (near: 16QAM,64QAM) |
| train/1021/1 | 1 (16QAM) | 600 | 49 | 32.3 | 1.382 | 1.412 | 1.32 | yes (near: 16QAM,64QAM) · *overlap stratum: co-channel contamination possible* |
| train/1128/3 | 2 (64QAM) | 600 | 49 | 31.8 | 1.481 | 1.498 | 1.38 | yes (near: 64QAM) |
| train/1556/0 | 2 (64QAM) | 600 | 49 | 31.1 | 1.497 | 1.525 | 1.38 | yes (near: 64QAM) |
| train/1494/2 | 3 (QPSK) | 600 | 99 | 35.4 | 1.174 | 1.204 | 1.00 | no (near: 16QAM) |
| train/2982/0 | 3 (QPSK) | 600 | 99 | 28.0 | 1.108 | 1.144 | 1.00 | yes (near: QPSK) |
| train/1504/2 | 3 (QPSK) | 600 | 99 | 34.0 | 1.191 | 1.238 | 1.00 | no (near: 16QAM) |
| train/2240/5 | 3 (QPSK) | 600 | 99 | 34.2 | 1.241 | 1.276 | 1.00 | no (near: 16QAM,64QAM) |
| train/1021/3 | 3 (QPSK) | 600 | 99 | 335.4 | 1.474 | 1.506 | 1.00 | no (near: 64QAM) · *B==bw: subcarrier noise region empty; snr_db_proxy/n_strong_sc not meaningful* · *overlap stratum: co-channel contamination possible* |
| train/1337/4 | 4 (16QAM) | 600 | 99 | 34.2 | 2.049 | 2.060 | 1.32 | no (near: none) |
| train/1550/0 | 4 (16QAM) | 600 | 99 | 29.7 | 1.422 | 1.478 | 1.32 | yes (near: 16QAM,64QAM) |
| train/1836/1 | 4 (16QAM) | 600 | 99 | 33.8 | 1.648 | 1.669 | 1.32 | no (near: none) |
| train/1975/0 | 5 (64QAM) | 600 | 99 | 30.2 | 1.497 | 1.506 | 1.38 | yes (near: 64QAM) |
| train/312/0 | 5 (64QAM) | 600 | 99 | 31.8 | 1.438 | 1.445 | 1.38 | yes (near: 16QAM,64QAM) |


## 4. STEP D — failing frames: per-frame evidence and causes

(a) is pre-registered: in/out < 6 dB → "no target energy". The narrowband label (b) uses occupied-width-at−10 dB < 30% of the box — a post-hoc reading aid; the measured numbers are the evidence. (c) overlap stratum is flagged wherever it applies. Per-frame facts only; no dataset-level generalization is made here.

| id | cls | stratum | in/out dB | occ@−10dB MHz (frac) | peak offsets MHz (rel dB) | causes |
|---|---:|---|---:|---|---|---|
| train/208/0 | 0 | clean | 12.4 | 19.9 (1.00) | -5.3 (0 dB); +5.3 (-0 dB); -4.1 (-0 dB) | broadband_energy_no_cp_comb; sub-floor comb evidence |
| train/1009/0 | 0 | overlap | 7.9 | 1.6 (0.08) | +4.5 (0 dB); +10.0 (-8 dB); -5.0 (-16 dB) | narrowband_energy; overlap_contamination_possible |
| train/123/2 | 1 | clean | 13.8 | 20.0 (1.00) | -5.3 (0 dB); -4.1 (-0 dB); +5.6 (-1 dB) | broadband_energy_no_cp_comb; sub-floor comb evidence |
| train/1620/1 | 1 | clean | 4.2 | 20.0 (1.00) | +9.1 (0 dB); -5.9 (-0 dB); -3.2 (-0 dB) | no_target_energy; sub-floor comb evidence |
| train/1730/1 | 1 | clean | 11.1 | 19.9 (1.00) | -5.6 (0 dB); -4.3 (-0 dB); +5.6 (-0 dB) | broadband_energy_no_cp_comb; sub-floor comb evidence |
| train/1216/3 | 2 | clean | 6.8 | 13.9 (0.69) | +5.0 (0 dB); +2.8 (-0 dB); -2.8 (-3 dB) | broadband_energy_no_cp_comb |
| train/1425/2 | 2 | clean | 1.2 | 10.0 (0.50) | -0.6 (0 dB); +2.5 (-3 dB); -1.9 (-3 dB) | no_target_energy; metric above floor but ratio below min |
| train/101/3 | 2 | overlap | 10.0 | 1.8 (0.09) | +0.0 (0 dB); -5.5 (-2 dB); -4.1 (-10 dB) | narrowband_energy; overlap_contamination_possible |
| train/1980/3 | 4 | clean | 4.3 | 19.2 (0.48) | -4.7 (0 dB); -20.0 (-2 dB); +5.3 (-2 dB) | no_target_energy |
| train/103/2 | 4 | overlap | 5.7 | 2.0 (0.05) | +5.0 (0 dB); +10.0 (-11 dB); +11.2 (-12 dB) | no_target_energy; overlap_contamination_possible; sub-floor comb evidence |
| train/10/3 | 5 | clean | 3.8 | 7.2 (0.18) | -5.0 (0 dB); -3.7 (-4 dB); +2.5 (-6 dB) | no_target_energy |
| train/3042/0 | 5 | clean | 13.0 | 14.4 (0.36) | -2.5 (0 dB); +15.0 (-0 dB); +3.7 (-1 dB) | broadband_energy_no_cp_comb |
| train/1016/0 | 5 | overlap | -- | 3.9 (0.10) | -16.0 (0 dB); -5.0 (-4 dB); +17.5 (-7 dB) | in_out_unavailable_box_equals_observation; narrowband_energy; overlap_contamination_possible |

Primary-cause counts: {'broadband_energy_no_cp_comb': 5, 'narrowband_energy': 3, 'no_target_energy': 5}. Sub-floor comb evidence: ['train/208', 'train/123', 'train/1620', 'train/1730', 'train/103']; metric>floor with ratio<min: ['train/1425']; overlap-flagged failing frames: ['train/1009', 'train/101', 'train/103', 'train/1016'].


## 5. Facts vs inferences

**Facts (mechanical, label-blind):**

- Calibration: 15 signal cases × 4 SNR levels and 15 noise negatives; noise-best metric 0.114, noise-best ratio 4.3; no noise trial passes the final rule.
- Real frames: 17/30 pass the pre-registered rule (metric > 0.342 AND ratio > 15); pass/fail by class/stratum: {'0/clean': 3, '1/clean': 1, '1/overlap': 1, '2/clean': 2, '3/clean': 4, '3/overlap': 1, '4/clean': 3, '5/clean': 2}.
- All passing frames select the same numerology (3.2, 0.8) µs with cfo ≈ 0 kHz and best-other-T_u at noise level; failing frames with ratio ≈ 1.0–1.7 (tone-like phase distribution) are ['train/1009', 'train/1216', 'train/101', 'train/1980', 'train/10', 'train/3042', 'train/1016'].
- PASS train/1024/0: best candidate (3.2,0.8) µs, metric 0.662, ratio 32, cfo -1.1 kHz, best-other-T_u 0.100@51.2.
- PASS train/1153/3: best candidate (3.2,0.8) µs, metric 0.914, ratio 87, cfo -0.1 kHz, best-other-T_u 0.105@51.2.
- PASS train/20/1: best candidate (3.2,0.8) µs, metric 0.959, ratio 59, cfo -0.0 kHz, best-other-T_u 0.100@51.2.
- PASS train/2945/0: best candidate (3.2,0.8) µs, metric 0.837, ratio 32, cfo -0.1 kHz, best-other-T_u 0.114@51.2.
- PASS train/1021/1: best candidate (3.2,0.8) µs, metric 0.958, ratio 129, cfo -0.1 kHz, best-other-T_u 0.103@51.2.
- PASS train/1128/3: best candidate (3.2,0.8) µs, metric 0.929, ratio 18, cfo +0.0 kHz, best-other-T_u 0.100@51.2.
- PASS train/1556/0: best candidate (3.2,0.8) µs, metric 0.962, ratio 19, cfo +0.0 kHz, best-other-T_u 0.109@51.2.
- PASS train/1494/2: best candidate (3.2,0.8) µs, metric 0.877, ratio 204, cfo +0.0 kHz, best-other-T_u 0.068@51.2.
- PASS train/2982/0: best candidate (3.2,0.8) µs, metric 0.921, ratio 316, cfo -0.0 kHz, best-other-T_u 0.069@51.2.
- PASS train/1504/2: best candidate (3.2,0.8) µs, metric 0.881, ratio 220, cfo -0.1 kHz, best-other-T_u 0.104@51.2.
- PASS train/2240/5: best candidate (3.2,0.8) µs, metric 0.789, ratio 157, cfo +0.0 kHz, best-other-T_u 0.093@51.2.
- PASS train/1021/3: best candidate (3.2,0.8) µs, metric 0.867, ratio 25, cfo +1.2 kHz, best-other-T_u 0.071@51.2.
- PASS train/1337/4: best candidate (3.2,0.8) µs, metric 0.905, ratio 241, cfo -0.1 kHz, best-other-T_u 0.101@51.2.
- PASS train/1550/0: best candidate (3.2,0.8) µs, metric 0.903, ratio 198, cfo -0.1 kHz, best-other-T_u 0.086@51.2.
- PASS train/1836/1: best candidate (3.2,0.8) µs, metric 0.940, ratio 94, cfo +0.1 kHz, best-other-T_u 0.101@51.2.
- PASS train/1975/0: best candidate (3.2,0.8) µs, metric 0.908, ratio 248, cfo -0.1 kHz, best-other-T_u 0.080@51.2.
- PASS train/312/0: best candidate (3.2,0.8) µs, metric 0.947, ratio 374, cfo +0.1 kHz, best-other-T_u 0.073@51.2.
- FAIL train/208/0: best (3.2,0.8) µs metric 0.253 ratio 31.4; in/out 12.4 dB; occ@−10 dB 19.9 MHz; causes: broadband_energy_no_cp_comb.
- FAIL train/1009/0: best (3.2,0.8) µs metric 0.204 ratio 1.7; in/out 7.9 dB; occ@−10 dB 1.6 MHz; causes: narrowband_energy; overlap_contamination_possible.
- FAIL train/123/2: best (3.2,0.8) µs metric 0.308 ratio 31.8; in/out 13.8 dB; occ@−10 dB 20.0 MHz; causes: broadband_energy_no_cp_comb.
- FAIL train/1620/1: best (3.2,0.8) µs metric 0.195 ratio 33.5; in/out 4.2 dB; occ@−10 dB 20.0 MHz; causes: no_target_energy.
- FAIL train/1730/1: best (3.2,0.8) µs metric 0.265 ratio 62.6; in/out 11.1 dB; occ@−10 dB 19.9 MHz; causes: broadband_energy_no_cp_comb.
- FAIL train/1216/3: best (3.2,0.8) µs metric 0.240 ratio 1.1; in/out 6.8 dB; occ@−10 dB 13.9 MHz; causes: broadband_energy_no_cp_comb.
- FAIL train/1425/2: best (3.2,0.8) µs metric 0.739 ratio 13.1; in/out 1.2 dB; occ@−10 dB 10.0 MHz; causes: no_target_energy.
- FAIL train/101/3: best (3.2,0.8) µs metric 0.220 ratio 1.6; in/out 10.0 dB; occ@−10 dB 1.8 MHz; causes: narrowband_energy; overlap_contamination_possible.
- FAIL train/1980/3: best (3.2,0.2) µs metric 0.140 ratio 1.1; in/out 4.3 dB; occ@−10 dB 19.2 MHz; causes: no_target_energy.
- FAIL train/103/2: best (3.2,0.8) µs metric 0.341 ratio 27.4; in/out 5.7 dB; occ@−10 dB 2.0 MHz; causes: no_target_energy; overlap_contamination_possible.
- FAIL train/10/3: best (3.2,0.2) µs metric 0.208 ratio 1.1; in/out 3.8 dB; occ@−10 dB 7.2 MHz; causes: no_target_energy.
- FAIL train/3042/0: best (3.2,0.2) µs metric 0.241 ratio 1.1; in/out 13.0 dB; occ@−10 dB 14.4 MHz; causes: broadband_energy_no_cp_comb.
- FAIL train/1016/0: best (3.2,0.2) µs metric 0.253 ratio 1.0; in/out -- dB; occ@−10 dB 3.9 MHz; causes: in_out_unavailable_box_equals_observation; narrowband_energy; overlap_contamination_possible.

**Inferences (interpretive, explicitly per-frame):**

- Context, scoped to these 30 frames only: the round-1 census (pre-frequency-fix demodulation) concluded that WiFi boxes carry no OFDM structure; with the audited demodulation, 17 of these 30 frames show a strong CP-OFDM comb at (3.2, 0.8) µs — for at least these frames the earlier negative is attributable to the demodulation bug, not to missing structure.
- train/1024/0: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.365 is NOT consistent with the labeled order QPSK (theory 1.00); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1153/3: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.130 is consistent with the labeled order QPSK (theory 1.00); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/20/1: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.043 is consistent with the labeled order QPSK (theory 1.00); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/2945/0: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.461 is consistent with the labeled order 16QAM (theory 1.32); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1021/1: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.382 is consistent with the labeled order 16QAM (theory 1.32); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1128/3: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.481 is consistent with the labeled order 64QAM (theory 1.38); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1556/0: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.497 is consistent with the labeled order 64QAM (theory 1.38); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1494/2: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.174 is NOT consistent with the labeled order QPSK (theory 1.00); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/2982/0: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.108 is consistent with the labeled order QPSK (theory 1.00); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1504/2: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.191 is NOT consistent with the labeled order QPSK (theory 1.00); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/2240/5: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.241 is NOT consistent with the labeled order QPSK (theory 1.00); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1021/3: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.474 is NOT consistent with the labeled order QPSK (theory 1.00); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1337/4: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 2.049 is NOT consistent with the labeled order 16QAM (theory 1.32); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1550/0: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.422 is consistent with the labeled order 16QAM (theory 1.32); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1836/1: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.648 is NOT consistent with the labeled order 16QAM (theory 1.32); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/1975/0: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.497 is consistent with the labeled order 64QAM (theory 1.38); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/312/0: the in-box content carries a CP-OFDM comb at the detected numerology; its per-subcarrier m4 median 1.438 is consistent with the labeled order 64QAM (theory 1.38); noise inflates m4 toward 2 and 16QAM vs 64QAM (Δtheory 0.06) is not separable at this n — exploratory.
- train/208/0: in-band energy roughly fills the box (occ 19.9 MHz, in/out 12.4 dB, peak/floor 1.8 dB) but no candidate numerology produces a CP comb above the calibrated floor.
- train/208/0: BORDERLINE — the scan shows a comb-like phase structure at (3.2,0.8) µs with ratio 31.4 > 15 but metric 0.253 below the floor 0.342 (~2.2× noise-best): structure may be present at low in-band SNR; the verdict follows the pre-registered rule.
- train/1009/0: energy is present but occupies 1.6 MHz of the 20 MHz box — narrowband content (peaks at +4.5 MHz, +10.0 MHz, -5.0 MHz), not a box-filling CP-OFDM.
- train/1009/0: overlap stratum — co-channel contamination is possible; no claim is made about other frames.
- train/123/2: in-band energy roughly fills the box (occ 20.0 MHz, in/out 13.8 dB, peak/floor 2.2 dB) but no candidate numerology produces a CP comb above the calibrated floor.
- train/123/2: BORDERLINE — the scan shows a comb-like phase structure at (3.2,0.8) µs with ratio 31.8 > 15 but metric 0.308 below the floor 0.342 (~2.7× noise-best): structure may be present at low in-band SNR; the verdict follows the pre-registered rule.
- train/1620/1: in/out 4.2 dB < 6 dB → by the pre-registered rule the box median sits at the capture floor; the box is not literally empty (occ@−10 dB 20.0 MHz, in-box peak/floor 1.6 dB — sparse tone-like spikes at +9.1 MHz, -5.9 MHz, -3.2 MHz).
- train/1620/1: BORDERLINE — the scan shows a comb-like phase structure at (3.2,0.8) µs with ratio 33.5 > 15 but metric 0.195 below the floor 0.342 (~1.7× noise-best): structure may be present at low in-band SNR; the verdict follows the pre-registered rule.
- train/1730/1: in-band energy roughly fills the box (occ 19.9 MHz, in/out 11.1 dB, peak/floor 2.1 dB) but no candidate numerology produces a CP comb above the calibrated floor.
- train/1730/1: BORDERLINE — the scan shows a comb-like phase structure at (3.2,0.8) µs with ratio 62.6 > 15 but metric 0.265 below the floor 0.342 (~2.3× noise-best): structure may be present at low in-band SNR; the verdict follows the pre-registered rule.
- train/1216/3: in-band energy roughly fills the box (occ 13.9 MHz, in/out 6.8 dB, peak/floor 7.7 dB) but no candidate numerology produces a CP comb above the calibrated floor.
- train/1425/2: in/out 1.2 dB < 6 dB → by the pre-registered rule the box median sits at the capture floor; the box is not literally empty (occ@−10 dB 10.0 MHz, in-box peak/floor 9.9 dB — sparse tone-like spikes at -0.6 MHz, +2.5 MHz, -1.9 MHz).
- train/1425/2: BORDERLINE — comb metric 0.739 is ABOVE the floor but the phase-structure ratio 13.1 ≤ 15; also, a strong in-box comb with in/out median 1.2 dB suggests comparable energy exists outside the label box (per-frame observation).
- train/101/3: energy is present but occupies 1.8 MHz of the 20 MHz box — narrowband content (peaks at +0.0 MHz, -5.5 MHz, -4.1 MHz), not a box-filling CP-OFDM.
- train/101/3: overlap stratum — co-channel contamination is possible; no claim is made about other frames.
- train/1980/3: in/out 4.3 dB < 6 dB → by the pre-registered rule the box median sits at the capture floor; the box is not literally empty (occ@−10 dB 19.2 MHz, in-box peak/floor 10.3 dB — sparse tone-like spikes at -4.7 MHz, -20.0 MHz, +5.3 MHz).
- train/103/2: in/out 5.7 dB < 6 dB → by the pre-registered rule the box median sits at the capture floor; the box is not literally empty (occ@−10 dB 2.0 MHz, in-box peak/floor 15.9 dB — sparse tone-like spikes at +5.0 MHz, +10.0 MHz, +11.2 MHz).
- train/103/2: BORDERLINE — the scan shows a comb-like phase structure at (3.2,0.8) µs with ratio 27.4 > 15 but metric 0.341 below the floor 0.342 (~3.0× noise-best): structure may be present at low in-band SNR; the verdict follows the pre-registered rule.
- train/103/2: overlap stratum — co-channel contamination is possible; no claim is made about other frames.
- train/10/3: in/out 3.8 dB < 6 dB → by the pre-registered rule the box median sits at the capture floor; the box is not literally empty (occ@−10 dB 7.2 MHz, in-box peak/floor 14.6 dB — sparse tone-like spikes at -5.0 MHz, -3.7 MHz, +2.5 MHz).
- train/3042/0: in-band energy roughly fills the box (occ 14.4 MHz, in/out 13.0 dB, peak/floor 12.3 dB) but no candidate numerology produces a CP comb above the calibrated floor.
- train/1016/0: energy is present but occupies 3.9 MHz of the 40 MHz box — narrowband content (peaks at -16.0 MHz, -5.0 MHz, +17.5 MHz), not a box-filling CP-OFDM. in/out is unavailable because the box equals the whole observation (B=bw).
- train/1016/0: overlap stratum — co-channel contamination is possible; no claim is made about other frames.

No statement in this report generalizes to the full dataset; the audit decides that at a later phase.


## 6. Report-back summary

- Pass/fail: **17/30**; by class: {0: 3, 1: 2, 2: 2, 3: 5, 4: 3, 5: 2}; by stratum: {'clean': 15, 'overlap': 2}; by class/stratum: {'0/clean': 3, '1/clean': 1, '1/overlap': 1, '2/clean': 2, '3/clean': 4, '3/overlap': 1, '4/clean': 3, '5/clean': 2}.
- Every passing frame's best candidate is **(T_u, CP) = (3.2, 0.8) µs** (SCS 312.5 kHz, legacy-style numerology); the pre-registered HE candidate T_u = 12.8 µs never wins, and best-other-T_u metrics stay at noise level (max 0.114 vs noise-best 0.114).
- Passing candidates and m4 vs theory: train/1024/0 cls0 (3.2,0.8)µs m4_med=1.365 [QPSK MISMATCH]; train/1153/3 cls0 (3.2,0.8)µs m4_med=1.130 [QPSK ok]; train/20/1 cls0 (3.2,0.8)µs m4_med=1.043 [QPSK ok]; train/2945/0 cls1 (3.2,0.8)µs m4_med=1.461 [16QAM ok]; train/1021/1 cls1 (3.2,0.8)µs m4_med=1.382 [16QAM ok]; train/1128/3 cls2 (3.2,0.8)µs m4_med=1.481 [64QAM ok]; train/1556/0 cls2 (3.2,0.8)µs m4_med=1.497 [64QAM ok]; train/1494/2 cls3 (3.2,0.8)µs m4_med=1.174 [QPSK MISMATCH]; train/2982/0 cls3 (3.2,0.8)µs m4_med=1.108 [QPSK ok]; train/1504/2 cls3 (3.2,0.8)µs m4_med=1.191 [QPSK MISMATCH]; train/2240/5 cls3 (3.2,0.8)µs m4_med=1.241 [QPSK MISMATCH]; train/1021/3 cls3 (3.2,0.8)µs m4_med=1.474 [QPSK MISMATCH]; train/1337/4 cls4 (3.2,0.8)µs m4_med=2.049 [16QAM MISMATCH]; train/1550/0 cls4 (3.2,0.8)µs m4_med=1.422 [16QAM ok]; train/1836/1 cls4 (3.2,0.8)µs m4_med=1.648 [16QAM MISMATCH]; train/1975/0 cls5 (3.2,0.8)µs m4_med=1.497 [64QAM ok]; train/312/0 cls5 (3.2,0.8)µs m4_med=1.438 [64QAM ok].
- Failure causes: {'broadband_energy_no_cp_comb': 5, 'narrowband_energy': 3, 'no_target_energy': 5}; sub-floor comb evidence: ['train/208', 'train/123', 'train/1620', 'train/1730', 'train/103']; metric>floor with ratio<min: ['train/1425']; overlap-flagged: ['train/1009', 'train/101', 'train/103', 'train/1016'].
