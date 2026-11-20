# Baseline-3 Phase A' · Track E — label-location energy audit

Generated 2026-11-21 by `scripts/baseline3_pa_extract.py` in 4 s on 4 workers. Full per-frame numbers in `phase_ap_extract.json`.
Sample set: `reports/local_iq_round1/phase_ap_frames.json` — 46 frames, fit split only, all `train/`. Nothing under `advanced/test` was read.

## Method

- Analysis window: min(10 ms, t2−t1) centered in [t1, t2] — 10.0 ms for every frame in this set; FFT length N = B·10k samples.
- Frequency convention (audited, unit-tested): np.fft zero frequency = observation center `f_lo + B/2`, i.e. `abs = f_lo + B/2 + digital`.
- **RAW** (un-demodulated, full band): `in/out dB` = 10·log10(median PSD inside [f1,f2] / median PSD outside the box but inside the observation, ≥2 MHz away from the box); top-3 peaks (PSD smoothed ~50 kHz, min separation 200 kHz) as absolute MHz with inside-box flags; strongest in-box peak vs the out-of-box median (dB).
- **CORRECTED**: `demod_to_center` to (f1+f2)/2 → `band_limit_fft`(keep_bw = bw·1.02, edge_frac = 0.05) → trim 0.5 ms both ends → dominant peak offset from DC (kHz, signed), occupied width at −10 dB rel. in-band max (MHz), energy keep ratio.
- **OLD** (round-1 wrong demod, shift = f_center − f_lo): dominant full-band peak offset from DC (MHz). Under this shift the labeled content is displaced by exactly B/2, so a value near ±B/2 means the old code was looking at the wrong part of the capture (its DC-centered box filter kept the band displaced +B/2 from the label, wrapping at the observation edges).
- **Verdict `target_at_label`**: in/out dB ≥ 6 dB AND strongest raw in-box peak ≥ 10 dB above the out-of-box median. Energy presence ONLY — explicitly NOT about modulation structure.

## 1. Per-frame table (ids are `sample:signal`, all under `train/`)

| id | cls | stratum | B | bw | in/out dB | pk kHz | occ MHz | old MHz | @label |
|---:|---:|---|---:|---:|---:|---:|---:|---:|:--:|
| train/1024:0 | 0 | clean | 80 | 20 | +5.8 | +3730.9 | 16.22 | -11.9 | ✗ |
| train/1153:3 | 0 | clean | 50 | 20 | +4.7 | -4997.6 | 16.45 | +1.7 | ✗ |
| train/20:1 | 0 | clean | 40 | 20 | +15.9 | -5006.8 | 16.32 | +15.0 | ✓ |
| train/208:0 | 0 | clean | 30 | 20 | +14.0 | -5267.7 | 19.62 | +9.4 | ✓ |
| train/1009:0 | 0 | overlap | 40 | 20 | +8.3 | +4525.1 | 1.42 | -15.5 | ✓ |
| train/123:2 | 1 | clean | 80 | 20 | +14.1 | -5019.8 | 19.58 | +25.0 | ✓ |
| train/1620:1 | 1 | clean | 40 | 20 | +4.5 | +9131.1 | 20.25 | -0.8 | ✗ |
| train/1730:1 | 1 | clean | 30 | 20 | +12.1 | -5606.3 | 19.64 | -2.3 | ✓ |
| train/2945:0 | 1 | clean | 50 | 20 | +8.6 | +3762.0 | 16.65 | -5.1 | ✓ |
| train/1021:1 | 1 | overlap | 40 | 20 | +2.4 | -4993.6 | 19.89 | +1.0 | ✗ |
| train/1128:3 | 2 | clean | 40 | 20 | +10.0 | +7483.0 | 16.31 | -0.4 | ✓ |
| train/1216:3 | 2 | clean | 80 | 20 | +6.8 | +4996.1 | 13.47 | +25.4 | ✓ |
| train/1425:2 | 2 | clean | 50 | 20 | +1.3 | -639.7 | 9.68 | -13.0 | ✗ |
| train/1556:0 | 2 | clean | 30 | 20 | +16.1 | +2484.0 | 16.29 | -12.5 | ✓ |
| train/101:3 | 2 | overlap | 30 | 20 | +9.9 | +0.9 | 1.88 | -0.0 | ✓ |
| train/1494:2 | 3 | clean | 80 | 40 | +8.5 | -5004.8 | 35.99 | -10.9 | ✓ |
| train/2982:0 | 3 | clean | 50 | 40 | +11.9 | +15622.7 | 35.79 | -9.4 | ✓ |
| train/1504:2 | 3 | clean | 80 | 40 | +8.5 | +1881.8 | 35.28 | -38.1 | ✓ |
| train/2240:5 | 3 | clean | 80 | 40 | +6.7 | +2504.0 | 38.42 | -37.5 | ✓ |
| train/1021:3 | 3 | overlap | 40 | 40 | n/a | -18961.3 | 6.99 | +1.0 | ✗ᵉ |
| train/1337:4 | 4 | clean | 80 | 40 | +10.4 | -12201.3 | 35.66 | +28.1 | ✓ |
| train/1550:0 | 4 | clean | 50 | 40 | +10.3 | +7513.7 | 35.62 | -18.8 | ✓ |
| train/1836:1 | 4 | clean | 80 | 40 | +8.3 | +15926.7 | 35.81 | -11.1 | ✓ |
| train/1980:3 | 4 | clean | 80 | 40 | +4.6 | -20752.6 | 13.36 | -14.9 | ✗ |
| train/103:2 | 4 | overlap | 80 | 40 | +5.7 | +5015.3 | 2.01 | -35.0 | ✗ |
| train/10:3 | 5 | clean | 80 | 40 | +3.7 | -5005.6 | 6.66 | -5.0 | ✗ |
| train/1975:0 | 5 | clean | 50 | 40 | +8.3 | -17521.4 | 35.96 | +7.5 | ✓ |
| train/3042:0 | 5 | clean | 50 | 40 | +13.0 | -2510.7 | 14.12 | +22.5 | ✓ |
| train/312:0 | 5 | clean | 50 | 40 | +13.5 | +11883.6 | 35.67 | -14.1 | ✓ |
| train/1016:0 | 5 | overlap | 40 | 40 | n/a | -16000.4 | 0.68 | +4.0 | ✗ᵉ |
| train/1037:1 | 10 | clean | 50 | 10 | +12.6 | -468.9 | 9.38 | +5.1 | ✓ |
| train/1058:1 | 10 | clean | 40 | 10 | +22.1 | +2678.2 | 9.46 | -2.0 | ✓ |
| train/1099:0 | 10 | clean | 80 | 10 | +9.8 | +1840.9 | 9.46 | -14.6 | ✓ |
| train/1160:1 | 10 | clean | 20 | 10 | +11.3 | +10.3 | 9.99 | +2.0 | ✓ |
| train/1011:0 | 10 | overlap | 20 | 10 | +4.5 | -2.1 | 0.07 | +10.0 | ✗ |
| train/1093:0 | 11 | clean | 20 | 10 | +17.6 | -2691.9 | 9.41 | +7.4 | ✓ |
| train/123:1 | 11 | clean | 80 | 10 | +9.7 | -2822.1 | 8.26 | +25.0 | ✓ |
| train/1471:0 | 11 | clean | 50 | 10 | +13.4 | -968.9 | 9.63 | +15.5 | ✓ |
| train/1986:0 | 11 | clean | 40 | 10 | +6.8 | +3167.4 | 9.63 | +9.9 | ✓ |
| train/1004:1 | 11 | overlap | 20 | 10 | +23.1 | -8.1 | 9.05 | +10.0 | ✓ |
| train/9:3 | 6 | control | 40 | 1 | +17.7 | +51.3 | 0.97 | -14.0 | ✓ |
| train/21:4 | 6 | control | 30 | 1 | +24.0 | +16.6 | 0.90 | -11.0 | ✓ |
| train/50:2 | 6 | control | 20 | 1 | +11.6 | -77.8 | 0.90 | -9.0 | ✓ |
| train/9:2 | 8 | control | 40 | 2 | +21.5 | -439.1 | 1.88 | +14.0 | ✓ |
| train/18:2 | 8 | control | 80 | 2 | +11.5 | -433.0 | 1.90 | +13.0 | ✓ |
| train/57:3 | 8 | control | 50 | 2 | +21.2 | -23.1 | 1.96 | -19.0 | ✓ |

ᵉ box spans the whole observation → no out-of-box region ≥2 MHz away; in/out dB not evaluable, verdict false by rule. Frames: train/1021:3, train/1016:0.

## 2. Negative controls through the identical chain

### 2a. Synthetic complex white noise, (B, bw) matched to real frames

| config | B | bw | in/out dB | pk kHz | occ MHz | keep | old MHz | @label |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| noise (40,20) | 40 | 20 | -0.0 | +4287.8 | 20.40 | 0.495 | -15.7 | ✗ |
| noise (80,40) | 80 | 40 | -0.0 | +7240.4 | 40.80 | 0.498 | +9.1 | ✗ |
| noise (40,10) | 40 | 10 | -0.0 | -1010.0 | 10.20 | 0.248 | +6.2 | ✗ |
| noise (30,2) | 30 | 2 | -0.0 | +283.9 | 2.04 | 0.066 | -2.3 | ✗ |

Expected "nothing there": in/out ≈ 0 dB, occupied ≈ full keep band, no in-box peak dominance, verdict false.

### 2b. Real decoy bands: +15 MHz from the box center, clipped into the observation, JSON-clear when possible

| id | cls | decoy band MHz | shift | JSON clear | in/out dB | pk kHz | occ MHz | old MHz | @label |
|---:|---:|---|---:|:--:|---:|---:|---:|---:|:--:|
| train/123:1 | 11 | [2457,2467] | +15 | yes | -0.9 | -4123.4 | 10.20 | +10.0 | ✗ |
| train/1471:0 | 11 | [2437,2447] | +15 | yes | -3.1 | -1947.1 | 7.59 | +0.5 | ✗ |
| train/1986:0 | 11 | [2447,2457] | +15 | yes | -3.5 | -3990.1 | 1.82 | -5.1 | ✗ |
| train/57:3 | 8 | [2446,2448] | +15 | yes | +0.2 | -537.3 | 2.04 | +16.0 | ✗ |

Rejected decoy candidates (their +15 MHz band contains JSON signals): train/1099:0 → c4@[2437.0,2477.0]MHz×[0.0,39.0]ms; train/1058:1 → c9@[2464.0,2464.0]MHz×[0.0,40.0]ms;c9@[2465.0,2465.0]MHz×[0.0,40.0]ms; train/9:2 → c9@[2437.9,2438.1]MHz×[15.0,37.0]ms; train/21:4 → c8@[2461.0,2463.0]MHz×[41.0,97.0]ms; train/1037:1 → c6@[2457.5,2458.5]MHz×[9.0,20.0]ms;c13@[2459.9,2460.1]MHz×[0.0,20.0]ms.

## 3. Per-class/stratum summary

| group | n | pass | med in/out dB | med box-pk dB | med \|pk\| kHz | med occ MHz | med \|old\| MHz |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0/clean | 4 | 2 | +9.9 | +17.0 | 5002 | 16.38 | 10.6 |
| 0/overlap | 1 | 1 | +8.3 | +28.8 | 4525 | 1.42 | 15.5 |
| 1/clean | 4 | 3 | +10.3 | +16.4 | 5313 | 19.61 | 3.7 |
| 1/overlap | 1 | 0 | +2.4 | +7.1 | 4994 | 19.89 | 1.0 |
| 2/clean | 4 | 3 | +8.4 | +18.4 | 3740 | 14.88 | 12.8 |
| 2/overlap | 1 | 1 | +9.9 | +31.8 | 1 | 1.88 | 0.0 |
| 3/clean | 4 | 4 | +8.5 | +14.8 | 3754 | 35.89 | 24.2 |
| 3/overlap | 1 | 0 | n/a | n/a | 18961 | 6.99 | 1.0 |
| 4/clean | 4 | 3 | +9.3 | +16.6 | 14064 | 35.64 | 16.9 |
| 4/overlap | 1 | 0 | +5.7 | +24.6 | 5015 | 2.01 | 35.0 |
| 5/clean | 4 | 3 | +10.6 | +20.4 | 8445 | 24.90 | 10.8 |
| 5/overlap | 1 | 0 | n/a | n/a | 16000 | 0.68 | 4.0 |
| 6/control | 3 | 3 | +17.7 | +22.3 | 51 | 0.90 | 11.0 |
| 8/control | 3 | 3 | +21.2 | +24.8 | 433 | 1.90 | 14.0 |
| 10/clean | 4 | 4 | +12.0 | +15.5 | 1155 | 9.46 | 3.5 |
| 10/overlap | 1 | 0 | +4.5 | +28.4 | 2 | 0.07 | 10.0 |
| 11/clean | 4 | 4 | +11.5 | +17.1 | 2757 | 9.52 | 12.7 |
| 11/overlap | 1 | 1 | +23.1 | +29.3 | 8 | 9.05 | 10.0 |
| **ALL** | 46 | 35 | +10.2 | +17.7 | 3746 | 11.68 | 11.0 |

Pass by class: 0 3/5 · 1 3/5 · 2 4/5 · 3 4/5 · 4 3/5 · 5 3/5 · 6 3/3 · 8 3/3 · 10 4/5 · 11 5/5.  
Pass by stratum: clean 26/32 · control 6/6 · overlap 3/8.

## 4. Facts vs inferences

### Facts (measured)

- Processed 46/46 real frames plus 4 synthetic-noise and 4 decoy controls in 4 s; analysis windows: all 46 are 10.0 ms (GT durations are all ≥ 10 ms in this set).
- `target_at_label` passes 35/46 frames overall (clean 26/32, overlap 3/8, control 6/6).
- Raw in/out dB per class (medians): 0 8.29 · 1 8.55 · 2 9.86 · 3 8.48 · 4 8.28 · 5 10.62 · 6 17.73 · 8 21.23 · 10 11.32 · 11 13.36; overall median 10.18 dB. No evaluable frame has a NEGATIVE in/out dB (no labeled band is emptier than its surroundings).
- Corrected peak offset: overall median |pk| = 3746 kHz from DC; occupied width overall median 11.68 MHz. Controls land at |pk| 65 kHz median with occupied ≈ their true bw; target classes sit MHz away from box center and occupy a large fraction of the keep band.
- Synthetic white noise through the identical chain: in/out dB median -0.01 dB, in-box peak only 2.25 dB above the out-median, occupied median 15.3 MHz ≈ keep band, 0/4 pass.
- Real decoy bands (JSON-clear 4/4): in/out dB -3.55…+0.19, energy keep ratio median 0.01, 0/4 pass — a real-capture "nothing there" reference: at or below the surrounding floor.
- OLD wrong demod: the dominant peak lands within bw/2+0.5 MHz of ±B/2 in 18/46 frames — and this set coincides (verified) with exactly the frames whose top-1 raw peak is INSIDE the box. In the other 28 frames the strongest capture energy is outside the box and the old-demod peak simply follows it.
- 2 frames have a box equal to the whole observation (train/1021:3, train/1016:0): in/out dB is not evaluable there by construction (both still have their top-1 raw peak inside the box).

### Inferences (interpretation — cross-check with the other phase-A' tracks before relying on them)

- Where `target_at_label` passes, the labeled band genuinely concentrates energy relative to its surroundings. This is a statement about ENERGY PRESENCE only; phase A already established that the labeled modulations (WiFi OFDM, SRRC single-carrier) are absent, and nothing here supports or requires modulation content.
- The old round-1 demod displaced the labeled content to the ±B/2 edge of the digital band (fact above). Round-1 per-frame statements about "box content" therefore described a band shifted B/2 from the label, not the label band; the round-1 census is unreliable at frame level. Where its class-level findings agree with this audit, they do so despite the geometry error, not because of it.
- The labeled bands are NOT empty: every evaluable frame has an in-box peak ≥ 7.1 dB above its out-of-box median, and no in/out dB is negative. But in 28/46 frames the single strongest peak of the whole observation window lies OUTSIDE the labeled box — these captures are crowded, and the label is usually not the dominant energy.
- The verdict failures are dominated by median-contrast failures: narrow tone-like content at/near the label with a box MEDIAN close to the capture floor (see §5 taxonomy). 2 frame(s) carry no significant content at the label at all — the closest thing to "JSON band empty, energy elsewhere" in this sample set.
- The decoy/noise controls define the "nothing there" signature: in/out ≈ 0 or below, keep ratio ≈ keep_bw/B, occupied ≈ full keep band, no in-box peak dominance. No target frame matches this signature completely; the failures differ from it mainly by having a narrow in-box peak above the floor.

## 5. Anomalies (verdict failures, taxonomy)

- **Not evaluable** (box = whole observation, no out-region; verdict false by rule): train/1021:3, train/1016:0 — both have their top-1 raw peak INSIDE the box.
- **Narrow tone at/near the label, box median at the floor** (in-box peak ≥ 10 dB above out-median but in/out < 6 dB): train/1024:0, train/1153:3, train/1425:2, train/1980:3, train/103:2, train/10:3, train/1011:0.
- **No significant content at the label** (best in-box peak < 10 dB above out-median AND in/out < 6 dB; dominant energy elsewhere): train/1620:1, train/1021:1.

- `train/1024:0` class 0 (clean): in/out +5.8 dB, box peak +15.5 dB above out-median; top-1 raw peak at 2475.09 MHz (OUTSIDE the box).
- `train/1153:3` class 0 (clean): in/out +4.7 dB, box peak +11.4 dB above out-median; top-1 raw peak at 2473.70 MHz (OUTSIDE the box).
- `train/1620:1` class 1 (clean): in/out +4.5 dB, box peak +8.2 dB above out-median; top-1 raw peak at 2436.17 MHz (OUTSIDE the box).
- `train/1021:1` class 1 (overlap): in/out +2.4 dB, box peak +7.1 dB above out-median; top-1 raw peak at 2433.01 MHz (OUTSIDE the box).
- `train/1425:2` class 2 (clean): in/out +1.3 dB, box peak +17.9 dB above out-median; top-1 raw peak at 2444.01 MHz (OUTSIDE the box).
- `train/1021:3` class 3 (overlap): in/out n/a dB, box peak n/a dB above out-median; top-1 raw peak at 2433.00 MHz (inside the box) — not evaluable (out-region empty).
- `train/1980:3` class 4 (clean): in/out +4.6 dB, box peak +17.5 dB above out-median; top-1 raw peak at 2472.05 MHz (OUTSIDE the box).
- `train/103:2` class 4 (overlap): in/out +5.7 dB, box peak +24.6 dB above out-median; top-1 raw peak at 2437.02 MHz (inside the box).
- `train/10:3` class 5 (clean): in/out +3.7 dB, box peak +21.0 dB above out-median; top-1 raw peak at 2407.03 MHz (OUTSIDE the box).
- `train/1016:0` class 5 (overlap): in/out n/a dB, box peak n/a dB above out-median; top-1 raw peak at 2436.00 MHz (inside the box) — not evaluable (out-region empty).
- `train/1011:0` class 10 (overlap): in/out +4.5 dB, box peak +28.4 dB above out-median; top-1 raw peak at 2437.00 MHz (inside the box).

test split untouched; no models trained; raw IQ/JSON unmodified.
