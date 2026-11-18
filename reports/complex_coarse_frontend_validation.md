# Complex coarse STFT front-end validation

> Follow-up: the candidate decision after band-edge/dual-tone/burst tests is in
> `reports/complex_coarse_decision.md`.

Question: can a 2048-point full-band complex STFT at the original IQ sample
rate share time-frequency coordinates and a complex channel definition with the
existing 0.7 ms complex fine slices? What do the short-window gaps do on long
samples, and is there a better construction?

Scope: front-end only. No detector training, no original IQ/JSON modification,
test sealed. Control remains the 0.649725 filtered full val.

## 1. What was checked

Representative fit samples: 14 samples covering 20/40/80 MHz observation and
20/40/150 ms durations, each with at least one wide (>=20 MHz) and one narrow
signal. 76 GT instances total. Sample IDs: `train/29, 68, 103, 14, 15, 108,
270, 130, 214, 33, 188, 151, 357, 342`. For every sample:

- native 0.7 ms complex STFT, `F x 512`, existing `h=ceil(N/511)`, right-pad to
  `511h`, `center=True`, periodic Hann, bilateral, fftshift;
- sparse 2048-point complex STFT, `2048 x 512`, same `h`, same padding and
  frame centers;
- dense 2048-point complex STFT, `hop=min(1024, max(256, h/2))`, then power
  pooled and phase-aligned complex pooled to 512 columns;
- 0.7 ms frequency-pooled complex coarse, rectangular and Hann-weighted, to
  2048 bands;
- synthetic complex tones at 20/80 MHz for phase/amplitude calibration.

All complex values use the existing PSD scale
`Z / sqrt(Fs * sum(w^2))`; a coherent `Z / sum(w)` scale is also reported for
Re/Im amplitude comparability. The fine power `|Z|^2` matches the existing
`stft_power_fast` to at most 4.3e-7 relative error over the 14 samples.

## 2. Coordinates: time yes, frequency global yes, bin centers no

**Time axis: exactly shareable.** Both views use `h=ceil(N/511)`,
`n_ext=511h`, `center=True`, and therefore 512 frame centers
`t_j = j*h/Fs` (j=0..511). The sparse 2048 STFT returns exactly 512 columns;
the max center difference is 0 by construction and by check. The same
normalized x label mapping applies.

**Frequency axis: same global physical coordinate, different sampling grid.**
Both map a physical frequency f to normalized `y=(f-f_lo)/B`, so labels can be
remapped with one convention. But the bin centers are not the same:

| observation B | fine rows F | fine spacing | coarse 2048 spacing | ratio F/2048 |
| ---: | ---: | ---: | ---: | ---: |
| 20 MHz | 14,000 | 1.429 kHz | 9.766 kHz | 6.84 |
| 40 MHz | 28,000 | 1.429 kHz | 19.531 kHz | 13.67 |
| 80 MHz | 56,000 | 1.429 kHz | 39.063 kHz | 27.34 |

A coarse row is not a fine row. The coarse view has 2048 rows over the full
band; a fine slice has 2048 rows over about 2.93 MHz, with a per-slice start
offset. The model must know the view scale and offset; the normalized label
convention is shared, the row index is not. Coarse edge quantization is half a
coarse row (4.9/9.8/19.5 kHz at 20/40/80 MHz), still small relative to 20/40 MHz
boxes, but much larger than the 0.71 kHz fine half-bin.

## 3. Channel definition: phase correction and amplitude scale matter

A raw STFT coefficient carries a window-center phase term proportional to the
**bin frequency** times the window half-length. Coarse 2048 and fine 0.7 ms
have different half-lengths, so raw Re/Im are not on a common phase reference.
For a complex tone at 20 MHz / 2 MHz, the nearest coarse bin is 2.001953 MHz
(0.2 bin off) and the raw phase difference relative to the fine coefficient was
**pi**; the coefficient was negative-real. This is not a signal property.

Apply the common frame-center correction

```text
Z_ref[k, j] = Z[k, j] * exp(+j 2 pi f_k (L/2) / Fs)
```

to both views. After correction the same synthetic tone has phase difference
**1e-8 rad** at 20 MHz, and 3e-9 rad at 80 MHz. The interior columns match;
edge columns (j=0/511) show small leakage effects. This correction is required
for a shared Re/Im channel definition.

**Amplitude scale.** PSD normalization makes power a spectral density, but a
tone's peak PSD scales with window length. At 80 MHz/5 MHz, coarse peak
amplitude was 0.00558 and fine 0.02916 (ratio 0.191 = sqrt(2048/56000));
coherent `Z/sum(w)` gives 1.34986 for **both** views. At 20 MHz the tone is
0.2 coarse bins off; coherent amplitudes were 1.315 vs 1.350 (2.6% low, expected
window leakage). Recommendation:

- use `Z / sqrt(Fs sum(w^2))` for the power branch, so `|Z|^2` matches the
  existing power view;
- use `Z / sum(w)` (with the frame-center phase correction) for Re/Im, so a
  unit tone has a comparable amplitude across views;
- if a single tensor must carry both, use PSD plus a fixed per-view scale
  estimated on fit and frozen; do not claim raw PSD Re/Im amplitudes are equal.

## 4. Long-sample gaps: coverage, missed bins, boundaries

The sparse 2048 window is 2048/Fs long. Its duty cycle relative to the 512
column spacing is measured below (union coverage of GT signal intervals):

| B/T | sparse coverage |
| ---: | ---: |
| 20 MHz / 20 ms | 1.000 |
| 80 MHz / 20 ms | 0.654 |
| 40 MHz / 40 ms | 0.654 |
| 20 MHz / 150 ms | 0.349 |
| 40 MHz / 150 ms | 0.174 |
| 80 MHz / 150 ms | **0.087** |

This is the fraction of signal **integration time** per column, not the
fraction of columns that are present. Every column still contains a short
window; for a continuous signal the column is bright even though the window
integrates only 8.7% of the column spacing. The detector sees 512 columns, not
holes. The real effect is snapshot variance and loss of within-column detail.

Measured on the 76 GT instances:

- sparse/dense power ratio median **0.998** (range 0.687-1.042), so the center
  snapshot is an approximately unbiased energy estimate for continuous signals;
- missed bins (dense bin energy >5x sparse) **157 / 17,619 GT-interval columns = 0.89%**;
  worst **12.1%** for one 20 MHz/150 ms BLE/Zigbee-like signal. The worst cases
  are bursty narrowband signals, not continuous WiFi;
- boundary errors (support threshold 10% of peak): median **0.039 ms** at
  20/40 ms and **0.294 ms** at 150 ms (one column). The 2.94 ms p90 is **not a
  true boundary error**: 30-35 support estimates hit the 10-column outward
  search limit, mostly from adjacent-signal/leakage energy. The real-IQ support
  metric is therefore only a visibility check; known-boundary synthetic bursts
  are needed for a clean boundary comparison (next round).

The raw train minimum annotation is 1 ms; the 76-instance subset minimum is
3 ms. At 150 ms the column spacing is 0.294 ms, so a 1 ms signal spans more than
three columns and cannot be entirely skipped, but an internal burst shorter
than the column spacing can be weakly sampled or missed.

So on long samples the short-window gaps do **not** break box boundaries or
wholesale signal detection for continuous signals; they reduce temporal
integration/detail and can miss 6-12% of bins for bursty narrowband signals.
A 2048-point window at 80 MHz is 25.6 us, about two WiFi OFDM symbols, versus
about 55 symbols in the 0.7 ms fine window; it is not a substitute for fine
modulation detail.

## 5. Cost

Median seconds per sample on the RTX 4090 host; Python loops for pooling are not
optimized.

| B/T | fine 0.7 ms | sparse 2048 | dense 2048 | dense pool | fine pool / Hann |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 80/150 | 0.444 | 0.049 | 0.408 | 0.561 | 0.098 |
| 40/150 | 0.209 | 0.030 | 0.199 | 0.471 | 0.095 |
| 20/150 | 0.089 | 0.006 | 0.086 | 0.293 | 0.076 |
| 80/20 | 0.367 | 0.005 | 0.027 | 0.244 | 0.101 |
| 20/20 | 0.096 | 0.004 | 0.011 | 0.233 | 0.076 |

The fine STFT always computes 512 long FFTs, so its cost is roughly constant
per sample. Sparse 2048 is negligible. Dense 2048 + power pooling costs about
0.95 s at 80 MHz/150 ms, comparable to or slightly above the fine STFT; at
20 MHz/150 ms it is about 0.38 s. Fine frequency pooling is about 0.1 s. A
vectorized/GPU pooling pass would reduce the pool column, not the STFT cost.

## 6. Complex pooling coherence

For each GT band/time, coherence is `|mean z|^2 / mean |z|^2` (1 = power
preserving, 0 = complete cancellation):

| coarse complex construction | median | p10 | p90 |
| --- | ---: | ---: | ---: |
| sparse 2048, no pooling | 1.000 | 1.000 | 1.000 |
| dense 2048, phase-aligned complex pool | 0.222 | 0.058 | 0.661 |
| 0.7 ms, rectangular frequency pool | 0.184 | 0.094 | 0.352 |
| 0.7 ms, single-band Hann pool | 0.303 | 0.150 | 0.591 |

Coherent complex pooling of a broadband signal loses 70-80% of the power; this
is physical phase cancellation, not a numerical bug. Power pooling preserves
energy and should be used for the energy/context view. Sparse 2048 preserves
complex power but has the time gaps. A single-band Hann filter also has a null
at band boundaries: a synthetic tone exactly on a boundary had Hann amplitude
0.00075 versus fine 1.35; rectangular pooling split it (0.075). A 50%-overlap
filter bank is needed for a complex coarse that is well behaved at boundaries.

## 7. Judgment

**Can they share coordinates and channels?** Time yes, exactly. Frequency only
in the global physical/normalized coordinate, not in bin centers: coarse 2048
full-band is 6.8-27.3x coarser than the fine grid. Channel definition yes, but
only after (a) the frame-center phase correction and (b) an explicit amplitude
convention: PSD for power, coherent gain for Re/Im (or a fixed per-view scale).

**Do long-sample gaps hurt?** For continuous WiFi-like signals, no catastrophic
boundary or detection effect: sparse/dense energy ratio ~1, missed bins 0.89%
overall, boundary median one column (0.294 ms at 150 ms). The real-IQ p90 is
search-limited, so it is not used as a boundary comparison; synthetic bursts
are needed. For bursty narrowband signals, up to 6-12% of time bins are missed at 150 ms,
and the 25.6 us snapshot cannot carry the 0.7 ms modulation detail. The sparse
2048 view is usable as a cheap localization/context view, not as the sole
coarse view for all classes.

**Better construction.**

1. **Preferred complex coarse:** keep the existing 0.7 ms complex STFT and form
   2048 complex bands with a **50%-overlap sqrt-Hann/RRC frequency filter
   bank**, after the frame-center phase correction. Physical basis: same time
   atom as the fine slices, 100% time coverage, common phase reference, and
   overlapping band filters avoid the single-band Hann boundary null. Cost:
   one fine STFT (already computed) + O(F*512) pooling, about 0.1 s/sample;
   no extra 2048 STFT. Minimal control: rectangular vs single-band Hann vs
   50%-overlap filter bank; measure tone response at band center/boundary,
   complex coherence, and detector AP on the same 512 grid.
2. **Preferred gap-free power coarse:** dense 2048 STFT (hop <= h/2) + power
   pooling to 512 columns. It preserves energy and has 100% time coverage; cost
   about 0.95 s/sample at 80/150, comparable to the fine STFT. Minimal control:
   sparse 2048 power vs dense 2048 power vs existing 0.7 ms power coarse.
3. **Do not use sparse 2048 as the only coarse**, and do not use single-band
   rectangular/Hann complex pooling as an energy view. Sparse is cheap and
   preserves complex power but is a 8.7%-duty snapshot at 80/150; coherent
   pooling loses 70-80% of broadband power.

**Recommended next step:** front-end control only, no full training yet. On the
same 512 grid and labels, compare (a) existing 0.7 ms power coarse, (b) dense
2048 power pooled, (c) 0.7 ms 50%-overlap complex coarse; use a frozen or small
detector and the current validation protocol. If the complex coarse is adopted,
use coherent Re/Im normalization and the phase correction; keep the power
branch on PSD. Test remains sealed.

Artifacts: `reports/complex_coarse_frontend_summary.json` (tracked),
`reports/complex_coarse_frontend.json` (full profiles, server-local),
`scripts/validate_complex_coarse_frontend.py`.
