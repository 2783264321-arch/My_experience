# Complex coarse front end: independent review and candidate decision

Scope: front-end only. Reuses the previous 14-sample IQ set, adds controlled
band-edge tones, same-band dual tones, and cross-window short bursts. No
detector training; test sealed; existing 0.649725 control unchanged.

## 1. Independent review of the two-candidate framing

The two candidates are useful, but they are **not the same kind of operator**:

- **Sparse 2048**: a direct 25.6 us (at 80 MHz) short-window complex STFT. It
  is a clean narrowband measurement of the complex coefficient at each of the
  512 frame centers. Its cost is time gaps between centers and a 2048-row
  full-band frequency grid that is 6.8-27.3x coarser than the fine grid.
- **0.7 ms STFT -> 2048 overlapping complex bands**: a long-window,
  frequency-filtered complex envelope. It has 100% time coverage and no
  band-boundary null (with 50% overlap), but each band is an effective
  ~2-band-wide filter, so it has more in-band phase cancellation and a longer
  time response. It is not a drop-in short-window coefficient.

The previous report used sum-normalized single-band Hann weights; that made
the comparison look like a narrowband tone loss. This round uses a **unit-gain
50%-overlap linear-Hann filter bank** for the complex candidate, and a
sum-normalized version only for the 0-1 coherence diagnostic. The old
single-Hann numbers are retained only as a filter-shape baseline.

Two corrections to the previous evidence:

- the missed-bin fraction is **157 / 17,619 GT-interval columns = 0.89%**, not
  157/38,912 = 0.40%;
- the 2.94 ms boundary p90 is **search-limited** (30-35 support estimates hit
  the 10-column outward search limit), so it is not a true boundary error;
  known-boundary synthetic bursts are used here. The raw train minimum
  annotation is 1 ms (the 76-instance subset minimum is 3 ms), not 4 ms.

## 2. Controlled tests

All synthetic tests use `Fs=80 MHz`; the overlap filter bank uses 50%
overlapping linear-Hann bands with unit gain at the band center. Complex
amplitudes are coherent-scale normalized; power is the same-weight filtered
power. Values are normalized to the band-center response of each candidate
when shape is compared.

### 2.1 Band-center, band-boundary and observation-edge tones

Normalized adjacent responses (band b-1 / b / b+1); the center reference is
the 5.019531 MHz tone at the band center.

| tone | sparse | overlap | single Hann | rectangular |
| --- | --- | --- | --- | --- |
| band center 5.0195 MHz | 0.20 / 1.00 / 1.00 | 0.00 / 1.00 / 0.002 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 |
| boundary 5.000 MHz | 0.00 / 0.589 / 1.178 | **0.00 / 0.473 / 0.530** | 0.00 / 0.00 / 0.004 | 0.00 / 0.25 / 0.75 |
| boundary 5.039 MHz | 0.00 / 0.589 / 1.178 | **0.00 / 0.473 / 0.530** | 0.00 / 0.00 / 0.007 | 0.00 / 0.119 / 0.881 |
| near edge 39.5 MHz | 0.417 / 1.148 / 0.765 | 0.185 / 0.818 / 0.00 | 0.00 / 0.330 / 0.00 | 0.00 / 1.00 / 0.00 |
| observation edge 39.9 MHz | 0.238 / 1.038 / 0.958 | 0.006 / 0.997 / 0.00 | 0.00 / 0.992 / 0.00 | 0.00 / 1.00 / 0.00 |
| observation edge -39.9 MHz | 0.165 / 0.958 / 1.038 | 0.00 / 0.986 / 0.017 | 0.00 / 0.949 / 0.00 | 0.00 / 1.00 / 0.00 |

The overlap candidate is the only one whose **sum of adjacent normalized
responses stays ~1.00** at band boundaries and observation edges: 1.003 at
center, 1.002 at both 5 MHz boundaries, 1.002 at ±39.9 MHz. Single Hann has a
near-null at the 5 MHz boundary (sum 0.004-0.007) and attenuates near some
edges (0.223 at -39.5 MHz, 0.330 at 39.5 MHz). Rectangular splits a boundary
tone unevenly (0.25/0.75). Sparse has no null but leaks across several rows
(adjacent magnitude sum 1.8-2.3), so it is not a clean single-row measurement
at a boundary.

### 2.2 Same-band dual tones

Two tones at 5.019531 MHz +/- 2.5 kHz, i.e. inside one coarse band; relative
phase 0, pi/2, pi. The complex amplitude at the center column:

| phase | sparse | overlap | single Hann | rectangular | overlap power | rect power |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.591 | 3.708 | 3.543 | 3.755 | 0.001379 | 0.001396 |
| pi/2 | 0.715 | 1.659 | 1.590 | 1.680 | 0.001387 | 0.001404 |
| pi | 0.588 | 1.362 | 1.306 | 1.379 | 0.001387 | 0.001404 |
| max/min | 2.71x | 2.72x | 2.71x | 2.72x | 1.006x | 1.006x |

**Every complex candidate has the same ~2.7x phase-dependent amplitude** for
same-band tones. The overlap filter does not fix in-band cancellation; it
changes the band shape, not the fundamental complex-sum behavior. Power is
stable to <1%. A 2-channel complex coarse therefore cannot be the sole
energy/context view for a general detector.

### 2.3 Cross-window short bursts

80 MHz, 150 ms, burst at 5.019531 MHz; `h=0.294 ms`; offset 0 = on a frame
center, offset h/2 = worst gap. Support is the 10%-of-peak column range.

| duration / offset | sparse | dense 2048 power | overlap power | fine 0.7 ms power |
| --- | --- | --- | --- | --- |
| 0.1 ms / center | detected, [256] | detected, [256] | detected, [256] | detected, [256] |
| 0.1 ms / mid-gap | **not detected (SNR 0.1)** | detected, [257] | detected, [256-257] | detected, [256-257] |
| 0.3 ms / center | detected, [256-257] | detected, [256-257] | detected, [256-257] | detected, [256-257] |
| 0.3 ms / mid-gap | detected, [257] | detected, [257] | detected, [257] | detected, [257] |
| 1.0 ms / center | detected, [256-259] | detected, [256-259] | detected, [256-259] | detected, [256-259] |
| 1.0 ms / mid-gap | detected, [257-259] | detected, [257-260] | detected, [257-260] | detected, [257-260] |

Boundary errors for the mid-gap cases:

- 0.1 ms: sparse fails; dense start +0.147 ms, end +0.047 ms; overlap/fine
  start -0.147 ms, end +0.047 ms.
- 0.3 ms: all support [257]; start +0.147 ms, end -0.153 ms.
- 1.0 ms: sparse start +0.147, end -0.266 ms; dense/overlap/fine start +0.147,
  end +0.027 ms.

The sparse gap is real for a 0.1 ms burst; for >=0.3 ms it detects but can lose
~0.27 ms at the end boundary. The 0.7 ms overlap detects the short burst but
**smears it like the fine window**; its time response is not sharper. Dense
2048 power detects all and has the shortest window, so it is the best
short-burst/time-boundary power view. The overlap's advantage is band-edge
reconstruction and 100% time coverage, not time sharpness.

### 2.4 Real IQ

The previous 14-sample set (76 GT instances) still shows:

- sparse/2048 GT coverage 8.7% at 80 MHz/150 ms, 17.4% at 40/150, 34.9% at
  20/150, 65.4% at 80/20, 100% at 20/20;
- missed bins 157/17,619 = 0.89%, worst 12.1% for bursty narrowband at
  20 MHz/150 ms;
- sparse/dense power ratio median 0.998, so continuous-signal energy is not
  badly biased;
- boundary support is a visibility check, not a clean boundary estimate;
  synthetic bursts above provide the boundary evidence.

For the 50%-overlap complex bands on the same real IQ, the sum-normalized
coherence (0..1) is median **0.145**, p10 0.073, p90 0.284. The narrower
single-band Hann is 0.303, rectangular 0.184, dense 2048 complex 0.222. The
overlap filter doubles the effective band width, so it has more in-band
cancellation than the single-band Hann; this is the price of its better
band-edge response. The values are not a like-for-like bandwidth comparison;
they show the direction of the trade-off.

## 3. What each choice preserves and discards

| construction | preserves | discards / costs |
| --- | --- | --- |
| sparse 2048 Re/Im | clean short-window complex coefficient; full observation band; exact 512 centers; lowest cost (~0.05 s/sample at 80/150); complex power | time gaps; 0.1 ms mid-gap bursts can be missed; 6-12% weak bins for bursty narrowband at 150 ms; 25.6 us snapshot has little modulation detail |
| 0.7 ms 50%-overlap Re/Im | 100% time coverage; same time atom as fine; band-edge reconstruction (adjacent sum ~1); ~0.1 s pooling on existing STFT | effective band twice as wide; lower complex coherence (~0.145); same-band phase cancellation ~2.7x; 0.7 ms time smearing; 2048 bands discard within-band frequency degrees of freedom |
| **3-channel power + overlap Re/Im** | explicit energy channel + phase; 100% coverage; band-edge reconstruction; same STFT/weights/time centers for all channels | C=3 instead of 2; complex channel still cancels in-band, but power carries the energy; 0.7 ms time response |
| **3-channel power + sparse Re/Im** | explicit energy channel + clean short-window phase; low cost; complex power preserved | mixed time atoms (0.7 ms power vs 25.6 us complex); complex gaps/short-burst blind spots remain |

## 4. Decision

**My choice: do not make either 2-channel complex coarse the primary front end.
Use a 3-channel coarse: power + Re/Im, all from the same 0.7 ms STFT and the
same 50%-overlap band weights/time centers. Keep 3-channel existing-power +
sparse Re/Im as the low-cost control.**

Reasons:

1. **Same-band dual tones settle the 2-channel issue.** All complex candidates
   have ~2.7x phase-dependent amplitude; power is stable. A 2-channel complex
   coarse cannot represent same-band energy/multi-tone context without
   cancellation. A power channel is not optional for a 14-class detector.
2. **Overlap fixes the band-edge problem, not the cancellation problem.** Its
   adjacent-band sum stays ~1.00 at boundaries and observation edges, where
   single Hann has a near-null and rectangular splits unevenly. But its
   effective band is twice as wide and its normalized coherence is lower
   (0.145 vs 0.303), and same-band dual tones still cancel. Pairing it with
   power is the honest use of this operator.
3. **Sparse remains the cheapest clean complex measurement, but its gaps are
   real for short bursts and bursty narrowband signals.** It is a good control
   and a plausible fallback if training cost dominates, but it is not the best
   primary for a general all-class detector.
4. **The 3-channel overlap front end is physically consistent:** power, Re, Im
   all come from the same 0.7 ms STFT, the same 512 frame centers and the same
   linear-Hann band weights. Power is `sum(w*|z|^2)` (a band energy), not
   `|sum(w*z)|^2`; Re/Im is `sum(w*z)`, the band complex envelope. The extra
   cost is one input channel and about 0.1 s/sample of pooling, not an extra
   2048-point STFT.

If the project requires exactly two channels, choose **sparse 2048 Re/Im** as
the cleaner complex measurement and keep the existing power coarse as a
separate energy view; do not choose 2-channel overlap as a free upgrade. If the
project requires exactly two channels but allows different physical quantities,
use **power + sparse Re/Im** (2 channels, but not two complex channels); this
is the smallest way to keep energy and phase without cancellation.

## 5. Minimal experiment to distinguish the choices

Use the same fit/val split, proposals, 512 frame centers, labels, detector body
and training budget. Four arms, all 2048x512:

1. C2 sparse Re/Im (current direct 2048 candidate);
2. C2 overlap Re/Im (0.7 ms 50%-overlap bands);
3. C3 overlap: power + Re/Im from the same 0.7 ms STFT;
4. C3 sparse: power (existing 0.7 ms coarse) + sparse 2048 Re/Im.

Primary comparisons: C3-overlap vs C2-overlap (does the power channel recover
same-band/bursty energy?), C3-overlap vs C3-sparse (does gap-free/band-edge
matter at equal C=3?), and C3-overlap vs the 0.649725 power baseline. Report
overall mAP@0.5, localization recall, 0/1/2 and 3/4/5 AP, 10/11 and narrow
classes, burst/overlap subsets, and input cost. A small 10-20 epoch run on a
fixed subset is enough to reject a front end; only a stable paired gain should
trigger a full run.

Do not run the full training before this front-end control. Test remains
sealed.

Artifacts: `reports/complex_coarse_candidate_comparison.json`,
`scripts/validate_complex_coarse_candidates.py`,
`reports/complex_coarse_frontend_validation.md` (previous round, corrected).
