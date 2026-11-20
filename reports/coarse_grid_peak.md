# Grid-level coarse mean vs intra-band peak (2048x512)

Read-only. Same 0.7 ms STFT and the exact C1 coarse segmentation
(linspace row bounds, reduceat mean; the training pipeline only adds a fixed
global monotone dB scaling, so `mean` below is information-identical to the
channel the model sees). The three grids are computed first (images), then GT
boxes define evaluation regions restricted to each target's annotated time span.
Region score = max cell value inside the annotated extent (log10 power).
Background blocks are same-shape, same-B/T, empty of GT and B narrow predictions;
B's own narrow false positives are the second control.
Percentile calibration is per B/T against background; the equal-FP threshold is the
5% background false-positive budget. joint = p_mean + p_peak.

## Overall

- narrow GT: 2160 (both 1437, c_only 542, b_only 4, neither 177)
- background blocks: 5456; B narrow FPs: 2461
- c_only AUC vs background: mean 0.986, peak **0.995**, joint 0.996
- c_only peak AUC vs B narrow FPs: **0.929**
- c_only whose region-mean beats **every** same-shape bg block (p_mean = 1.0): **515/542**; both: 1349/1437
- at the 5% bg threshold, fraction of B narrow FPs passing: mean 0.474, peak 0.440, joint 0.458
- box height (rows): both h_med 12.0, c_only h_med 3.0, neither h_med 5.0; width (cols) both 307.0 vs c_only 272.0

## c_only mean-percentile histogram (vs same-B/T same-shape background)

| p_mean bin | c_only n | both n |
| --- | ---: | ---: |
| 0.00-0.50 | 0 | 0 |
| 0.50-0.75 | 0 | 0 |
| 0.75-0.90 | 0 | 1 |
| 0.90-0.97 | 2 | 5 |
| 0.97-1.00 | 25 | 82 |
| >=1.0 | 515 | 1349 |

## Hard c_only (p_mean < 1.0): n = 27

- by B: {'30': 7, '40': 1, '50': 4, '80': 15}; by class: {'9': 4, '12': 18, '13': 5}
- peak AUC vs background for this subset: 0.929 (mean AUC 0.888); peak beats every bg block for 12/27
- h_med 1.0 rows

## By observation bandwidth

| B | narrow GT | c_only | mean AUC | peak AUC | joint AUC | recall mean | recall peak | recall joint | peak AUC vs FP | c_only h_med | centroid row→cent err (MHz) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 197 | 9 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 22.0 | 0.011 → 0.001 (0.81 better than row) |
| 20 | 427 | 67 | 1.000 | **1.000** | 0.994 | 1.000 | 1.000 | 1.000 | 0.992 | 6.0 | 0.010 → 0.001 (0.85 better than row) |
| 30 | 388 | 82 | 0.999 | **1.000** | 0.996 | 1.000 | 1.000 | 1.000 | 0.977 | 4.0 | 0.011 → 0.001 (0.91 better than row) |
| 40 | 413 | 108 | 1.000 | **1.000** | 0.997 | 1.000 | 1.000 | 1.000 | 0.943 | 4.0 | 0.011 → 0.001 (0.92 better than row) |
| 50 | 365 | 111 | 0.999 | **0.999** | 0.997 | 1.000 | 1.000 | 1.000 | 0.897 | 3.0 | 0.011 → 0.002 (0.95 better than row) |
| 80 | 370 | 165 | 0.999 | **0.999** | 0.998 | 1.000 | 0.994 | 1.000 | 0.889 | 2.0 | 0.013 → 0.002 (0.92 better than row) |

## B narrow FPs passing the 5% background threshold

| B | n FP | pass mean | pass peak | pass joint |
| --- | ---: | ---: | ---: | ---: |
| 5 | 525 | 0.242 | 0.234 | 0.240 |
| 20 | 535 | 0.518 | 0.465 | 0.505 |
| 30 | 291 | 0.454 | 0.436 | 0.436 |
| 40 | 251 | 0.474 | 0.422 | 0.438 |
| 50 | 364 | 0.610 | 0.569 | 0.580 |
| 80 | 495 | 0.586 | 0.549 | 0.570 |

## Mean-matched peak separation (all B)

| mean percentile bin | c_only n | bg n | peak AUC | peak recall @5% bg FP |
| --- | ---: | ---: | ---: | ---: |
| 0.00–0.25 | 0 | 1346 | n/a | n/a |
| 0.25–0.50 | 0 | 1357 | n/a | n/a |
| 0.50–0.75 | 0 | 1369 | n/a | n/a |
| 0.75–1.00 | 27 | 1351 | 0.844 | 0.296 |

## 80 MHz

- c_only 165, background 1109, B narrow FP 495
- AUC vs background: mean 0.999, peak **0.999**, joint 0.998
- peak AUC vs B narrow FP: **0.889**

Mean-matched 80 MHz:

| mean percentile bin | c_only n | bg n | peak AUC | peak recall @5% bg FP |
| --- | ---: | ---: | ---: | ---: |
| 0.00–0.25 | 0 | 273 | n/a | n/a |
| 0.25–0.50 | 0 | 276 | n/a | n/a |
| 0.50–0.75 | 0 | 278 | n/a | n/a |
| 0.75–1.00 | 15 | 276 | 0.967 | 0.933 |

## Centroid validation (frequency error vs coarse row centre, MHz)

Using the max-energy row centre as the frequency estimate vs adding the intra-band
energy centroid (energy-weighted, restricted to the target's annotated time columns):

| set | n | row err med | energy-weighted err med | centroid err med | centroid err p90 | centroid better than row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all narrow | 2160 | 0.011 | 0.002 | 0.001 | 0.010 | 0.90 |
| c_only | 542 | 0.006 | 0.003 | 0.002 | 0.010 | 0.85 |

For reference, one coarse row spans B/2048 MHz (e.g. 0.039 MHz at 80 MHz).


## Example images

- `/root/autodl-tmp/worktrees/baseline-0-group-a/reports/coarse_grid_peak_examples/example1_train_19.png`
- `/root/autodl-tmp/worktrees/baseline-0-group-a/reports/coarse_grid_peak_examples/example2_train_25.png`
- `/root/autodl-tmp/worktrees/baseline-0-group-a/reports/coarse_grid_peak_examples/example3_train_37.png`

Per-B/T thresholds are in the JSON; raw per-record values in
`reports/coarse_grid_peak_raw.json`.
