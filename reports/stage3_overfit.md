# Stage 3 - 16-source overfit gate

Artifacts:
- `runs/overfit16/` (first segment, 0-400 updates; best 0.7968 at 400)
- `runs/overfit16b/` (continuation from the 400-update checkpoint; pass at 600)
- `runs/overfit16b/overfit_history.json`, `overfit_summary.json`
- `runs/overfit16b/best_overfit.pt` (strict mAP@0.5 0.8168)

## Selection

16 fit sources covering all 14 classes, B in {5, 20, 30, 40, 50, 80} MHz and
T in {20, 40, 80, 100, 150} ms; same-class GT duplicates were screened out.
The exact IDs and per-sample B/T/class counts are in
`runs/overfit16b/overfit_config.json`; the split manifest and normalizer hashes
are recorded there too.

## Training definition

- same per-source sampling as stage 4: 1 coarse + 1 rotating fine view;
- physical 2 sources (4 views) x accumulation 4 = 16 views per update;
- BF16 autocast, AdamW lr0=1e-3, wd=5e-4, clip 10, cosine schedule;
- at most 2000 updates; validation on all fine+coarse views of the 16 sources;
- target: strict class-aware mAP@0.5 >= 0.8.

## Trajectory

| update | mAP@0.5 | AP50 agnostic | localization recall | n pred | val time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.6184 | 0.6852 | 0.724 | 1202 | 8 s |
| 400 | 0.7968 | 0.7797 | 0.797 | 849 | 4 s |
| 500 | 0.7865 | 0.7721 | 0.797 | 924 | 8 s |
| 600 | **0.8168** | 0.7898 | 0.813 | 569 | 4 s |

Loss (25-update mean) fell from 66.1 at update 25 to 6.5 at update 600.
The run passed at update 600 and stopped early.

## Per-class AP at the passing checkpoint

| class | AP@0.5 | GT | pred |
| ---: | ---: | ---: | ---: |
| 0 | 0.881 | 9 | 58 |
| 1 | 0.802 | 5 | 13 |
| 2 | 1.000 | 4 | 21 |
| 3 | 0.753 | 4 | 14 |
| 4 | 1.000 | 4 | 13 |
| 5 | 1.000 | 4 | 16 |
| 6 | 0.907 | 13 | 38 |
| 7 | 0.989 | 13 | 44 |
| 8 | 0.988 | 18 | 53 |
| 9 | 0.291 | 15 | 72 |
| 10 | 0.871 | 8 | 64 |
| 11 | 1.000 | 8 | 31 |
| 12 | 0.555 | 9 | 54 |
| 13 | 0.399 | 9 | 78 |

Classes 9/12/13 remain the weak points on this small subset; this is recorded
rather than hidden. 6,001 fine-view boxes were dropped as internal-boundary
candidates over the 324 validation views, per the fixed inference rule.

**Verdict: PASS** (0.8168 >= 0.8). This is a pipeline/loss/coordinate sanity
gate on 16 sources, not a claim about full-validation accuracy.
