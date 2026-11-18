# Merge comparison: both vs fine-0-5 filtered

Protocol: `docs/collaboration/README.md`, commit `67a453a`.
Fixed `best_corrected.pt`, full 1543-sample val, one all-view inference.
Per-view NMS (`conf=0.001`, `iou=0.5`, `max_det=300`) and internal-boundary
filter, then before global merge:

- **both**: all coarse + all fine predictions.
- **filtered**: all coarse + fine predictions whose predicted class is `6–13`
  (fine predictions of class `0–5` are removed).

Both use the same class-wise global NMS (IoU 0.5) and 100-box cap.

Artifact: `runs/yolo11s_c1_directreg_2048x512/stage5_val/fine_0_5_filter_comparison.json`
(server-local). Script: `scripts/fine_0_5_filter_comparison.py`.

## Reproduction

| metric | baseline `stage5_val` | both | filtered |
| --- | ---: | ---: | ---: |
| mAP50 | 0.6427009 | 0.6427025 | **0.6497250** |
| n_pred | 65,025 | 65,027 | 47,783 |

The both group reproduces the baseline to +1.7e-6 mAP / +2 boxes, the same
numerical chunking difference as the earlier view attribution.

## Overall

| metric | both | filtered | delta |
| --- | ---: | ---: | ---: |
| mAP50 | 0.6427025 | **0.6497250** | **+0.00702 (+0.70 pp)** |
| AP50 agnostic | 0.9045925 | 0.9059753 | +0.00138 |
| R_agn | 0.9578330 | 0.9576934 | -0.00014 |
| n_pred | 65,027 | 47,783 | -17,244 |
| TP / FP / FN | 6489 / 58538 / 673 | 6492 / 41291 / 670 | +3 / **-17,247** / -3 |
| FP misclass / duplicate / background | 5242 / 287 / 53009 | 4864 / 288 / 36139 | -378 / +1 / **-16,870** |
| FP@0.5 misclass / duplicate / background | 959 / 10 / 315 | 959 / 10 / 315 | 0 / 0 / 0 |

The filter removes **low-score background FPs** (16,870) and some low-score
misclass FPs (378); it does **not** change the high-score FP counts.

## Per group

| group | AP both | AP filtered | ΔAP | R_agn both/filtered | FP both | FP filtered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0/1/2 | 0.4020 | **0.4212** | +0.0192 | 0.9699 / 0.9699 | 15,970 | 2,777 |
| 3/4/5 | 0.4415 | **0.4550** | +0.0135 | 0.9661 / 0.9661 | 5,787 | 924 |
| 10/11 | 0.6395 | 0.6395 | 0.0000 | 0.9721 / 0.9721 | 6,934 | 7,072 |
| narrow 9/12/13 | 0.7805 | 0.7805 | 0.0000 | 0.9162 / 0.9167 | 21,039 | 21,538 |

- 0/1/2 and 3/4/5 AP now equal the **coarse-only** AP (0.4212 / 0.4550).
- 10/11 and narrow AP are unchanged; their FP counts rise slightly because
  removing 0–5 boxes frees top-100 slots for more low-score 10/11/narrow boxes,
  but this does not change their AP.
- High-score misclass remains unchanged: 0/1/2 464, 3/4/5 183, 10/11 273;
  total high-score FP 959 misclass / 10 duplicate / 315 background.

## Removed candidates

18,944 fine-view 0–5 predictions were removed before global merge:

| class | removed |
| ---: | ---: |
| 0 | 3,897 |
| 1 | 4,649 |
| 2 | 5,274 |
| 3 | 2,352 |
| 4 | 1,298 |
| 5 | 1,474 |
| **0/1/2** | **13,820** |
| **3/4/5** | **5,124** |

The filter is safe: it recovers 3 additional TPs and 3 FNs at the final 100-cap,
and it removes 17,247 FPs.

## Judgment

1. **Adopt the fine-0-5 filter as a deterministic post-processing rule.** It is
   a free +0.70 pp total mAP, with 0/1/2 +1.92 pp and 3/4/5 +1.35 pp; no AP loss
   for 10/11 or narrow. It makes the coarse-only 0–5 proposals the working set.
2. **It is a precision cleanup, not a classifier.** High-score misclass FPs and
   wrong-class GT are unchanged: 530/997 (0/1/2), 203/442 (3/4/5), 240/716
   (10/11) GT are covered by the wrong class; oracle-class AP is 0.83–0.87
   versus 0.42–0.46 actual. The classifier remains the main lever.
3. **10/11 must not be blanket-filtered.** Fine contributes narrow 10/11 TPs;
   use bandwidth/source gating later.

## Classification review proposal

### Priority

1. **0/1/2** (20 MHz): 530/997 wrong-class GT, 464 high-score misclass FPs.
2. **3/4/5** (40 MHz): 203/442 wrong, 183 high-score misclass.
3. **10/11** (10 MHz, mixed with 0.3/1.6 MHz): 240/716 wrong, 273 high-score
   misclass; only after 0–5, with bandwidth/source gating.
4. **Not** 6/7/8/9/12/13: near their oracle ceiling; 12/13 are tiny-box
   localization, not classification.

### Proposal set (first round)

- Freeze `best_corrected.pt`.
- Use **coarse-view detector proposals** for 0–5; do not use fine 0–5 proposals
  (they cannot contain a complete 20/40 MHz box; the filter experiment confirms
  they add only FPs).
- On **fit**, run the detector, keep coarse proposals; match to GT at IoU>=0.5
  for class labels; include hard negatives: unmatched high-score proposals and
  proposals overlapping a GT of a different class at IoU>=0.5. Balance the
  confusion families {0,1,2} and {3,4,5}.
- On **val**, use the detector's actual coarse proposals, not GT boxes.

### Classifier / input

- Start with a small CNN (ResNet18 or a 4-block CNN), input 128×256 or 256×256,
  a 6-class + background head for 0–5 (or separate 3-way family heads).
  Class-balanced sampling, focal/CE, hard-negative mining.
- Evaluate proposal accuracy/confusion and end-to-end mAP after replacing or
  fixed-fusing the detector class scores; keep the same boxes, class-wise NMS
  and 100 cap. Report high-score misclass, group AP and total mAP.

### Coarse crop vs native STFT crop (same proposal box)

**Yes, this is worth a controlled comparison, but as the second classifier
ablation, not the first.**

- First train the classifier on **coarse crops**; this matches the detector
  input and is cheap. If it already fixes most 0–5 confusion, native crop is
  unnecessary.
- If coarse-crop confusion remains, repeat with **native STFT crops using the
  same proposal box and the same physical context**. This isolates whether the
  coarse frequency aggregation is the information bottleneck.

Measured crop sizes/costs on a B=80/T=150 sample (10% context margin):

| box | coarse crop | native crop | coarse crop+resize | native crop+resize |
| --- | ---: | ---: | ---: | ---: |
| 20 MHz | 616×286 | 16,800×286 | 0.15 ms | 0.15 ms (128×256) / 0.19 ms (256×256) |
| 40 MHz | 1,230×170 | 33,601×170 | 0.15 ms | 0.15 ms (128×256) / 0.27 ms (256×256) |

ResNet18 forward (batch 64): 0.07 ms/proposal at 128×256, 0.15 ms at 256×256.
Extra inference for 65k proposals is roughly 15–25 s, small versus the full
inference. The native crop is **7–27× taller** than the coarse crop; a naive
same-size resize compresses it by 100–260× and may erase the detail that makes
the comparison interesting. For a fair test, use a larger input (e.g., 256×256
or 512×512), a 1D frequency profile branch, or a two-branch classifier
(coarse thumbnail + native profile), and record input size/cost.

Use the same proposal boxes, physical context, architecture, fit/val split and
fusion rule for both crops. Otherwise the comparison is confounded.

### Expected decision

- If coarse-crop classifier improves 0/1/2 and 3/4/5 group AP and high-score
  misclass, keep it and extend to 10/11 with bandwidth-aware proposals.
- If coarse-crop still confuses the families, the native crop/profile
  comparison is justified; if neither helps, move to a larger model or a
  targeted pairwise loss.
- The fine-0-5 filter is independent and should be adopted regardless.

Test remains sealed.
