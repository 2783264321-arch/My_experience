# Coarse-crop proposal classifier experiment (first round)

Protocol: `docs/collaboration/README.md`, commits `67a453a` / `b323bc5`.
Base: frozen `best_corrected.pt`, full 1543-sample val, with the shared
fine-0-5 filter enabled. First round: freeze the detector, train small
classifiers on **fit coarse proposals**, and at val change **only the class**
of the final 0-5 boxes; boxes and scores stay fixed, then re-run class-wise
NMS and the 100-box cap.

## 1. Shared fine-0-5 post-processing

The filter is now a switch in the common inference/validation path:

- `src/signals/infer.py::predict_sample(..., drop_fine_classes=...)`
- `src/signals/validate.py::run_validation(..., drop_fine_classes=...)`
- CLI: `scripts/validate_run.py --drop-fine-classes 0-5` (also exposed in
  `train.py`, `bench_inference.py`, `predict_examples.py`, `trace_sample.py`)

Full val reference: `stage5_val_filtered_shared`.

| metric | baseline | filtered shared |
| --- | ---: | ---: |
| mAP50 | 0.642701 | **0.649725** |
| AP50 agnostic | 0.904594 | 0.905977 |
| R_agn | 0.957973 | 0.957833 |
| n_pred | 65,025 | 47,784 |
| boundary dropped | 202,142 | 202,142 |

The filtered diagnostic script produced 0.6497250 / 47,783; the shared path
matches to 1.7e-6 mAP and +1 box (chunk numerical noise). This is the new
control for the classifier round.

## 2. Fit crop collection

- 3,418 fit samples: all B>=40 (2,618) plus 800 random B=20/30.
- Coarse view only; detector per-view NMS 300; each predicted box matched to
  its best GT at IoU>=0.5; up to 3 boxes per GT with GT class 0-5.
- Crop = coarse view box + 10% context, resize 128x256, float16.
- **7,978 crops**; 4,289 (53.8%) have detector predicted class != GT class,
  i.e. many "box correct, class wrong" examples are included.
- Class counts: 0=1,558; 1=1,620; 2=1,557; 3=995; 4=1,079; 5=1,169.
- Train/holdout split by source sample (not by crop): 012 = 4,278/457 crops
  (1,515/168 samples); 345 = 2,939/304 crops (1,234/137 samples).

## 3. Classifier training

Small CNN (4 conv blocks, BN/ReLU, 128x256 input, 3-class head), 20 epochs,
AdamW, class-weighted CE, no augmentation; holdout by sample.

| family | holdout acc | confusion (rows GT, cols pred) |
| --- | ---: | --- |
| 0/1/2 | **0.4311** | [[51,54,60],[41,49,50],[15,40,97]] |
| 3/4/5 | **0.4507** | [[52,16,4],[37,62,24],[56,30,23]] |

Both are close to chance (0.33) and far from the detector's group AP level.
Class 5 is especially weak (23/109 correct), class 0 only 51/165.

## 4. Val result (labels changed only; boxes/scores fixed)

| metric | filtered before | after classifier | delta |
| --- | ---: | ---: | ---: |
| mAP50 | **0.649725** | 0.589391 | **-0.0603** |
| AP50 agnostic | 0.905977 | 0.905551 | -0.0004 |
| R_agn | 0.957833 | 0.951689 | -0.0061 |
| n_pred | 47,784 | 46,083 | -1,701 |
| TP / FP / FN | 6493 / 41291 / 669 | 6049 / 40034 / 1113 | -444 / -1257 / +444 |
| FP@0.5 misclass / dup / background | 959 / 10 / 315 | 893 / 9 / 310 | -66 / -1 / -5 |

Group AP:

| group | AP before | AP after | delta |
| --- | ---: | ---: | ---: |
| 0/1/2 | 0.4212 | 0.2522 | **-0.1690** |
| 3/4/5 | 0.4550 | 0.3425 | **-0.1125** |
| 10/11 | 0.6395 | 0.6395 | 0 |
| narrow 9/12/13 | 0.7804 | 0.7804 | 0 |

The classifier changed **2,996 / 4,844** final 0-5 boxes. Change pairs:
0->1 374, 0->2 540, 1->0 334, 1->2 507, 2->0 233, 2->1 328,
3->4 101, 3->5 40, 4->3 194, 4->5 49, 5->3 177, 5->4 119.
It reduces high-score misclass only slightly (959 -> 893) but loses many TPs
and creates FNs; the fixed scores cannot compensate for wrong relabeling.

Extra time: crop + classifier + re-NMS **4.79 s** for 4,844 0-5 boxes
(~0.99 ms/box); full eval 70 s including the coarse views. Training was ~2 min
for both small CNNs.

## 5. Judgment

- **The first-round coarse-crop small classifier is not viable.** Holdout is
  near chance and val mAP drops 6.0 pp; do not adopt it.
- This is **not** proof that the coarse representation is at its ceiling: the
  classifier is small, underfit (train loss ~0.98/0.90), and changes every
  0-5 box without confidence gating. A stronger classifier or a confidence
  gate is needed before concluding.
- The fine-0-5 filter remains a free +0.70 pp and is independent of this
  negative classifier result.

## 6. Native STFT crop: worth a controlled follow-up, but not yet

**Recommendation: yes, native-crop comparison is worth doing, but only after
the coarse classifier is strengthened and/or gated; otherwise the comparison
is confounded by classifier capacity.**

Reasons:

- The current coarse classifier is near chance, so a native-crop run with the
  same weak architecture would not isolate the representation effect.
- The native crop carries 1.43 kHz frequency bins vs coarse B/2048 (2.4-39
  kHz/row); for 0/1/2 and 3/4/5 the modulation/edges may be lost by coarse
  aggregation. This is a plausible cause of the near-chance result.
- Measured on B=80/T=150 with 10% context: 20 MHz coarse crop 616x286 vs
  native 16,800x286; 40 MHz coarse 1,230x170 vs native 33,601x170. Crop+resize
  ~0.15-0.27 ms; ResNet18 forward ~0.07-0.15 ms/proposal; 4.8k val boxes add
  ~1-5 s, small. Training data generation is the larger cost.
- A naive native crop resized to 128x256 compresses 100-260x and may erase the
  detail being tested. Use 256x256/512x512, a 1D frequency profile, or a
  two-branch coarse+native-profile classifier; keep the **same proposal box,
  physical context, architecture, fit/val split and fusion rule**.

Recommended next steps:

1. Keep the shared fine-0-5 filter as the new baseline (0.6497).
2. Improve the 0-5 classifier before native crops:
   - stronger backbone (ResNet18/ConvNeXt-Tiny) or detector P5/ROI features;
   - more fit proposals/hard negatives, light augmentation, longer training;
   - **confidence gating**: only change a class when the classifier margin is
     high (threshold chosen on the fit holdout, not val); otherwise keep the
     detector label/score.
3. If the stronger coarse classifier still confuses 0/1/2 or 3/4/5, run the
   same-box coarse vs native STFT crop ablation with a fair input design.
4. 10/11 stays separate (mixed bandwidth, fine helps narrow instances);
   6/7/8/9/12/13 are not classifier targets.

Test remains sealed.
