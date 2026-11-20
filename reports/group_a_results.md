# Group A / B / C comparison

A = full-band single-image, independent training, coarse-only inference.
B = joint coarse+fine weights, coarse-only inference (same weights as C).
C = joint coarse+fine weights, all-view inference.
The primary A/C checkpoint is the last-epoch EMA. A best is reported separately.

## Overall

| metric | A | B | C | B−A | C−B | C−A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mAP@0.5 | 0.5404 | 0.5794 | 0.6427 | 0.0390 | 0.0633 | 0.1023 |
| AP50 agnostic | 0.8020 | 0.8178 | 0.9046 | 0.0158 | 0.0868 | 0.1026 |
| localization recall | 0.8570 | 0.8773 | 0.9580 | 0.0202 | 0.0807 | 0.1009 |
| classification accuracy | 0.8328 | 0.8278 | 0.8433 | -0.0051 | 0.0155 | 0.0105 |
| GT | 7162 | 7162 | 7162 | 0.0000 | 0.0000 | 0.0000 |
| predictions | 16173 | 25409 | 65025 | 9236.0000 | 39616.0000 | 48852.0000 |
| val views | 1543 | 1543 | 24975 | 0.0000 | 23432.0000 | 23432.0000 |

## Class groups (mean per-class AP@0.5)

| group | A | B | C | B−A | C−B | C−A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0/1/2 | 0.4061 | 0.4212 | 0.4020 | 0.0151 | -0.0192 | -0.0041 |
| 3/4/5 | 0.3947 | 0.4550 | 0.4415 | 0.0603 | -0.0136 | 0.0468 |
| 10/11 | 0.4899 | 0.5908 | 0.6395 | 0.1009 | 0.0487 | 0.1496 |
| 9/12/13 | 0.4599 | 0.4838 | 0.7804 | 0.0239 | 0.2967 | 0.3206 |

## Signal-level bandwidth groups (AP@0.5)

| group | A | B | C | B−A | C−B | C−A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| narrow ≤0.25 MHz | 0.4599 | 0.4838 | 0.7804 | 0.0239 | 0.2967 | 0.3206 |
| medium 0.3–2 MHz | 0.6844 | 0.7474 | 0.8388 | 0.0630 | 0.0914 | 0.1544 |
| wide 2.1–10 MHz | 0.3973 | 0.5055 | 0.4702 | 0.1082 | -0.0353 | 0.0729 |
| very wide 10.1–20 MHz | 0.4061 | 0.4212 | 0.4020 | 0.0151 | -0.0192 | -0.0041 |
| ultra >20 MHz | 0.3947 | 0.4550 | 0.4415 | 0.0603 | -0.0136 | 0.0468 |

## Inference latency

| mode | views/sample | build p50 s | model p50 s | end-to-end p50 s | end-to-end p95 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A coarse | 1.00 | 0.3170 | 0.0099 | 0.3269 | 0.7778 |
| C all | 19.10 | 0.4222 | 0.0859 | 0.5085 | 1.1587 |

Views ratio C/A: 19.10; end-to-end p50 ratio C/A: 1.555; model-only p50 ratio C/A: 8.635.

## Training budget and cost

| item | C (B uses same weights) | A |
| --- | ---: | ---: |
| epochs | 50 | 100 |
| views/source/epoch | 2 | 1 |
| effective views | 16 | 16 |
| optimizer steps | 37250 | 37300 |
| warmup steps | 2235 | 2238 |
| image presentations | 595700 | 595700 |
| training wall time, h | 2.69 | 1.62 |
| peak reserved, GiB | 10.1 | 10.1 |
| cache build, s | n/a | 2050 |
| cache size, GiB | n/a | 11.63 |
| cache training-path check | n/a | yes |

## A best (secondary, early-stop-style)

A last is overfit; `best.pt` is epoch 29 and is the relevant single-image
baseline if early stopping is allowed.

| metric | A best | B coarse | C all |
| --- | ---: | ---: | ---: |
| mAP@0.5 | 0.5763 | 0.5794 | **0.6427** |
| AP50 agnostic | **0.8208** | 0.8178 | 0.9046 |
| localization recall | **0.8789** | 0.8773 | 0.9580 |
| classification accuracy | 0.8270 | 0.8278 | 0.8433 |
| predictions | 30,981 | 25,409 | 65,025 |

| group | A best | B | C |
| --- | ---: | ---: | ---: |
| 0/1/2 | 0.3839 | 0.4212 | 0.4020 |
| 3/4/5 | 0.4285 | 0.4550 | 0.4415 |
| 10/11 | **0.6399** | 0.5908 | 0.6395 |
| 9/12/13 | 0.5002 | 0.4838 | **0.7804** |
| narrow ≤0.25 MHz | 0.5002 | 0.4838 | **0.7804** |
| medium 0.3–2 MHz | **0.8917** | 0.7474 | 0.8388 |
| wide 2.1–10 MHz | 0.4611 | **0.5055** | 0.4702 |
| very wide 10.1–20 MHz | 0.3839 | **0.4212** | 0.4020 |
| ultra >20 MHz | 0.4285 | **0.4550** | 0.4415 |

Interpretation:

1. **A best is effectively tied with B overall** (0.5763 vs 0.5794, −0.31 pp)
   and has slightly higher AP_agn/recall. It is a valid full-band single-image
   baseline, but it does not beat simply running the joint weights on the
   coarse view.
2. **C's gain is narrowband.** C−A best = +6.64 pp overall. The narrowband
   groups (9/12/13 and physical ≤0.25 MHz) go from 0.5002 to 0.7804,
   **+28.02 pp**; on medium 0.3–2 MHz A best is **+5.29 pp** better
   (0.8917 vs 0.8388), and on wide/very wide/ultra C is only
   +0.91/+1.81/+1.30 pp. The single image is not uniformly worse.

   2026-11-20 correction: the medium +5.29 pp is driven entirely by the
   57 GT of 1.6–2 MHz SRRC 16QAM (A best 0.766 vs C 0.488); BLE/Zigbee,
   the main medium residents (2,847 GT), are identical across arms
   (0.944–0.959), and medium-bandwidth SRRC QPSK slightly favors C
   (0.859 vs 0.843). Read it as a small-subset failure of the fine
   branch (cls11 fine-view AP 0.091), not a systematic single-image
   advantage. See docs/collaboration/opencode-ac-comparison-summaries.md.
3. **A last is not a fair single-image score without the overfitting caveat.**
   Last is 3.59 pp below best; training loss kept falling while val declined.
   The pre-registered main table uses last for both A and C, but the paper
   should also show A best and explain the gap as the early-stopping motivation.
4. **Inference slicing is not free.** C uses 19.10x views and 8.64x model time;
   end-to-end p50 is only 1.56x because the 0.7 ms STFT dominates. Report
   views/sample, model-only and end-to-end separately.
5. **Training budget parity holds.** 595,700 image presentations, effective
   batch 16, 37,300 vs 37,250 steps. A's lower wall time is partly cheaper
   coarse-only validation and the cache, not less model training.

Notes:
- B−A isolates the effect of training with fine views when inference is coarse-only.
- C−B isolates inference-time fine slices with the same joint weights.
- C−A is the full slicing scheme relative to the full-band single-image baseline.
- Test split remains sealed.
