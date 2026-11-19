# C3 front-end pilot (matched 20k steps)

Same power image, labels, 1536/384 subset, 20,000 optimizer steps and
effective batch 16. C1 is the power-only control; C1 mean is the average
of seeds 1729/1730. Absolute mAP is a subset result, not comparable to
the full 0.6497 baseline. Best EMA per arm is used.

## Best EMA

| arm | mAP | best step | AP_agn | R_agn | 0/1/2 | 3/4/5 | 10/11 | 9/12/13 | n_pred | train h | peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **C1 mean** | **0.4999** | - | 0.7825 | 0.8495 | 0.2625 | 0.3575 | 0.5319 | 0.4202 | 8494 | 0.79 | 8.6 |
| C1 s1729 | 0.4997 | 7680 | 0.7870 | 0.8470 | 0.2722 | 0.3570 | 0.4993 | 0.4266 | 7170 | 0.79 | 8.6 |
| C1 s1730 | 0.5000 | 6400 | 0.7779 | 0.8520 | 0.2529 | 0.3580 | 0.5644 | 0.4137 | 9818 | 0.78 | 8.6 |
| overlap s1729 | 0.5190 | 6400 | 0.7882 | 0.8515 | 0.3019 | 0.4012 | 0.5020 | 0.4403 | 8892 | 0.92 | 8.8 |
| overlap s1730 | 0.4966 | 7680 | 0.7808 | 0.8515 | 0.2473 | 0.3940 | 0.4856 | 0.4230 | 7701 | 0.91 | 8.8 |
| sparse s1729 | 0.5249 | 3840 | 0.7661 | 0.8529 | 0.2819 | 0.4213 | 0.6082 | 0.4018 | 13641 | 0.87 | 8.8 |
| sparse s1730 | 0.4747 | 8960 | 0.7669 | 0.8283 | 0.2336 | 0.3270 | 0.4703 | 0.4172 | 6237 | 0.96 | 8.8 |
| dense s1729 | 0.5072 | 7680 | 0.7850 | 0.8493 | 0.2452 | 0.4646 | 0.4465 | 0.4288 | 7155 | 0.87 | 8.8 |

## Wi-Fi matched counts (best EMA)

| arm | 0/1/2 correct | 0/1/2 wrong | 3/4/5 correct | 3/4/5 wrong |
| --- | ---: | ---: | ---: | ---: |
| **C1 mean** | **97.0** | **188.5** | **62.5** | **79.0** |
| C1 s1729 | 98 | 189 | 59 | 83 |
| C1 s1730 | 96 | 188 | 66 | 75 |
| overlap s1729 | 112 | 173 | 64 | 75 |
| overlap s1730 | 86 | 199 | 70 | 72 |
| sparse s1729 | 109 | 181 | 67 | 73 |
| sparse s1730 | 99 | 178 | 55 | 77 |
| dense s1729 | 101 | 183 | 68 | 69 |

## Paired deltas vs C1 mean (best EMA; pp)

| arm | ΔmAP | ΔAP_agn | ΔR_agn | Δ0/1/2 | Δ3/4/5 | Δ10/11 | Δ9/12/13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap s1729 | 1.92 | 0.58 | 0.20 | 3.94 | 4.37 | -2.99 | 2.01 |
| overlap s1730 | -0.32 | -0.16 | 0.20 | -1.52 | 3.65 | -4.63 | 0.28 |
| sparse s1729 | 2.51 | -1.64 | 0.34 | 1.94 | 6.38 | 7.63 | -1.84 |
| sparse s1730 | -2.51 | -1.55 | -2.12 | -2.89 | -3.06 | -6.15 | -0.30 |
| dense s1729 | 0.73 | 0.25 | -0.02 | -1.73 | 10.70 | -8.54 | 0.86 |

## Cache / front-end cost

| frontend | build s | shape | scale |
| --- | ---: | --- | ---: |
| overlap | 1470 | [1536, 2, 2048, 512] | 0.003780 |
| sparse | 201 | [1536, 2, 2048, 512] | 0.007293 |
| dense | 1229 | [1536, 2, 2048, 512] | 1.000000 |

Front-end per-sample: 1.42 s; 1536+384 estimate 45.5 min.

## Means and independent judgment

| arm | mAP | AP_agn | R_agn | 0/1/2 | 3/4/5 | 10/11 | 9/12/13 | 0/1/2 wrong | 3/4/5 wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **C1 mean** | **0.4999** | 0.7825 | 0.8495 | 0.2625 | 0.3575 | 0.5319 | 0.4202 | 188.5 | 79.0 |
| overlap mean | 0.5078 | 0.7845 | 0.8515 | 0.2746 | **0.3976** | 0.4938 | 0.4316 | 186.0 | 73.5 |
| sparse mean | 0.4998 | 0.7665 | 0.8406 | 0.2577 | 0.3741 | 0.5393 | 0.4095 | 179.5 | 75.0 |
| dense s1729 | 0.5072 | 0.7850 | 0.8493 | 0.2452 | **0.4646** | 0.4465 | 0.4288 | 183 | 69 |

Deltas vs C1 mean (pp):

| arm | ΔmAP | Δ0/1/2 | Δ3/4/5 | Δ10/11 | Δ9/12/13 |
| --- | ---: | ---: | ---: | ---: | ---: |
| overlap mean | +0.79 | +1.21 | **+4.01** | −3.81 | +1.14 |
| sparse mean | −0.01 | −0.49 | +1.66 | +0.74 | −1.07 |
| dense s1729 | +0.73 | −1.73 | **+10.70** | −8.54 | +0.86 |

Seed spread is large enough to invalidate the single-seed winners:

- C1: 0/1/2 1.93 pp, 3/4/5 0.10 pp, 10/11 6.51 pp, narrow 1.29 pp,
  overall 0.03 pp.
- overlap: 0/1/2 5.46 pp, 3/4/5 0.72 pp, 10/11 1.64 pp, narrow 1.73 pp,
  overall 2.24 pp.
- sparse: 0/1/2 4.83 pp, 3/4/5 **9.43 pp**, 10/11 **13.79 pp**, narrow
  1.54 pp, overall **5.02 pp**.

**Conclusions:**

1. **No complex front-end gives a stable 0/1/2 gain.** Overlap seed 1729
   reached 0.3019 but seed 1730 fell to 0.2473; sparse went 0.2819 → 0.2336;
   dense was 0.2452. The C1 mean is 0.2625. The 20 MHz Wi-Fi errors are not
   reliably reduced by any of the three.
2. **3/4/5 is the only replicated complex gain.** overlap improves both seeds
   (0.4012, 0.3940; mean +4.01 pp) with small seed spread. dense has the
   largest point estimate (+10.70 pp, wrong 79→69) but only one seed. sparse
   is not stable (0.4213 vs 0.3270). For 40 MHz Wi-Fi, the promising order is
   **dense > overlap > sparse**, but dense still needs a seed repeat.
3. **Sparse's apparent overall +2.51 pp was seed luck.** Its second seed was
   −2.51 pp; mean overall is identical to C1. The 10/11 +7.63 pp also did not
   replicate (0.6082 → 0.4703). Do not adopt sparse on this evidence.
4. **Dense is a 3/4/5 specialist, not a global replacement.** It lowers 3/4/5
   wrong from 79 to 69 but worsens 0/1/2 and 10/11; overall +0.73 pp is not
   enough to justify a global switch.
5. **Capacity is still confounded.** C1 is 1-channel; all C3 arms are 3-channel.
   The 3/4/5 gain may be extra capacity/optimization rather than complex
   information. A C3-zero-complex control is required before claiming a causal
   complex benefit. The prior coarse+fine audit reached the same caveat.
6. **Recommendation.** Do not replace C1 globally. If the priority is 40 MHz
   Wi-Fi (`3/4/5`), run a focused follow-up with **overlap (replicated) and
   dense (largest point estimate)**, a C3-zero-complex capacity control, and
   at least one more seed. For 20 MHz (`0/1/2`), this pilot provides no stable
   complex-front-end win.

Limits: 1536 fit / 384 val subset, 20,000 steps, single seed for dense,
two seeds for C1/overlap/sparse; absolute mAP is not the full 0.6497 result.
Test remains sealed.

