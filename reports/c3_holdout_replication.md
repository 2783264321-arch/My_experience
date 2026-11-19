# C3 holdout replication (1,159 samples)

Frozen best/last checkpoints; no re-selection or threshold tuning.
Holdout = full val 1,543 minus the 384-sample C3 subset.

## Overall / groups

| pair | model | mAP | AP_agn | R_agn | 0/1/2 | 3/4/5 | 10/11 | 9/12/13 | n_pred |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1729 best | dense | 0.4746 | 0.7810 | 0.8510 | 0.3129 | 0.3214 | 0.4902 | 0.3536 | 17676 |
| s1729 best | zero3 | 0.5148 | 0.7843 | 0.8774 | 0.3732 | 0.3437 | 0.5957 | 0.3747 | 36308 |
| s1729 last | dense | 0.4587 | 0.7497 | 0.8200 | 0.2894 | 0.3081 | 0.4432 | 0.3655 | 9964 |
| s1729 last | zero3 | 0.4527 | 0.7441 | 0.8166 | 0.2544 | 0.3547 | 0.4361 | 0.3365 | 9705 |
| s1730 best | dense | 0.5220 | 0.7826 | 0.8584 | 0.3260 | 0.4534 | 0.5450 | 0.3827 | 22938 |
| s1730 best | zero3 | 0.4831 | 0.7799 | 0.8560 | 0.3184 | 0.3540 | 0.5068 | 0.3647 | 22067 |
| s1730 last | dense | 0.4579 | 0.7511 | 0.8222 | 0.2627 | 0.3245 | 0.4950 | 0.3243 | 10450 |
| s1730 last | zero3 | 0.4513 | 0.7426 | 0.8033 | 0.2707 | 0.3383 | 0.4144 | 0.3357 | 9561 |

## Common-box 3/4/5 classification

| pair | common n | dense acc | zero3 acc | Δ | dense R | zero3 R | dense-only correct | zero3-only correct | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1729 best | 260 | 0.4308 | 0.4654 | -3.46 pp | 0.9276 | 0.9414 | 37 | 46 | 0.380 |
| s1729 last | 240 | 0.4875 | 0.5000 | -1.25 pp | 0.8828 | 0.8828 | 38 | 41 | 0.822 |
| s1730 best | 268 | 0.5299 | 0.4254 | 10.45 pp | 0.9414 | 0.9448 | 65 | 37 | 0.007 |
| s1730 last | 239 | 0.4519 | 0.5063 | -5.44 pp | 0.8690 | 0.8862 | 43 | 56 | 0.228 |

## Means

| metric | dense mean | zero3 mean | Δ |
| --- | ---: | ---: | ---: |
| mAP | 0.4783 | 0.4755 | 0.28 pp |
| 0/1/2 | 0.2977 | 0.3042 | -0.64 pp |
| 3/4/5 | 0.3518 | 0.3477 | 0.42 pp |
| 10/11 | 0.4933 | 0.4882 | 0.51 pp |
| 9/12/13 | 0.3565 | 0.3529 | 0.36 pp |
| R_agn | 0.8379 | 0.8383 | -0.04 pp |

## Notes

- Common boxes require both models to match the same 3/4/5 GT at IoU>=0.5.
- McNemar is exact two-sided on discordant common boxes.
- No weight selection or threshold tuning on the holdout.


## Best vs last means (descriptive)

| model | ckpt | mAP | 3/4/5 AP | common 3/4/5 acc | 0/1/2 |
| --- | --- | ---: | ---: | ---: | ---: |
| dense | best mean | 0.4983 | 0.3874 | 0.4804 | 0.3195 |
| zero3 | best mean | **0.4990** | 0.3489 | 0.4454 | **0.3458** |
| dense | last mean | **0.4583** | 0.3163 | 0.4697 | 0.2761 |
| zero3 | last mean | 0.4520 | **0.3465** | **0.5032** | 0.2626 |

- Best checkpoints: dense is +3.85 pp on 3/4/5 AP and +3.50 pp on common
  classification, but the gain is entirely from seed1730; seed1729 is negative.
- Last checkpoints: zero3 is +3.02 pp on 3/4/5 AP and +3.35 pp on common
  classification; dense has a slightly higher overall mAP but worse 3/4/5.
- The 384-subset conclusion (dense common accuracy higher in 4/4 comparisons)
  **did not replicate** on the 1,159 holdout.

## Per-seed common-box detail

| seed/ckpt | common n | dense acc | zero3 acc | dense-only correct | zero3-only correct | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s1729 best | 260 | 0.4308 | **0.4654** | 37 | 46 | 0.380 |
| s1729 last | 240 | 0.4875 | **0.5000** | 38 | 41 | 0.822 |
| s1730 best | 268 | **0.5299** | 0.4254 | 65 | 37 | **0.007** |
| s1730 last | 239 | 0.4519 | **0.5063** | 43 | 56 | 0.228 |

Per-class common accuracy (class 3 / 4 / 5):

| pair | dense | zero3 |
| --- | --- | --- |
| s1729 best | .516 / .432 / .329 | **.645** / .375 / **.354** |
| s1729 last | .575 / .551 / .320 | **.609** / .449 / **.427** |
| s1730 best | **.592** / **.449** / **.543** | .510 / .393 / .358 |
| s1730 last | **.584** / .364 / .384 | .573 / **.545** / .384 |

The only significant comparison is s1730 best (p=0.007, dense better); the
other three are neutral or favour zero3. With four tests and seed/checkpoint
selection already made on the 384 subset, this is not a stable effect.

## Judgment

1. **The holdout does not confirm that dense improves modulation
   classification of already localized 40 MHz Wi-Fi.** The 384-subset 4/4
   direction did not replicate. Best checkpoints split by seed; last
   checkpoints favour zero3 on 3/4/5.
2. **Overall mAP is unchanged.** Best means are effectively tied (0.4983 vs
   0.4990); last dense is +0.63 pp overall but −3.02 pp on 3/4/5.
3. **Localization is not the explanation.** Dense recall is equal/lower; the
   one significant seed1730-best result is classification, not more boxes.
4. **Checkpoint selection is unstable.** Best weights chosen on 384 do not
   generalize consistently (dense s1729 best is much worse than s1730 best;
   last flips the common-box ranking). Future comparison should pre-register
   a fixed checkpoint rule (e.g. last) before looking at holdout.
5. **Do not expand to full training.** There is no stable evidence to justify
   a content-bearing channel follow-up. If continuing, the only defensible
   next step is a small pre-registered multi-seed study with a fixed
   checkpoint rule and a live uninformative capacity control
   (power+power+power or random live channels); otherwise stop C3-for-3/4/5.
6. Test remains sealed; this is a 1,159-sample holdout, not the test split.
