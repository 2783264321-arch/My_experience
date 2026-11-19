# Dense correlation estimator comparison

Current = per-pair normalized correlation, then average per column.
Aggregate = sum phase-corrected correlation and pair energy per column, then divide once.
No cache/training operator was changed.

## Synthetic tone + noise

| SNR dB | current target | aggregate target | current noise | aggregate noise | current phase | aggregate phase |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -10.0 | 0.9989 | 0.9989 | 0.5282 | 0.6192 | -0.0003 | -0.0002 |
| -5.0 | 0.9997 | 0.9997 | 0.5281 | 0.6192 | -0.0002 | -0.0002 |
| 0.0 | 0.9999 | 0.9999 | 0.5281 | 0.6192 | -0.0001 | -0.0001 |

## Real samples, GT 3/4/5 and noise

| sample | B/T | GT n | current GT | aggregate GT | current noise | aggregate noise |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train/29 | 80/150 | 1 | 0.3144 | 0.4530 | 0.3156 | 0.4557 |
| train/14 | 40/150 | 1 | 0.6866 | 0.8636 | 0.6886 | 0.8595 |
| train/1697 | 40/20 | 1 | 0.7877 | 0.8819 | 0.7830 | 0.8799 |
| train/1205 | 40/40 | 2 | 0.7330 | 0.8708 | 0.7341 | 0.8715 |
| train/7401 | 50/150 | 1 | 0.5837 | 0.7737 | 0.5845 | 0.7737 |
| train/3441 | 40/150 | 2 | 0.6882 | 0.8585 | 0.6888 | 0.8597 |

## Weak/strong 3/4/5 split

Median GT coarse power = 0.554027.

| bucket | current mean mag | aggregate mean mag | current coherence | aggregate coherence |
| --- | ---: | ---: | ---: | ---: |
| weak (n=4) | 0.5686 | 0.7359 | 0.9468 | 0.9651 |
| strong (n=4) | 0.7351 | 0.8718 | 0.9143 | 0.9463 |

Interpretation: aggregate-first should suppress independent noise more (lower noise mean), but on weak signals it can be dominated by a few strong pairs; compare both GT magnitude and coherence. This is a diagnostic, not a training result.

