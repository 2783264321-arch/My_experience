# c_only vs frozen coarse-only (B) predictions

c_only narrow GT: 542 (C finds them at IoU>=0.5, B does not).
Best-IoU B narrow prediction per instance:

| bucket | n | frac | score med | score p90 |
| --- | ---: | ---: | ---: | ---: |
| IoU>=0.5 (matched: score/NMS cut) | 0 | 0.0% | n/a | n/a |
| IoU 0.3-0.5 (near miss) | 136 | 25.1% | 0.032 | 0.322 |
| IoU 0.1-0.3 | 105 | 19.4% | 0.010 | 0.185 |
| IoU<0.1 (no fire) | 301 | 55.5% | 0.003 | 0.012 |

best-IoU median 0.027; GT height rows med 2.5 (p25 1.3, p75 3.6)
