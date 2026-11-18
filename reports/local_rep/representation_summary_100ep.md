# Local representation comparison: generated tables

## representation_comparison_100ep

Shared sample-level holdout: 758 crops / 501 samples; train 2996 crops / 2002 samples; class counts train [562, 568, 584, 438, 402, 442], holdout [143, 148, 148, 92, 115, 112].

| rep | group | acc mean | acc std | macro-F1 | confusion (sum) |
| --- | --- | ---: | ---: | ---: | --- |
| coarse | 012 | 0.4085 | 0.0172 | 0.4081 | `[[167, 160, 102], [171, 153, 120], [116, 110, 218]]` |
| coarse | 345 | 0.5528 | 0.0015 | 0.5513 | `[[199, 39, 38], [91, 168, 86], [118, 56, 162]]` |
| local_complex | 012 | 0.3485 | 0.0000 | 0.3486 | `[[53, 48, 42], [50, 50, 48], [47, 51, 50]]` |
| local_complex | 345 | 0.4107 | 0.0000 | 0.3954 | `[[39, 9, 44], [25, 24, 66], [27, 17, 68]]` |
| local_power | 012 | 0.4313 | 0.0173 | 0.4325 | `[[197, 141, 91], [180, 170, 94], [108, 135, 201]]` |
| local_power | 345 | 0.5799 | 0.0210 | 0.5800 | `[[188, 38, 50], [101, 199, 45], [113, 55, 168]]` |

### coarse / 012 confusion (rows GT, cols pred)

| GT \ pred | 0 | 1 | 2 | recall |
| --- | ---: | ---: | ---: | ---: |
| 0 | 167 | 160 | 102 | 0.389 |
| 1 | 171 | 153 | 120 | 0.345 |
| 2 | 116 | 110 | 218 | 0.491 |
Total: 1317

### coarse / 345 confusion (rows GT, cols pred)

| GT \ pred | 3 | 4 | 5 | recall |
| --- | ---: | ---: | ---: | ---: |
| 3 | 199 | 39 | 38 | 0.721 |
| 4 | 91 | 168 | 86 | 0.487 |
| 5 | 118 | 56 | 162 | 0.482 |
Total: 957

### local_complex / 012 confusion (rows GT, cols pred)

| GT \ pred | 0 | 1 | 2 | recall |
| --- | ---: | ---: | ---: | ---: |
| 0 | 53 | 48 | 42 | 0.371 |
| 1 | 50 | 50 | 48 | 0.338 |
| 2 | 47 | 51 | 50 | 0.338 |
Total: 439

### local_complex / 345 confusion (rows GT, cols pred)

| GT \ pred | 3 | 4 | 5 | recall |
| --- | ---: | ---: | ---: | ---: |
| 3 | 39 | 9 | 44 | 0.424 |
| 4 | 25 | 24 | 66 | 0.209 |
| 5 | 27 | 17 | 68 | 0.607 |
Total: 319

### local_power / 012 confusion (rows GT, cols pred)

| GT \ pred | 0 | 1 | 2 | recall |
| --- | ---: | ---: | ---: | ---: |
| 0 | 197 | 141 | 91 | 0.459 |
| 1 | 180 | 170 | 94 | 0.383 |
| 2 | 108 | 135 | 201 | 0.453 |
Total: 1317

### local_power / 345 confusion (rows GT, cols pred)

| GT \ pred | 3 | 4 | 5 | recall |
| --- | ---: | ---: | ---: | ---: |
| 3 | 188 | 38 | 50 | 0.681 |
| 4 | 101 | 199 | 45 | 0.577 |
| 5 | 113 | 55 | 168 | 0.500 |
Total: 957

### Overlap-condition accuracy (mean over seeds)

| rep | group | clean WiFi | WiFi overlap | cross-class WiFi | strong WiFi >=25% |
| --- | --- | ---: | ---: | ---: | ---: |
| coarse | 012 | 0.4354 (n=222) | 0.3810 (n=217) | 0.3979 (n=191) | 0.3752 (n=183) |
| coarse | 345 | 0.5915 (n=173) | 0.5068 (n=146) | 0.5076 (n=132) | 0.4720 (n=113) |
| local_complex | 012 | 0.3514 (n=222) | 0.3456 (n=217) | 0.3351 (n=191) | 0.3443 (n=183) |
| local_complex | 345 | 0.4277 (n=173) | 0.3904 (n=146) | 0.3636 (n=132) | 0.3717 (n=113) |
| local_power | 012 | 0.4144 (n=222) | 0.4485 (n=217) | 0.4520 (n=191) | 0.4444 (n=183) |
| local_power | 345 | 0.6262 (n=173) | 0.5251 (n=146) | 0.5202 (n=132) | 0.5162 (n=113) |
