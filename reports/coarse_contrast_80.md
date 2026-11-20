# Local contrast: coarse mean vs intra-band peak (80 MHz)

Contrast = max cell inside the GT region - median cell in a local ring
(rows +-3, cols +-2, region excluded), dB. Same grids as
diagnose_coarse_grid_peak.py. Read-only.

| group | n | mean contrast med | peak contrast med | peak-mean med | peak >= mean+3dB | h rows med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| both | 174 | 32.6 | 33.5 | -0.1 | 0.28 | 7 |
| c_only | 165 | 35.8 | 42.0 | +6.6 | 0.79 | 2 |
| neither | 31 | 9.8 | 8.6 | +0.8 | 0.32 | 4 |

c_only per class:

| class | n | mean contrast med | peak contrast med |
| --- | ---: | ---: | ---: |
| 9 | 102 | 37.2 | 43.7 |
| 12 | 35 | 14.2 | 16.3 |
| 13 | 28 | 39.6 | 44.4 |
