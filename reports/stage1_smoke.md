# Stage 1 data / STFT / coordinate smoke

Command:

```bash
cd /root/autodl-tmp/workspace/baseline
PYTHONPATH=. .venv/bin/python scripts/smoke_data.py
```

Raw output: `reports/smoke_data.json`.

Six real samples covering B=5/20/40/80 MHz and T=20/80/150 ms were checked:

| sample | B MHz | T ms | F | h | fine slices | coarse labels | nonempty fine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train/1 | 5 | 20 | 3500 | 196 | 2 | 1 | 2 |
| train/4 | 20 | 20 | 14000 | 783 | 9 | 2 | 7 |
| train/108 | 20 | 150 | 14000 | 5871 | 9 | 6 | 9 |
| train/9 | 40 | 80 | 28000 | 6263 | 17 | 6 | 10 |
| train/86 | 80 | 20 | 56000 | 3132 | 34 | 4 | 8 |
| train/29 | 80 | 150 | 56000 | 23484 | 34 | 6 | 26 |

Checks passed:
- every STFT is `[F, 512]`, every fine/coarse view is `[1, 2048, 512]`;
- empty fine slices exist and are preserved (no filtering);
- all boxes stay in `[0,1]` with positive width/height;
- coarse-view coordinate inverse round-trips with max time error `7.1e-15 ms`
  and max frequency error `0.0 MHz`;
- labels are kept per original instance (no dedup / same-box ignore).

DataLoader check: two sources -> four `[1,2048,512]` views, `cls` `[N,1]`,
`bboxes` `[N,4]` normalized xywh, `batch_idx` remapped to the flattened views.
