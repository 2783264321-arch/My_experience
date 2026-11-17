# Stage 2 - loss, AMP and resource acceptance

Artifacts:
- `reports/stage2_resources.json` (real optimizer steps + data throughput)
- `reports/stage2_grad_probe.json` (FP32/FP16/BF16 gradients + full-height test)
- `/tmp/bench_resources_bf16.log`, `/tmp/grad_probe.log`

## Numerical acceptance on the real head

Real shape: `[n, 1, 2048, 512]`, YOLO11s body, C1, nc=14, `reg_max=1`,
`end2end=False`, DirectReg L1 branch.

| precision | non-finite gradient elements (probe) | verdict |
| --- | ---: | --- |
| FP32 | 0 | accepted (baseline) |
| FP16 autocast | 9 on the full-height probe; millions over a 16-view step | **rejected** |
| BF16 autocast | 0 | **accepted as AMP** |

The FP16 GradScaler reduced its scale on every step but still skipped every
optimizer update on the real 16-view batch. BF16 keeps the FP32 exponent range
and produced finite gradients and decreasing losses on the same batches.

Full-height regression probe (BF16, 200 updates on one full-frequency box):

| step | loss | box | L1 | decoded max height px |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 62.93 | 0.00 | 0.000 | 164.5 |
| 200 | 3.62 | 2.43 | 0.119 | 1408.0 |

DirectReg is therefore able to move a full-height box; a normal box also has
finite, non-zero box-head gradients. The box head final conv gradient channels
are recorded per case in the JSON.

## Real optimizer-step memory/time

AdamW + EMA + BF16, real batches (2 views per source), real targets:

| sources | views | step median s | views/s | peak reserved GiB |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 0.041 | 98 | 2.57 |
| 4 | 8 | 0.066 | 122 | 5.34 |
| 8 | 16 | 0.126 | 127 | 10.12 |
| 12 | 24 | 0.184 | 130 | 15.06 |

Chosen physical batch: **8 sources = 16 views**, accumulation 1. Peak reserved
is about 10.1 GiB, well under the 20 GiB gate. GPU is ~94 s/epoch of compute;
the data pipeline is the binding constraint.

## Data pipeline

On-the-fly scipy FFT backend (numerically equivalent to `torch.stft` to
~4e-7 relative), 64 real fit sources per measurement, batch 8:

| workers | FFT threads | sources/s | projected fit epoch min |
| ---: | ---: | ---: | ---: |
| 8 | 2 | 17.5 | 5.7 |
| 12 | 2 | 16.2-16.9 | 5.9-6.1 |
| 16 | 1 | 14.7 | 6.8 |
| 16 | 2 | 17.0 | 5.8 |

Chosen: 12 workers, 2 FFT threads. The CPU cgroup is 16 vCPU, so the pipeline
saturates near 17 sources/s; this is a CPU/STFT bound, not an I/O bound
(the IQ working set is ~102 GiB for fit and is streamed on the fly).

## Validation cost and cadence

`run_validation` uses every fine slice + coarse view (24,975 views for 1543 val
sources). Measured 324-view/16-source overfit validation: 4-8 s. Full-val
estimate is ~5-10 min including STFT, model/NMS and the evaluator, i.e. below
1/3 of a 5-epoch training block (~29 min). The pre-fixed cadence for stage 4
is therefore **every 5 epochs plus the final epoch**, recorded before training.
