# SRRC GT 小跑准备：40 组试切与全量缓存

2026-11-22。基分支 `work/srrc-gt-pilot-prep`（`8a40ff3`）。本轮只做
CPU 只读准备：先生成完整 fit 标注清单，再用低优先级、单进程、
单线程试切 40 个文件组，观察 D-FINE 步时后决定是否切完整 fit。
**未训练、未改推理、未使用 test；GPU 留给 D-FINE。**

## 0. 结论先行

- 完整 fit 标注清单已生成：**1,824 个 10 MHz SRRC 实例 / 1,571 个
  group**；内部 train/dev 为 **1,458 / 366**，两组类别和交叠层
  都保留；没有 group 跨侧。
- 40 组试切：**47 个实例 / 40 个 group**，prepare 用时 **33.34 s**，
  缓存 **3,094,463 B（约 3.0 MB）**，8×2×1,024 float32；无零窗、
  无 NaN。
- D-FINE 相邻稳定训练段：试切前 200 步中位 **0.780 s**，试切期间
  46 步中位 **0.792 s**，**+1.54%**；没有触发持续变慢或验证/保存
  空档，试切完成。
- 全量 fit 缓存：**已完成**，prepare **1,112.84 s（18.55 min）**，
  缓存 **119,980,212 B（114.4 MiB）**；D-FINE 同时段 1,377 步
  中位 **0.788 s**，较基线 **+1.03%**，未触发停止。
- 试切速率 0.709 s/实例；全量实际 0.610 s/实例（试切样本偏长），
  全量墙钟比按试切外推的 21.6 min 更快。

## 1. 完整 fit 标注清单

命令（只读 JSON，不读 `.bin`）：

```bash
CUDA_VISIBLE_DEVICES="" python -m scripts.baseline3srrc_gt_pilot prepare \
  --metadata-only --out /root/autodl-fs/runs_local/srrc-gt-pilot/fit_manifest.json
```

结果：

| 项 | 数值 |
| --- | ---: |
| 10 MHz SRRC 实例 | **1,824** |
| group 数 | **1,571** |
| 内部 train / dev | **1,458 / 366** |
| class10 / class11 | **941 / 883** |
| train 内 class10/11 | 752 / 706 |
| dev 内 class10/11 | 189 / 177 |
| 交叠层 clean / weak / strong | **383 / 594 / 847** |
| train 交叠层 | 312 / 480 / 666 |
| dev 交叠层 | 71 / 114 / 181 |
| 标注读取耗时 | **70.2 s** |

分组按 manifest 的 `group_id`，fit 内 5 折第 0 折做 dev；同一 group
只进一侧。正式 val/test 未参与。group 大小分布：1 实例 1,343 组、
2 实例 205 组、3 实例 21 组、4 实例 2 组（最大 4）。

## 2. 40 组试切

命令：

```bash
nice -n 19 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.baseline3srrc_gt_pilot prepare --max-groups 40 \
  --out /root/autodl-fs/runs_local/srrc-gt-pilot/fit_8win_trial.npz
```

结果：

| 项 | 数值 |
| --- | ---: |
| 选中 group | **40** |
| 实例 | **47**（train 37 / dev 10） |
| class10 / class11 | 26 / 21 |
| 交叠层 clean / weak / strong | 12 / 9 / 26 |
| prepare 用时 | **33.34 s**（监控总耗时 38.03 s，含启动） |
| 缓存大小 | **3,094,463 B（约 3.0 MB）** |
| 张量 | `[47, 8, 2, 1024]` float32 |
| 有限值 / 零窗 | 全部有限 / **0** |
| 每窗 RMS 范围 | 0.345 – 4.928（保留原幅度） |

试切缓存只用于吞吐/接口验收，不进入训练。

## 3. D-FINE 步时对比

选择 epoch 40 的相邻稳定训练段；避开 epoch 39 的验证/保存。

| 窗口 | 步范围 | n | 中位 step_s | p95 | max | 时间 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 试切前 | 29,827–30,026 | 200 | **0.780** | 0.888 | 0.985 | 22:32:45–22:35:25 |
| 试切期间 | 30,027–30,072 | 46 | **0.792** | 0.894 | 0.938 | 22:35:26–22:36:02 |

- 中位差 **+1.54%**；p95 差约 +0.6%；没有持续变慢。
- 试切期间没有出现 >10 s 的训练日志空档，未触发验证/保存保护。
- 滚动中位在 0.774–0.792 s 之间，处于正常波动。

因此按同样低优先级单进程配置切完整 fit。

## 4. 全量 fit 缓存

全量使用与试切相同的配置：`nice -n 19`、单进程、单线程、
`CUDA_VISIBLE_DEVICES=""`、8 窗×1,024 点、20 MS/s。

| 项 | 数值 |
| --- | ---: |
| prepare 用时 | **1,112.84 s（18.55 min）** |
| 监控总耗时 | 1,117.49 s（18.62 min） |
| 实例 / group | **1,824 / 1,571** |
| train / dev | **1,458 / 366** |
| class10 / class11 | **941 / 883** |
| 交叠层 clean / weak / strong | **383 / 594 / 847** |
| 缓存大小 | **119,980,212 B（114.4 MiB）** |
| 张量 | `[1824, 8, 2, 1024]` float32 |
| 有限值 / 零窗 | 全部有限 / **0** |
| 每窗 RMS 范围 | 0.118 – 13.491（保留原幅度） |
| group 跨侧 | **0**（train 1,259 组 / dev 312 组） |
| 实际吞吐 | **0.610 s/实例**，0.708 s/group |

缓存到 `/root/autodl-fs/runs_local/srrc-gt-pilot/fit_8win.npz`；
`/root/autodl-tmp` 训练盘未增加大文件。

### 4.1 全量期间的 D-FINE 步时

| 窗口 | 步范围 | n | 中位 step_s | p95 | max |
| --- | --- | ---: | ---: | ---: | ---: |
| 试切前基线 | 29,893–30,092 | 200 | **0.780** | 0.893 | 0.985 |
| 全量期间 | 30,093–31,469 | 1,377 | **0.788** | 0.908 | 2.618 |
| epoch 40 内 | 30,093–30,545 | 453 | 0.784 | 0.898 | — |
| epoch 41 | — | 745 | 0.786 | 0.906 | — |
| epoch 42 前段 | — | 179 | 0.804 | 0.943 | — |

全量监控启动时重新取了相邻 200 步基线（29,893–30,092），中位仍为
0.780 s，避免跨轮次的漂移。

- 全量期间中位 **+1.03%**，p95 +1.7%；epoch 42 前段中位 +3.1%，
  仍低于 15% 停止阈值，且不是持续加速恶化。
- 1,377 步中只有个别尖峰（max 2.618 s），没有 >10 s 的
  验证/保存空档；监控未触发停止。
- 结论：在这台共享机器上，低优先级单进程切片对 D-FINE 的
  可观测影响很小；代价是 SRRC 缓存墙钟约 18.5 min。

## 4.2 产物

- `/root/autodl-fs/runs_local/srrc-gt-pilot/fit_manifest.json`
- `/root/autodl-fs/runs_local/srrc-gt-pilot/fit_8win_trial.npz`
- `/root/autodl-fs/runs_local/srrc-gt-pilot/fit_8win.npz`
- `/root/autodl-fs/runs_local/srrc-gt-pilot/trial_monitor.json`
- `/root/autodl-fs/runs_local/srrc-gt-pilot/full_monitor.json`
- `/root/autodl-fs/runs_local/srrc-gt-pilot/full_cache_summary.json`

## 5. 判断与后续

- **试切和全量都没有触发停止条件**：试切 +1.54%、全量 +1.03%，
  D-FINE 步时稳定；低优先级单进程方案可继续使用。
- 全量缓存只写 `/root/autodl-fs`，不占 `/root/autodl-tmp` 的
  5.6 GB 训练盘；缓存 114.4 MiB，加载后可一次放入内存。
- 墙钟 18.55 min 是单进程/单线程的代价；若以后要提速，可用
  2–4 个 worker 并行，或只处理 8 个窗附近的小段而不是整段
  FFT/重采样，但需要重新做步时监控和前端核验。
- GPU 仍留给 D-FINE；SRRC CNN 训练等空卡后，用同一缓存先跑
  magnitude / I/Q 两臂 8 窗小跑，再做 16 窗敏感性。
- 真实分类表现、迁移和 AP 尚未评估；本轮只交付可复现的缓存
  与吞吐证据。
