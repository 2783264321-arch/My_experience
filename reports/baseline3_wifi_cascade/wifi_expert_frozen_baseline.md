# Wi-Fi 专家冻结基线（W-Shared, E0/C1）

2026-11-21。本文件是 Wi-Fi 0–5 调制专家的**冻结基线记录**，用于
后续复现与对照；不包含 E1/E2/E3 微调权重，也不涉及 SRRC 10/11。
test 全程封存。

## 1. 权重与哈希

### YOLO 检测器（C0 候选源）

| 项 | 值 |
| --- | --- |
| 路径 | `runs/yolo11s_c1_directreg_2048x512/best_corrected.pt` |
| SHA256 | `f262207d98118aa7ff0980818cbf5d04916145b9b09290063f66c2fb1dd16b5d` |
| 大小 | 35 MB |
| 说明 | EMA of `last.pt`（epoch 49），Baseline-1 冻结；本轮未重训 |

### W-Shared 专家（三个 seed）

| seed | 路径 | SHA256 | 大小 |
| ---: | --- | --- | ---: |
| 1729 | `runs_local/baseline3b/weights/SHARED_seed1729.pt` | `e9193a0fef7afb80cdc6cacf40d04fed9a76d002c4460db1efd366f6c9c69c53` | 9,828,624 B |
| 1730 | `runs_local/baseline3b/weights/SHARED_seed1730.pt` | `d3e54c41a2d28ad39f28e9adeab2215f8e38a0dc95155f23f87c861c17807e38` | 9,828,624 B |
| 1731 | `runs_local/baseline3b/weights/SHARED_seed1731.pt` | `74b322f677ff3e52d2f33996fe362f1ab76f110c2ffe2f6f770a41439777aca8` | 9,828,624 B |

权重为本地文件，**不入库**（`.gitignore` 含 `*.pt` / `runs_local/`）。
E1/E2/E3 权重在 `runs_local/baseline3ca/weights/`，仅作审计留档，
不是本基线的一部分。

## 2. CNN 配置

`scripts/baseline3b_train_experts.py::HistCNN`，Sensors 2026 架构：

| 层 | 配置 |
| --- | --- |
| 输入 | `1 × 50 × 50`（单通道幅相直方图） |
| conv1 | 3×3, 32 filters, BN, ReLU, 2×2 max-pool |
| conv2 | 3×3, 64 filters, BN, ReLU, 2×2 max-pool |
| conv3 | 3×3, 128 filters, BN, ReLU, 2×2 max-pool |
| dropout | 0.5 |
| fc1 | 128×6×6 = 4,608 → 512, ReLU |
| fc2 | 512 → 3 |
| 参数 | 2,454,467 |
| 输出 | 3 类：QPSK / 16QAM / 64QAM（带宽家族内） |

预测：每个实例最多 4 个 128-symbol block；每个 block 过 CNN 后
对 softmax 取平均，再取 argmax。seed 集成时对三个 seed 的 softmax
再平均。

## 3. 直方图协议（50×50）

`scripts/baseline3b_histogram.py`：

- 每个 128-symbol block 构造 `H[amp_bin, phase_bin]`，50×50。
- 幅度池：block 内 data subcarriers 的 `|Y|`；`thr=P99(A)`，
  `amax=max(A[A<=thr])`，丢弃 `|Y|>thr`。
- 相位对：每个 8-symbol 组内相邻符号 `(s, s+1)`，取
  `wrap(angle(Y_{s+1,k}) − angle(Y_{s,k}))`。
- 幅度归一 `|Y|/amax ∈ [0,1]`；相位 `(Δφ+π)/(2π) ∈ [0,1)`。
- 分箱 `ceil(x·50)−1`，clip 到 `[0,49]`；计数累加。
- 训练输入 `H / H.max()`（per-instance max norm, float32）。
- 每个实例最多取最早的 4 个完整 block；不足 128 symbol 的实例
  不产生 block。
- 不使用 GT 标签做同步、掩码或直方图参数选择。

## 4. OFDM 前端与同步门限

`scripts/baseline3b_ofdm_frontend.py`：

- 频率约定：FFT bin 0 = 观测中心 `f_lo + B/2`；解调中心为预测/GT
  框中心。
- 带限：`band_limit_fft(x, fs, bw×1.02, edge_frac=0.05)`。
- 边缘裁剪：两端各 0.5 ms（若长度允许）。
- 时间上限：框时长 `cap = min(duration, 20 ms)`，居中截取。
- CP 扫描候选：`Tu ∈ {3.2, 6.4, 12.8, 25.6, 51.2} µs`，
  `CP ∈ {0.2, 0.4, 0.8, 1.6, 3.2, 6.4} µs`，label-blind。
- 通过门限：`metric > 0.34` **且** `ratio > 15`。
- SCS 312.5 kHz；20 MHz→64-FFT（52 occupied / 48 data），
  40 MHz→128-FFT（104 occupied / 96 data）；DC 与带边 null。
- `filter_symbol_rows` 按带内能量中位数的 0.5 倍丢弃弱行；只在
  连续 symbol run 内构造 block。
- 最大 4,000 symbols/实例。

## 5. 路由与回退规则

- 路由只使用 **YOLO 预测类别**，绝不使用 GT：
  - 预测 0/1/2 → W20 家族，W-Shared 输出 0/1/2；
  - 预测 3/4/5 → W40 家族，W-Shared 输出映射到 3/4/5（+3）。
- GT 只用于离线审计/评估，不进入推理链路。
- 同步失败/直方图不足：保留 YOLO 原始类别，不调用专家。
- 专家只在预测家族内改调制阶数，**不改变带宽家族**。
- 级联是 label-only：框、得分、NMS、候选集不变。
- 未匹配到 Wi-Fi GT 的同步成功预测：仍按预测家族调用专家并改
  标签，进入完整 AP；但不进入分类训练/条件准确率，不伪造标签。

## 6. 数据划分（冻结）

`reports/baseline3_wifi_cascade/adaptation_split_v2.json`：

| 项 | 值 |
| --- | --- |
| 算法 | 按 IQ `sample_id` 分组；按（是否有 sync_ok × 家族构成）分层；层内 `sha256(seed:sample_id)` 排序；70/15/15 最大余数 |
| seed | 20261121 |
| 分配 SHA256 | `57120ec766299136f51f2968dbc4c23e4255ef1e87bfff24e14b574ab4538f77` |
| IQ 文件 | train 654 / cal 141 / final 140 |
| 双同步配对（正确家族） | train 169（144 IQ）/ cal 34（31）/ final 35（31） |
| W40 64QAM 覆盖 | cal 2 / final 1（覆盖保障后） |

final-dev 是内部开发留出，不是独立测试；test 未使用。

## 7. 正式指标

| 指标 | 值 |
| --- | ---: |
| C0 全量 val mAP50 | **0.6497250**（47,783 preds） |
| C1 W-Shared 全量 val mAP50（三 seed 均值） | **0.6597**（+1.00 pp） |
| C1 逐 seed | 0.659954 / 0.659931 / 0.659321 |
| C0 final-dev 子集 mAP50 | 0.6634728 |
| E0（同一冻结 W-Shared）final-dev 子集 mAP50 | 0.6763083（+1.2835 pp） |
| E0−C0 final-dev 配对 bootstrap 95% CI | **[−0.58, +3.30] pp**（2,000 次，按 IQ 文件） |
| E0 条件准确率（35 双同步） | 0.7524 |
| E0 全部 matched 严格准确率（216 对，跨家族计错） | 0.5340 |
| E0 正确家族组内准确率（215 对） | 0.5364 |

**注意**：final-dev 子集 C0 与全量 val C0 不同，绝对 mAP 不可直接
比较；上表同时列出两者仅为记录。E0−C0 的 final-dev CI 包含 0，
全量 val 的 C1 增益 +1.00 pp 是本基线的主要历史证据，但本轮没有
对全量 val 重做 IQ-file bootstrap。

## 8. 已知局限

1. 同步覆盖率低：全量 val 807/4,844（16.7%），final-dev
   125/704（17.76%）；82% 的 Wi-Fi 预测回退 YOLO 标签。
2. 同频/跨类重叠普遍：any-overlap 91.5%，Wi-Fi overlap 48.1%，
   强重叠 48.5%；clean 子集仅 8.5%。
3. W40 16QAM/64QAM 样本极薄：训练配对 10/8，final-dev 双同步
   3/1；不能支撑逐类强结论。
4. 分类域偏移≈0：GT 0.7515 vs Pred 0.7554（train 169 对），
   E1/E2/E3 未在 E0 上给出可复现增益；不采用微调基线。
5. 微调“冻结层”勘误：E1/E2/E3 只冻结 conv1/conv2 **参数**，
   BN1/BN2 running statistics 仍更新；见
   `finetune_freeze_erratum.md`。本冻结基线 E0 不受影响。
6. label-only 级联不改善定位/召回；未匹配的 561 个同步成功预测
   只进入 AP，没有可靠监督标签。
7. 仅验证 Wi-Fi 0–5；SRRC 10/11 是独立任务，不能复用本前端/权重。

## 9. 复现入口

- 专家训练：`scripts/baseline3b_train_experts.py`
- 前端/直方图：`scripts/baseline3b_ofdm_frontend.py`、
  `scripts/baseline3b_histogram.py`
- 级联：`scripts/baseline3c_cascade.py`（及 `baseline3ca_*` 审计脚本）
- 冻结评估：`reports/baseline3_wifi_cascade/comparison.md`、
  `cascade_adaptation_comparison.md`
- 数据划分：`reports/baseline3_wifi_cascade/adaptation_split_v2.json`
- 权重哈希审计：`reports/baseline3_wifi_cascade/finetune_freeze_audit.json`
