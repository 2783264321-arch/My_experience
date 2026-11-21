# Baseline-3 Phase C′-Adaptation · 第一阶段数据审计报告

2026-11-21。分支 `work/baseline3-phase-c-adaptation`，HEAD `5cef887`。

## 1. 配对匹配结果

| 项 | 值 |
| --- | ---: |
| val 总样本 | 1,543 |
| 含 WiFi GT 的样本 | **936** |
| class-agnostic IoU≥0.5 一对一匹配 WiFi 对 | **1,394** |

## 2. 预测框同步可用率（matched predicted crops）

| 层 | n | sync_ok | 率 |
| --- | ---: | ---: | ---: |
| **全体** | **1,394** | **246** | **17.6%** |
| W20 (0/1/2) | 967 | 171 | 17.7% |
| W40 (3/4/5) | 427 | 75 | 17.6% |
| IoU ≥ 0.75 | 1,292 | 231 | 17.9% |
| IoU 0.5–0.6 | 26 | 3 | 11.5% |

**逐类同步率差异显著**：

| 类 | n | sync_ok | 率 |
| --- | ---: | ---: | ---: |
| 0 (W20 QPSK) | 328 | 47 | 14.3% |
| 1 (W20 16QAM) | 338 | 66 | 19.5% |
| 2 (W20 64QAM) | 301 | 58 | 19.3% |
| 3 (W40 QPSK) | 152 | **48** | **31.6%** |
| 4 (W40 16QAM) | 143 | 15 | **10.5%** |
| 5 (W40 64QAM) | 132 | 12 | **9.1%** |

W40 内部 QPSK 31.6% vs 16/64QAM ~10%——可能反映 W40 QPSK 信号更强/
更少重叠，或 YOLO 对 W40 16/64QAM 的框几何更差。后续 H1 验证需
对比同信号 GT 框的同步率。

## 3. 文件级划分（70/15/15，seed 1729）

| 划分 | 样本 | 配对 | sync_ok | 类别分布 |
| --- | ---: | ---: | ---: | --- |
| adaptation-train | 655 | 954 | **161** | 0:241 1:232 2:208 3:91 4:90 5:92 |
| calibration-dev | 140 | 221 | 48 | 0:42 1:56 2:45 3:30 4:32 5:16 |
| final-dev | 141 | 219 | 37 | 0:45 1:50 2:48 3:31 4:21 5:24 |

所有 GT crop、pred crop 及派生直方图按 sample_id 严格同一侧。
test 封存。

## 4. 样本量评估与风险

**E1（纯 predicted-crop 微调）**：161 个 sync_ok 样本
（每样本 1–4 张直方图，估计 ~550 块）跨 3 类——约为原 W20 训练量
的 40%。**可行但偏小**，需较小学习率和强早停。

**E2（GT + predicted 混合微调）**：可额外引入 adaptation-train 样本的
**GT 框 crop**（val 信号在 Phase B′ 中未训练过，是新鲜数据），
扩大有效训练集。GT replay 比例建议预固定为 1:1（块级）。

**E3（GT box jitter 增强）**：从 fit split 原 Phase B′ 训练数据的 GT 框
出发施加几何扰动——成本最低，但扰动分布须从 adaptation-train 的
预测框误差分布推导（已存储在 adaptation_pairs.json 的
f_center_delta / t_center_delta / iou 字段）。

## 5. 已知限制

1. **overlap 分层缺失**：当前 manifest 只含 fit split 的 overlap 标志，
   val 样本的 overlap 需直接从 JSON 标注计算——已标记为待修项，
   不影响匹配/同步/划分的主数字。
2. **final-dev 37 个 sync_ok** 做评估的 CI 会较宽——这是 val 量级的
   物理约束，无法通过更好的划分解决。

## 6. 结论：是否满足进入 E1/E2/E3 的条件

| 条件 | 状态 |
| --- | --- |
| matched pairs 足够微调 | **是**（161 ok + GT crop 补充） |
| 类别覆盖均衡 | 基本是（W40 类略少但非零） |
| 文件级划分无泄漏 | 是（同一 sample 全部同侧） |
| final-dev 独立性 | 内部开发留出（非完全独立） |

**建议进入 E2（GT+pred 混合微调）为首选**——纯 E1 的 161 样本
风险过高；E3 作为低成本对照。是否继续由项目负责人决定。

## 产物

- `adaptation_audit.json`：全量审计数据
- `adaptation_pairs.json`：逐配对记录（sample_id, GT/pred 标签,
  IoU, 几何偏差, 同步状态）
- `scripts/baseline3ca_data_audit.py`：审计脚本
