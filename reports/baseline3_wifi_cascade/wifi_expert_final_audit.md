# Wi-Fi 专家收尾审计（E0/E1/E2/E3 勘误与补充统计）

2026-11-21。针对 `bb9d288` 的审阅意见，本文件记录四项审计修订。
**不重训、不改权重、不改变正式 mAP**；只修正描述与统计口径。

## 1. 冻结层说明勘误

**原表述**：“冻结 conv1/conv2，训练 conv3+FC”。实际行为：

- `FineTuneNet` 只把 `features[0:8]` 的**参数**设为
  `requires_grad=False`；conv1/conv2 的 weight/bias 确实不变。
- 训练循环使用 `model.train()`，因此 BatchNorm1/2 的
  `running_mean/running_var/num_batches_tracked` **继续更新**。
- 逐权重审计（`finetune_freeze_audit.json`，9 个 run）：
  - conv1/conv2 weight/bias：max abs diff = 0；
  - BN1/BN2 running stats：全部 changed，max abs diff 0.006–0.76；
  - `num_batches_tracked`：+800（100 step）或 +3200（400 step）。
- 因此正确表述是：**冻结低层卷积参数 + BN 统计适配**。
  这是一个无监督统计适配成分，不能把 E1/E2/E3 的差异唯一归因于
  高层监督更新。

详见 [`finetune_freeze_erratum.md`](finetune_freeze_erratum.md)；权重
SHA256 同文件。原始 E0/W-Shared 不受影响。

## 2. 配对 bootstrap mAP 区间（final-dev，2,000 次）

按 140 个 IQ `sample_id` 聚类、成对重采样（同一 resample 同时评估
C0/E0/E2，保留文件内所有预测/GT）。E0/E2 使用三 seed mAP 的均值，
与报告表格一致；mAP 为 14 类 class-aware IoU=0.5、101-point AP、
max_dets=100。快速评估器与参考 `evaluate` 的逐点差异 ≤3.3e-16。

| 差值 | 点估计 | bootstrap 均值 | 95% CI | sd |
| --- | ---: | ---: | --- | ---: |
| **E0 − C0** | **+1.2835 pp** | +1.2872 | **[−0.5845, +3.3042] pp** | 0.996 |
| **E2 − E0** | **+0.2351 pp** | +0.1719 | **[−0.3840, +0.7601] pp** | 0.293 |
| E2 − C0（单独估计） | +1.5186 pp | +1.4592 | [−0.5202, +3.5936] pp | 1.032 |

**必须区分 E2−C0 与 E2−E0**：两者使用同一 bootstrap 抽样，但估计
目标不同；E2−C0 的 CI 更宽，不能拿来代替 E2−E0。E2−E0 的 CI
包含 0，且点估计仅 +0.235 pp。

逐 seed E2−E0（点估计 / 95% CI）：

| seed | 点估计 | 95% CI |
| ---: | ---: | --- |
| 1729 | +0.806 pp | [−0.068, +1.818] pp |
| 1730 | −0.219 pp | [−0.605, +0.109] pp |
| 1731 | +0.118 pp | [−1.051, +1.064] pp |

方向不一致，进一步支持“无可复现增益”。JSON：
`paired_bootstrap_map.json`；脚本：`scripts/baseline3cj_bootstrap_map.py`。

## 3. all-instance 分类口径修正

此前“all-instance”把 216 个 matched pair 全部映射到同一个三分类
标签后计分。final-dev 中有 **1 个跨家族路由错误**：
`train/5510`，GT class 2（W20 64QAM）→ YOLO 预测 class 4
（W40 16QAM），且该预测未同步（无 Pred histogram）。把它映射为
三分类标签会把“带宽家族错误”混入调制准确率。

修正后的口径：

- **主口径（组内准确率）**：只统计正确家族候选（215 对；
  预测家族 = GT 家族）。同步失败回退到 YOLO 的同家族标签。
- **跨家族路由错误**：单独报告，不映射为三分类标签、不计为正确。
- **严格全匹配**：216 对，跨家族那 1 对直接计错；此前的
  0.5340/0.5309/0.5355/0.5293 实际是这个严格口径。

| 模型 | 正确家族组内准确率（215） | 严格全匹配（216） | 旧报告值（严格） |
| --- | ---: | ---: | ---: |
| E0 | **0.5364** | 0.5340 | 0.5340 |
| E1 | **0.5333** | 0.5309 | 0.5309 |
| E2 | **0.5380** | 0.5355 | 0.5355 |
| E3 | **0.5318** | 0.5293 | 0.5293 |

差值只有约 0.2–0.25 pp，因为 final-dev 只有 1 个跨家族对；但口径
必须改正。JSON：`finaldev_metric_audit.json`；脚本：
`scripts/baseline3ck_metric_audit.py`。

## 4. `n_relabeled` 修正

旧字段 `n_relabeled` 实际是 `len(label_map)` = **专家分类过的
sync-ok 预测数**（full-val 807 / final-dev 125），不是“实际改标签
数”。修正后分三层：

| 量 | full-val | final-dev |
| --- | ---: | ---: |
| Wi-Fi 候选（专家调用入口） | 4,844 | 704 |
| 同步成功（实际 CNN 分类） | 807 | 125 |
| 其中 matched / unmatched | 246 / 561 | 37 / 88 |
| **实际标签变化（E0，3 seed 范围）** | **502–514** | **84–86** |
| 实际标签变化（E1） | 516–517 | 85–89 |
| 实际标签变化（E2） | 506–517 | 84–87 |
| 实际标签变化（E3） | 509–514 | 83–86 |

- 实际变化中，full-val matched 约 124–131、unmatched 约 377–390；
  final-dev matched 约 20–23、unmatched 约 62–67。
- 家族变化（W20↔W40）恒为 **0**：专家只在预测家族内改调制阶数。
- 因此“专家调用数”“成功同步数”“实际修改标签数”必须分开，
  不能再用 807/125 当作修改数。

逐 arm/seed 的 matched/unmatched 分解、YOLO/专家标签分布见
`finaldev_metric_audit.json`。

## 5. 是否改变正式 mAP

**不改变。** 本轮只做：

- 权重冻结行为的后验审计（读权重，不训练）；
- 对已保存预测做 IQ-file bootstrap（快速评估器与参考评估器逐点
  一致，差异 ≤3.3e-16）；
- 重新定义分类统计口径（组内 vs 严格）；
- 重新统计实际改标签数。

模型权重、候选框、得分、NMS、级联标签均未修改。正式 mAP 仍为：

| 指标 | 值 |
| --- | ---: |
| C0 全量 val | 0.6497250 |
| C1/E0 全量 val 均值 | 0.6597 |
| C0 final-dev 子集 | 0.6634728 |
| E0 final-dev 子集 | 0.6763083 |
| E2 final-dev 子集 | 0.6786591 |

## 6. 审计结论

1. E1/E2/E3 的“冻结”应表述为“冻结 conv1/conv2 参数、BN 统计仍
   更新”；该勘误不改变任何数字，但限制机制归因。
2. E0−C0 在 final-dev 上的配对 CI 为 [−0.58,+3.30] pp，包含 0；
   全量 val 的 +1.00 pp 仍是主要历史证据，但 final-dev 单独不足以
   证明显著增益。
3. E2−E0 的 CI 为 [−0.38,+0.76] pp，包含 0；E2−C0 是另一个估计，
   不可混用。
4. 正确家族组内准确率是主口径；跨家族路由错误单独报告。旧报告
   的 all-instance 数字是严格全匹配口径，需按新表引用。
5. `n_relabeled` 不等于实际改标签数；full-val 实际变化约
   502–517 / 807，final-dev 约 83–89 / 125。
6. 继续保留冻结 W-Shared E0/C1；不采用 E1/E2/E3。完整基线记录见
   [`wifi_expert_frozen_baseline.md`](wifi_expert_frozen_baseline.md)。
