# Baseline-3 · SRRC Stage 0（资料与数据审计）

2026-11-21。独立目录，供下一轮科研审查。**本轮不训练**；test 封存。

## 文件

| 文件 | 内容 |
| --- | --- |
| `srrc_literature_audit.md` | AMC-Net/AWN/其他 IQ 基线、代码/权重、输入协议、Q1–Q4 |
| `srrc_data_audit.md` | class10/11 fit/val 计数、时长、交叠、YOLO AP、标签质量、Q5–Q6 |
| `srrc_experiment_protocol.md` | Stage 1 预注册协议、模型/对照、停止判据、Q7 |
| `srrc_data_audit.json` | 机器可读审计数字 |
| `srrc_instances.csv` | fit+val class10/11 逐实例清单（含 split/几何/交叠） |

## 执行摘要

- **样本数量**：fit 10 MHz 档 **1,824 实例 / 1,572 文件**
  （10: 941/868，11: 883/824）；val **469 / 404**
  （10: 239/220，11: 230/209）。类别平衡，文件级划分可行。
- **交叠**：fit any-overlap 78–80%，强交叠 46–47%；
  val 80–82%，强交叠 45–53%；约一半与 Wi‑Fi 框相交。
- **YOLO AP**：all_labels 0.7079/0.5711；**10 MHz GT 子集仅
  0.3045/0.2377**——窄框长尾显著抬高了表面 AP。
- **标签质量是主要风险**：prior SRRC 结构审计 5/10 帧通过；
  全量 1,824 帧中 95.6% 无预期符号率线、63% 近纯音、
  56–57% 单音；`fill10` 中位 0.022、`kurt_time` 中位 2.02。
  数量够做试点，但不能假设 1,824 个标签都对应可恢复的 SRRC
  调制。
- **文献结论**：AMC-Net 是最适合作为**架构/代码主参考**的
  ICASSP 2023 官方基线（MIT、代码/权重齐全），但输入仅 2×128、
  200 kHz 量级生成域；AWN 的 RML2018.01a 2×1024 权重更适合
  长窗预训练对照；**首个主基线应是简单的 direct-IQ CNN/ResNet
  （2×1024，从零训练）**，而不是先堆预训练大模型或手工符号同步。
- **下一步**：先做 class10 vs 11 二分类试点 + label-shuffle/
  窄框排除/clean-vs-overlap 对照；若二分类接近随机，先修标签/
  数据，不扩大模型。

## 关键数字

| 项 | fit | val |
| --- | ---: | ---: |
| 10 MHz 实例 | 1,824 | 469 |
| 10 MHz 文件 | 1,572 | 404 |
| class10 实例/文件 | 941 / 868 | 239 / 220 |
| class11 实例/文件 | 883 / 824 | 230 / 209 |
| 时长中位（10/11） | 25 / 20 ms | 21 / 22.5 ms |
| any-overlap（10/11） | 78.0% / 80.1% | 82.0% / 80.0% |
| strong-overlap（10/11） | 45.9% / 47.0% | 52.7% / 44.8% |
| C0 AP50 all_labels（10/11） | — | 0.7079 / 0.5711 |
| C0 AP50 10 MHz 子集（10/11） | — | 0.3045 / 0.2377 |

## 来源与复现

- 外部代码/论文链接见 `srrc_literature_audit.md` 第 7 节。
- 本地审计：`scripts/baseline3srrc_stage0_data_audit.py`；
  原始数据只读；未提交原始 IQ/权重。
- 相关 prior 审计：`reports/local_iq_round1/phase_ap_srrc.md`、
  `phase_a_physics.md`、`reports/spaceNet_class_ids_and_wifi_labels.md`。

## 结论

SRRC 10/11 **有足够数量做试点**，但**没有足够证据证明标签可监督**。
下一阶段应先做标签质量筛查和简单 direct-IQ 二分类，而不是直接
训练 AMC-Net/AWN 或重设计符号同步。
