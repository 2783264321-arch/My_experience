# Baseline-3 · SRRC Stage 0（已审计归档）

**状态：PAUSED — Stage 0 audited, no expert training started**

2026-11-21。独立目录，供科研审查。**本轮不训练**；test 封存；
SRRC 专家研发暂停，Stage 1/Stage 2 均未启动。

## 文件

| 文件 | 内容 |
| --- | --- |
| `stage0_errata.md` | 本轮 5 项审计修正：原始错误、依据、旧/新值、影响、未决限制 |
| `srrc_literature_audit.md` | AMC-Net/AWN/其他 IQ 基线、代码/权重、输入协议、Q1–Q4 |
| `srrc_data_audit.md` | class10/11 fit/val 全量统计、时长、交叠、YOLO AP、标签质量、Q5–Q6 |
| `srrc_experiment_protocol.md` | Stage 1 历史草案（**当前不执行**） |
| `srrc_data_audit.json` | 机器可读审计数字（含 full/old/new AP、映射核验） |
| `srrc_instances.csv` | fit+val 全部 class10/11 逐实例清单（含 split/带宽/交叠） |

## 执行摘要（修订后）

- **样本数量**：fit 全量 class10/11 **2,752 实例 / 2,179 文件**；
  其中 10 MHz 档 **1,824 / 1,572**（10: 941/868，11: 883/824）。
  val 全量 **716 / 576**；10 MHz 档 **469 / 404**
  （10: 239/220，11: 230/209）。类别平衡，文件级划分可行。
- **窄框比例**：fit 928/2,752 = **33.7%**；val 247/716 =
  **34.5%**。两者几乎相同；旧版“val 窄框显著高于 fit”是
  `frames.npz` 检索范围伪影，已撤销。
- **交叠**：10 MHz 档 any-overlap 78–82%，强交叠 45–53%；
  约一半与 Wi‑Fi 框相交。
- **YOLO AP**：all_labels 0.7079/0.5711；旧版“删除非 10 MHz
  GT、保留全部预测”= 0.3045/0.2377（**已知有偏的历史协议**）；
  新版 ignore-aware = **0.4318/0.3432**（+12.72/+10.55 pp）。
  这不是分类精度，仍受检测/定位与窄框长尾影响。
- **标签质量**：有效 Phase A′ 证据为 10 个真实样本：**5/10**
  通过 SRRC-like 结构门限，**6/10** 有约 8.333 MHz 符号率线，
  class10/11 的 m4 呈方向性差异。未通过门限≠标签错误；10 个
  样本不足以估计全量；**没有有效的全量结构恢复率统计**。
  不能认定大多数标签错误。
- **文献结论**：AMC-Net 是成熟论文方法及官方代码，适合作为
  架构/代码主参考；AWN 是成熟论文方法及官方代码，2×1024 权重
  适合长窗预训练对照；**简单 direct-IQ CNN/ResNet 是项目自定义
  工程基线，不是某篇论文的忠实复现**。首个主基线仍建议用简单
  direct-IQ CNN/ResNet，而非直接套预训练模型。
- **下一步**：SRRC 专家研发暂停。若恢复，先做 class10 vs 11
  二分类试点 + label-shuffle/窄框排除/clean-vs-overlap 对照；
  若二分类接近随机，先修标签/数据，不扩大模型。

## 关键数字

| 项 | fit | val |
| --- | ---: | ---: |
| manifest 文件 | 5,957 | 1,543 |
| 含 class10/11 文件 | 2,179 | 576 |
| all_labels 实例 | 2,752 | 716 |
| 10 MHz 实例 | 1,824 | 469 |
| 10 MHz 文件 | 1,572 | 404 |
| class10 10MHz 实例/文件 | 941 / 868 | 239 / 220 |
| class11 10MHz 实例/文件 | 883 / 824 | 230 / 209 |
| 窄框（<9.5 MHz）实例 | 928 | 247 |
| 时长中位（10/11, 10MHz） | 25 / 20 ms | 21 / 22.5 ms |
| any-overlap（10/11, 10MHz） | 78.0% / 80.1% | 82.0% / 80.0% |
| strong-overlap（10/11, 10MHz） | 45.9% / 47.0% | 52.7% / 44.8% |
| C0 AP50 all_labels（10/11） | — | 0.7079 / 0.5711 |
| C0 AP50 10MHz old drop-GT（历史有偏） | — | 0.3045 / 0.2377 |
| C0 AP50 10MHz ignore-aware（主口径） | — | **0.4318 / 0.3432** |

## 审核发现的问题与修订状态

| 问题 | 状态 |
| --- | --- |
| Wi-Fi 冻结文档 40 MHz 写成 96 data（实际 98） | 已修正；权重/mAP 不变 |
| SRRC 引用了已撤销的 Phase A 全量物理普查 | 已从有效结论移除；仅作历史记录 |
| SRRC fit 统计只枚举 `frames.npz` 的 10 MHz 子集 | 已改为全量 manifest 枚举；旧/新值见 errata |
| 10 MHz 子集 AP 删除 GT 但保留预测，产生有偏 FP | 已改为 ignore-aware；旧值保留为历史协议 |
| 自定义 CNN 被称为论文复现 | 已明确为项目自定义工程基线 |
| Stage 0 状态未标记暂停 | 已标记 PAUSED；Stage 1 协议仅历史草案 |

## 已验证结论（当前可引用）

1. fit/val 的 10 MHz 档数量与文件数：**1,824/1,572** 与
   **469/404**，与旧版一致（已核验）。
2. full-label C0 AP50：class10 0.7079、class11 0.5711。
3. 10 MHz ignore-aware AP50：class10 **0.4318**、class11
   **0.3432**；旧 drop-GT 值 0.3045/0.2377 仅作历史对照。
4. JSON 标注与 GT 缓存映射：1,543/1,543 val 样本的 class、
   顺序、几何完全一致。
5. ignore-aware 评估器在“所有 GT 有效”时与生产 full-label AP
   完全一致（class10 0.7079000376、class11 0.5710524866）。
6. Phase A′ 部分真实样本存在 SRRC-like 结构（5/10 通过门限，
   6/10 有 8.333 MHz 线）。

## 尚未验证（不得当作结论）

1. 全量 class10/11 的 SRRC 结构恢复率/标签可靠率。
2. 10 MHz 子集的实际调制分类精度（AP 不是分类精度）。
3. 预训练 AMC-Net/AWN 权重在 SpaceNet 域的可迁移性（未下载、
   未复现）。
4. 任何 SRRC 专家模型的性能——**没有训练**。

## 来源与复现

- 外部代码/论文链接见 `srrc_literature_audit.md` 第 7 节。
- 本地审计：`scripts/baseline3srrc_stage0_data_audit.py`；
  ignore-aware AP：`scripts/srrc_ignore_ap.py`；
  单元测试：`scripts/test_srrc_ignore_ap.py`。
- 原始数据只读；未提交原始 IQ/权重；未使用 test。
- 修订记录：`stage0_errata.md`。

## 后续

2026-11-21 的 fit-only 局部 IQ 分类探针见
[`reports/baseline3_srrc_local_iq/`](../baseline3_srrc_local_iq/README.md)：
未接 YOLO、未用 val/test；结论是未同步幅度统计已接近恢复幅度，
相位差无可重复收益，恢复率是主瓶颈。Stage 0 仍保持归档状态。


### YOLO 框离线复核

[`reports/baseline3_srrc_yolo_offline/`](../baseline3_srrc_yolo_offline/README.md)：
净 +3 个匹配框、mAP −0.238 pp，建议收尾。
