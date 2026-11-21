# SRRC Stage 0 · 审计勘误与归档记录

2026-11-21。对应审阅基准 `6c6010e`。本轮只做勘误、只读统计和
指标重算；**没有训练任何模型、没有使用 test、没有修改原始 IQ/
YOLO/专家权重**。SRRC Stage 0 状态：**PAUSED — Stage 0 audited,
no expert training started**。

## 1. Wi-Fi 冻结基线的 40 MHz 子载波数量

- **原始错误**：`wifi_expert_frozen_baseline.md` 写
  `40 MHz → 128-FFT（104 occupied / 96 data）`。
- **修正依据**：`scripts/baseline3b_ofdm_frontend.py::legacy_masks(128)`
  实际返回：
  - occupied = `−52…−1, +1…+52` = **104**；
  - pilots = `±11, ±25, ±39, ±53`；其中 `±53` 不在 occupied 集合；
  - data = occupied − pilots = 104 − 6 = **98**。
- **修改文件**：
  - `reports/baseline3_wifi_cascade/wifi_expert_frozen_baseline.md`；
  - `reports/baseline3_wifi_expert/frontend_validation.md`；
  - `scripts/baseline3b_ofdm_frontend.py` 模块 docstring（原写
    “104 occupied (96 data + 8 pilots)”，已改为 98 data + 6
    occupied pilots，并注明 ±53 不在 occupied）。
- **旧值 → 新值**：96 data → **98 data**；104 occupied 不变。
- **是否改变科研判断**：**否**。现有训练与推理使用相同的
  48/98-bin 特征定义，已报告的 mAP、权重和冻结基线仍然有效。
  这属于与原论文不同的、已记录的适配实现。**不能**用
  “histogram per-instance max normalization”声称 48/98 与其他
  子载波集合在数值上等价——归一化不保证信息等价。
- **未决限制**：原论文的 52-bin/导频处理与本实现 48/98-bin
  的差异仍是已记录偏差；本轮不重训、不重评。

## 2. SRRC 引用了已撤销的 Phase A 全量物理普查

- **原始错误**：Stage 0 文档把 `phase_a_physics.md` 的全量普查
  数字当作当前标签质量证据：
  - 95.6% 无预期符号率线；
  - 63% 近纯音；
  - 56–57% 单音；
  - `fill10` 中位 0.022；
  - `kurt_time` 中位 2.02；
  - 并据此推断“绝大多数 SRRC 标签不可靠”。
- **修正依据**：`phase_a_physics.md` 已在文件头正式勘误：旧版
  数字下变频参考错误（`Δf=f_c−f_lo`，正确为
  `f_c−(f_lo+B/2)`），约 100% 的 SRRC 帧测量了偏移 B/2 的错误
  频带；其全量物理结论已撤销。正确坐标下的有效结果在
  `phase_ap_srrc.md` 与 `phase_ap_report.md`。
- **修改文件**：
  - `reports/baseline3_srrc_stage0/README.md`；
  - `srrc_data_audit.md`；
  - `srrc_literature_audit.md`；
  - `srrc_experiment_protocol.md`。
- **旧结论 → 新结论**：
  - 旧：“全量普查显示大多数标签不可靠” → **撤销**；
  - 新：有效 Phase A′ 只覆盖 10 个真实样本：**5/10 通过
    SRRC-like 结构门限**，**6/10 有约 8.333 MHz 符号率线**，
    class10/11 的 m4 呈方向性差异；**未通过门限≠标签错误；
    10 个样本不足以估计全量；目前没有有效的全量结构恢复率
    统计**。SpaceNet 的 SRRC 类别已在部分真实 IQ 中观察到可
    恢复的单载波结构；不能认定大多数标签错误。
- **是否改变科研判断**：**是**。Stage 0 不能再把“标签普遍错误”
  当作既定事实；只能把“标签可靠性未验证”作为风险。
- **未决限制**：全量 SRRC 结构恢复率/标签可靠率仍未知；需要
  未来的分层筛查，但**本轮不启动**。
- **历史记录**：`phase_a_physics.md` 及其勘误保留，不删除；
  上述旧数字只作为已撤销历史记录，不进入当前结论。

## 3. SRRC fit 全量标签统计的检索范围

- **原始错误**：`baseline3srrc_stage0_data_audit.py` 从
  `reports/local_iq_round1/frames.npz` 建立 fit 的 SRRC 文件
  列表。该表只覆盖之前选中的 10 MHz 档实例，遗漏了仅包含其他
  带宽 class10/11 标注的 fit 文件。
- **修正依据**：改为 fit 与 val 都直接枚举
  `split_manifest.csv` 对应 `split_new` 的**全部文件**，逐文件
  解析 JSON，提取全部 class10/11 实例，再筛 9.5–10.5 MHz。
- **修改文件**：
  - `scripts/baseline3srrc_stage0_data_audit.py`；
  - `srrc_data_audit.json`；
  - `srrc_instances.csv`；
  - `README.md`、`srrc_data_audit.md`、`srrc_literature_audit.md`、
    `srrc_experiment_protocol.md`。
- **旧值 → 新值**：

| 项 | 旧值 | 新值 | 是否变化 |
| --- | ---: | ---: | --- |
| fit all_labels 实例 | 2,093 | **2,752** | 变化（旧值低估） |
| fit all_labels 文件 | 1,572 | **2,179** | 变化（旧值低估） |
| fit 10 MHz 实例 | 1,824 | **1,824** | **不变（已核验）** |
| fit 10 MHz 文件 | 1,572 | **1,572** | **不变（已核验）** |
| val all_labels 实例 | 716 | **716** | 不变 |
| val all_labels 文件 | 576 | **576** | 不变 |
| val 10 MHz 实例 | 469 | **469** | **不变（已核验）** |
| val 10 MHz 文件 | 404 | **404** | **不变（已核验）** |
| fit 窄框（<9.5 MHz） | 269（12.9%） | **928（33.7%）** | 变化（旧值低估） |
| val 窄框（<9.5 MHz） | 247（34.5%） | **247（34.5%）** | 不变 |

- **是否改变科研判断**：**是（部分）**。旧版“val 的窄框比例
  显著高于 fit”不成立：新版 fit 33.7% vs val 34.5%，两者几乎
  相同；旧版差异是 fit 检索范围伪影。10 MHz 档的 1,824/469
  与文件数 1,572/404 不变，因此基于 10 MHz 档的时长/交叠/
  文件分组结论仍然有效。
- **未决限制**：窄框 class10/11 是否应视为 SRRC 10 MHz 仍未定；
  本轮只统计，不训练、不合并标签。

## 4. SRRC 10 MHz 子集 AP 评估

- **原始错误**：旧脚本删除非 10 MHz 的 class10/11 GT，但保留
  全部检测预测；原本可匹配被排除 GT 的预测变成 FP，导致
  0.3045/0.2377 的低估。
- **修正依据**：实现独立 ignore-aware 评估器
  `scripts/srrc_ignore_ap.py`，不修改
  `src/signals/metrics.py` 的生产行为。规则：
  1. 保留 9.5–10.5 MHz class10/11 GT 为有效目标；
  2. 非该带宽 class10/11 GT 为 ignored；
  3. 有效 GT 优先一对一匹配；
  4. 未匹配有效 GT、但与 ignored GT IoU≥0.5 的预测不计 TP/FP；
  5. 其他未匹配预测仍计 FP；
  6. 原始预测框/分数/类别不变；
  7. IoU=0.5、101-point AP、max_dets=100 不变；
  8. JSON 与 GT 缓存用 class+顺序+归一化几何严格核验
     （1,543/1,543 样本 exact，atol=2e-4），不使用未核对的
     `sigs[j]`。
- **修改文件**：
  - `scripts/srrc_ignore_ap.py`；
  - `scripts/test_srrc_ignore_ap.py`；
  - `scripts/baseline3srrc_stage0_data_audit.py`；
  - `srrc_data_audit.json`、`srrc_instances.csv`；
  - `README.md`、`srrc_data_audit.md`、`srrc_literature_audit.md`、
    `srrc_experiment_protocol.md`。
- **旧值 → 新值**：

| class | 旧 drop-GT AP50 | 新 ignore-aware AP50 | 差值 |
| ---: | ---: | ---: | ---: |
| 10 | 0.3045 | **0.4318** | **+12.72 pp** |
| 11 | 0.2377 | **0.3432** | **+10.55 pp** |

- 新统计：class10 有效 GT 239、ignored GT 137、预测 4,338、
  TP 219、FP 3,983、ignored 预测 136；class11 有效 GT 230、
  ignored GT 110、预测 3,368、TP 179、FP 3,086、ignored 预测 103。
- 单元测试：9/9 通过（全 GT 有效时与参考 AP 一致；仅命中
  ignored GT 不计 TP/FP；命中有效 GT 计 TP；重复预测；有效/
  ignored 重叠；无 GT；输入不变；旧协议偏差）。
- 额外真实数据回归：把 ignored 集合设为空（所有 GT 有效）时，
  class10/11 的 ignore-aware AP 与生产 `evaluate` 的 full-label
  AP 完全一致：0.7079000376 / 0.5710524866（diff=0.0）。
- **是否改变科研判断**：**是（数值）**。旧 0.3045/0.2377 只能
  作为已知有偏的历史协议结果；新版 0.4318/0.3432 才是当前
  10 MHz 子集 AP。**AP 仍不是调制分类精度**，低值仍反映检测/
  定位与窄框长尾。
- **未决限制**：ignore 规则是“any ignored GT IoU≥0.5 即排除”，
  不做 ignored GT 一对一；这会在重复预测上比严格一对一忽略
  更多 FP。规则已写入评估器 docstring 与测试；未来若改规则需
  重新预注册。

## 5. 文献复现与工程对照的性质

- **原始错误**：Stage 0 文档把“简单 direct-IQ CNN/ResNet”与
  AMC-Net/AWN 并列，容易被理解为论文复现。
- **修正依据**：
  - AMC-Net：成熟论文方法（ICASSP 2023）及官方代码；
  - AWN：成熟论文方法（TCCN 2023）及官方代码；
  - 简单 direct-IQ CNN/ResNet：**项目自定义的工程基线**，不是
    某篇论文的忠实复现。
- **修改文件**：`srrc_literature_audit.md`（新增 §4.1）、
  `README.md`、`srrc_experiment_protocol.md`。
- **旧表述 → 新表述**：“首个主基线 = 简单 CNN”仍保留，但明确
  它是工程下限，不是论文复现；不得把结果归因于任何论文方法。
- **是否改变科研判断**：**否（方法选择）**，但**是（归因/
  表述）**。
- **未决限制**：本轮不下载模型、不核验大型权重、不执行论文
  复现实验；这些属于未来工作，当前 SRRC 研发暂停。

## 6. 归档状态

- SRRC Stage 0：**PAUSED — Stage 0 audited, no expert training
  started**。
- `srrc_experiment_protocol.md` 仅作历史草案保留，当前不执行；
  本轮不再生成 Stage 1/Stage 2 实施计划。
- 已验证结论：10 MHz 档计数 1,824/1,572、469/404；full-label
  AP 0.7079/0.5711；ignore-aware AP 0.4318/0.3432；映射
  1,543/1,543；Phase A′ 5/10 通过、6/10 有 8.333 MHz 线。
- 尚未验证：全量 SRRC 结构恢复率/标签可靠率；10 MHz 子集的
  调制分类精度；预训练权重可迁移性；任何 SRRC 专家性能。
- 原始 `phase_a_physics.md`、旧报告、旧 JSON/CSV 的历史提交均
  保留；本轮只新增勘误与修订版统计，不删除历史。
