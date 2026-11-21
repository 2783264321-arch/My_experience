# SRRC Stage 0 · 数据审计（class 10/11）

2026-11-21。只做资料/数据审计，不训练。目标：SpaceNet advanced
中的 class 10（10 MHz SRRC QPSK）与 class 11（10 MHz SRRC 16QAM）。
test 继续封存。

## 1. 口径

- 数据：`/root/autodl-fs/SpaceNet_Dataset/advanced/train/<stem>.{bin,json}`。
- 划分：`split_manifest.csv` 的 `split_new`；本报告只用 `fit` / `val`。
- **10 MHz 档**：标注框带宽 `9.5–10.5 MHz`。这是 prior Phase A′
  的 SRRC 10 MHz 口径，也是本报告的主口径。
- **all_labels**：JSON 中 `class=10/11` 的全部框，包括带宽
  0.3–1.6 MHz 的窄框长尾；这些窄框是否真是 SRRC 10 MHz 存疑，
  不应直接混入训练。
- 交叠：目标框面积被其他框覆盖的最大比例；`srrc` 指与 class
  10/11 的交叠，`strong25` 指 any-overlap ≥25%。
- 样本清单：`srrc_instances.csv`（fit+val，含 `split`、频率/时间、
  `dur_ms`、`bw_mhz`、any/srrc/wifi/cross/same/strong 标志）。

## 2. 样本数

| 划分 | 口径 | class 10 | class 11 | 合计 | 独立文件（并集） |
| --- | --- | ---: | ---: | ---: | ---: |
| fit | all_labels | 1,068（923 文件） | 1,025（887 文件） | 2,093 | 1,572 |
| fit | **10 MHz** | **941（868 文件）** | **883（824 文件）** | **1,824** | **1,572** |
| val | all_labels | 376（337 文件） | 340（300 文件） | 716 | 576 |
| val | **10 MHz** | **239（220 文件）** | **230（209 文件）** | **469** | **404** |

- 10 MHz 档占 all_labels 的 88.1%（fit class10）、86.1%（fit
  class11）；val 只有 63.6% / 67.6%——**val 的窄框长尾更多**。
- 类别基本平衡（fit 941 vs 883；val 239 vs 230）。
- 文件级划分可用：fit 1,572 文件、val 404 文件；同一文件的多个
  实例必须同侧。

## 3. 目标持续时间（10 MHz 档，ms）

| 划分/类 | n | q05 | median | q95 | min–max |
| --- | ---: | ---: | ---: | ---: | --- |
| fit class10 | 941 | 4.8 | 25.0 | 80.0 | 1–125 |
| fit class11 | 883 | 5.0 | 20.0 | 76.0 | 1–117 |
| val class10 | 239 | 3.0 | 21.0 | 71.4 | 1–99 |
| val class11 | 230 | 4.0 | 22.5 | 75.0 | 1–124 |

中位 20–25 ms，q05 3–5 ms；足够取 1024/2048 点的局部 IQ 窗
（20 Msps 下 1024 点 = 51.2 µs；2048 点 = 102.4 µs）。

## 4. 交叠程度（10 MHz 档）

| 划分/类 | any | SRRC 交叠 | 强交叠 ≥25% | Wi‑Fi 交叠 |
| --- | ---: | ---: | ---: | ---: |
| fit class10 | 734（78.0%） | 157（16.7%） | 432（45.9%） | 424（45.1%） |
| fit class11 | 707（80.1%） | 128（14.5%） | 415（47.0%） | 418（47.3%） |
| val class10 | 196（82.0%） | 43（18.0%） | 126（52.7%） | 124（51.9%） |
| val class11 | 184（80.0%） | 36（15.7%） | 103（44.8%） | 108（47.0%） |

**约 80% 的目标框与别的信号相交，46–53% 为强交叠；约一半与
Wi‑Fi 框相交。** 直接 IQ 窗若只按 GT 框裁剪，会混入其他信号；
必须按 clean/overlap 分层报告，不能把 overall accuracy 当作
调制识别能力。

## 5. 现有 YOLO（C0）AP50

全量 val（1,543 文件，C0 权重 `best_corrected.pt`）：

| 口径 | class 10 AP50 | class 11 AP50 | n_gt 10/11 | n_pred 10/11 |
| --- | ---: | ---: | ---: | ---: |
| all_labels | **0.7079** | **0.5711** | 376 / 340 | 4,338 / 3,368 |
| **10 MHz GT 子集** | **0.3045** | **0.2377** | 239 / 230 | 4,338 / 3,368 |

- 10 MHz 子集 AP 的算法：保留所有预测，只删除带宽不在
  9.5–10.5 MHz 的 class 10/11 GT；其他类 GT 保留。这样窄框不再
  提供“容易匹配”的目标，AP 大幅下降。
- 这说明 **all_labels AP 被窄框长尾显著抬高**；真正 10 MHz SRRC
  的检测/定位质量远低于表面数字。下一阶段若用 YOLO 框裁剪，
  必须按 10 MHz 子集重新评估，不能引用 0.7079/0.5711 作为
  SRRC 定位质量。

## 6. 标签质量：比样本数更关键的瓶颈

Prior Phase A′ 的 SRRC 结构审计（`phase_ap_srrc.md`）在 10 个
真实 class10/11 帧上：

- 仅 **5/10** 通过 SRRC 门限（8.333 MHz 符号率音 + RC 形 PSD
  + W50≈R）；
- 6 帧有精确 8.333 MHz 线（α≈0.2），但 90% 平坦段有 ≥10%
  涟漪；α 宽度估计过读；
- 2 个 overlap 帧污染，`train/1160` 为平坦 10 MHz 带/无符号率音/
  近 Rayleigh，`train/123` 内容偏心 −2.27 MHz 且框触捕获端；
- class11 clean 的 m4 均值 1.649 > class10 1.276，方向符合
  QPSK/16QAM，但每组 n=4 且含失配帧。

全量 1,824 个 fit 10 MHz 帧的普查（`census_all.npz`）：

- `fill10` 中位 0.022；34.8% 的 `fill10<0.01`；63% 的框为
  “近纯音”（x² 线 >100）；56–57% 单音框；
- `x2_line_str` 中位 317（极高）；`kurt_time` 中位 2.02
  （Rayleigh-like），不是典型 QPSK/16QAM 的常包络/离散幅度；
- 95.6% 的帧在 R=bw/1.2 处没有预期谱线；三候选全败 82.5%。

**结论：样本数量足够做“试点”，但标签有效性未知。** 这些统计
不能证明所有 class10/11 标签都错，但足以要求在监督训练前做
**分层标签质量筛查**；否则模型可能在学习窄带音/捕获伪影，而不是
SRRC 调制阶数。

## 7. 是否足够监督训练与文件分组验证？

**数量上足够做试点，质量上必须先审计。**

- fit 10 MHz：1,824 实例 / 1,572 文件；val 469 / 404 文件；
  类别平衡。文件级 train/val 可行，且可做 5-fold 文件分组 CV。
- 但有效独立单位是 **文件**，不是实例；同文件多实例相关。
- 约 80% 交叠、46–53% 强交叠；clean 子集很小（需按
  any/strong 分层）。
- 标签质量未验证：在筛查前，不能把 1,824 当作 1,824 个可信
  调制样本。若 SRRC 门限通过率只有 ~50%，有效样本可能减半，
  且剩余样本仍有 overlap 污染。
- 建议：先做 **class10 vs 11 二分类试点**（不是 14 类），
  以 clean 子集为主、overlap 子集为分层；同时跑 label-shuffle
  与窄框排除对照。若二分类都接近随机，优先修标签/数据，而不是
  加大网络。

## 8. 产物

- `srrc_data_audit.json`：逐类/逐口径计数、时长、交叠、AP
- `srrc_instances.csv`：fit+val 全部 class10/11 实例清单
- 脚本：`scripts/baseline3srrc_stage0_data_audit.py`
- 关联审计：`reports/local_iq_round1/phase_ap_srrc.md`、
  `phase_a_physics.md`、`reports/spaceNet_class_ids_and_wifi_labels.md`
