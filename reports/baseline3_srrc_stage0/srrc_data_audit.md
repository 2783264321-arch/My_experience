# SRRC Stage 0 · 数据审计（class 10/11，修订版）

**状态：PAUSED — Stage 0 audited, no expert training started。**

2026-11-21。只做资料/数据审计，不训练。目标：SpaceNet advanced
中的 class 10（10 MHz SRRC QPSK）与 class 11（10 MHz SRRC 16QAM）。
test 继续封存。**本文件已按 `stage0_errata.md` 修订：fit 统计改为
全量 `split_manifest.csv` 枚举；10 MHz 子集 AP 改为 ignore-aware。**

## 1. 口径

- 数据：`/root/autodl-fs/SpaceNet_Dataset/advanced/train/<stem>.{bin,json}`。
- 划分：`split_manifest.csv` 的 `split_new`；本报告只用 `fit` / `val`。
  **fit 与 val 都从 manifest 全量枚举文件**，不再从
  `frames.npz` 反推 fit 文件列表。
- **10 MHz 档**：标注框带宽 `9.5–10.5 MHz`。这是 prior Phase A′
  的 SRRC 10 MHz 口径，也是本报告的主口径。
- **all_labels**：JSON 中 `class=10/11` 的全部框，包括带宽
  0.3–1.6 MHz 的窄框长尾；这些窄框是否真是 SRRC 10 MHz 存疑。
- 交叠：目标框面积被其他框覆盖的最大比例；`srrc` 指与 class
  10/11 的交叠，`strong25` 指 any-overlap ≥25%。
- 样本清单：`srrc_instances.csv`（fit+val，含 `split`、`is_10mhz`、
  `bandwidth_category`、频率/时间、`dur_ms`、`bw_mhz`、交叠标志）。

## 2. 全量样本数（修订）

| 划分 | 口径 | class 10 | class 11 | 合计 | 独立文件（并集） |
| --- | --- | ---: | ---: | ---: | ---: |
| fit | all_labels | **1,394（1,240 文件）** | **1,358（1,202 文件）** | **2,752** | **2,179** |
| fit | **10 MHz** | **941（868 文件）** | **883（824 文件）** | **1,824** | **1,572** |
| val | all_labels | 376（337 文件） | 340（300 文件） | 716 | 576 |
| val | **10 MHz** | **239（220 文件）** | **230（209 文件）** | **469** | **404** |

- fit manifest 文件总数 **5,957**；其中含 class10/11 的文件
  2,179，含 10 MHz class10/11 的文件 1,572。
- val manifest 文件总数 **1,543**；含 class10/11 的文件 576，
  含 10 MHz class10/11 的文件 404。
- **10 MHz 档的 1,824 / 469、1,572 / 404 与旧版完全一致**
  （已核验，未变化）。
- 旧版 fit all_labels **2,093 实例 / 1,572 文件** 是 `frames.npz`
  的 10 MHz 子集文件列表造成的低估；新版 **2,752 实例 /
  2,179 文件** 才是全量。
- 窄框占比：fit **928 / 2,752 = 33.7%**（851 文件）；val
  **247 / 716 = 34.5%**（230 文件）。**“val 窄框比例显著高于 fit”
  不成立**；两者几乎相同。旧版 fit 12.9% vs val 34.5% 的差异是
  检索范围伪影。
- 类别基本平衡（fit 941 vs 883；val 239 vs 230，10 MHz 档）。

### 带宽分布

| 划分 | <9.5 MHz | 9.5–10.5 MHz | >10.5 MHz |
| --- | ---: | ---: | ---: |
| fit | 928 | 1,824 | 0 |
| val | 247 | 469 | 0 |

## 3. 目标持续时间（10 MHz 档，ms）

| 划分/类 | n | q05 | median | q95 | min–max |
| --- | ---: | ---: | ---: | ---: | --- |
| fit class10 | 941 | 4.8 | 25.0 | 80.0 | 1–125 |
| fit class11 | 883 | 5.0 | 20.0 | 76.0 | 1–117 |
| val class10 | 239 | 3.0 | 21.0 | 71.4 | 1–99 |
| val class11 | 230 | 4.0 | 22.5 | 75.0 | 1–124 |

中位 20–25 ms，q05 3–5 ms；足够取 1024/2048 点的局部 IQ 窗
（20 Msps 下 1024 点 = 51.2 µs；2048 点 = 102.4 µs）。

全量 all_labels 的中位时长：fit class10 24 ms / class11 21 ms；
val class10 21 ms / class11 20 ms。窄框长尾主要影响带宽，不改变
“时窗足够”的结论。

## 4. 交叠程度（10 MHz 档）

| 划分/类 | any | SRRC 交叠 | 强交叠 ≥25% | Wi‑Fi 交叠 |
| --- | ---: | ---: | ---: | ---: |
| fit class10 | 734（78.0%） | 157（16.7%） | 432（45.9%） | 424（45.1%） |
| fit class11 | 707（80.1%） | 128（14.5%） | 415（47.0%） | 418（47.3%） |
| val class10 | 196（82.0%） | 43（18.0%） | 126（52.7%） | 124（51.9%） |
| val class11 | 184（80.0%） | 36（15.7%） | 103（44.8%） | 108（47.0%） |

全量 all_labels 的 any-overlap 比例略低（fit class10 989/1394 =
70.9%，class11 1006/1358 = 74.1%）；10 MHz 档的 78–80% 仍是
主口径。**约 80% 的目标框与别的信号相交，46–53% 为强交叠；
约一半与 Wi‑Fi 框相交。** 直接 IQ 窗若只按 GT 框裁剪，会混入
其他信号；必须按 clean/overlap 分层报告。

## 5. 现有 YOLO（C0）AP50（修订）

全量 val（1,543 文件，C0 权重 `best_corrected.pt`）：

| 口径 | class 10 AP50 | class 11 AP50 |
| --- | ---: | ---: |
| all_labels（全部 class10/11 GT） | **0.7079** | **0.5711** |
| 旧版 10 MHz：删除非 10 MHz GT、保留全部预测 | 0.3045 | 0.2377 |
| **新版 10 MHz ignore-aware（主口径）** | **0.4318** | **0.3432** |

- 旧版把非 10 MHz 的 class10/11 GT 直接删除，原本可匹配这些 GT
  的预测变成 FP，因此 **0.3045/0.2377 是已知有偏的历史协议结果**，
  不能称为无偏的 10 MHz 子集 AP。
- 新版 ignore-aware：保留 9.5–10.5 MHz GT 为有效目标；非该带宽
  class10/11 GT 为 ignored；有效 GT 优先一对一匹配；未匹配有效
  GT、但与 ignored GT IoU≥0.5 的预测不计 TP/FP；其他未匹配预测
  仍计 FP。IoU=0.5、101-point AP、max_dets=100 不变。
- 新版比旧版：class10 **+12.72 pp**，class11 **+10.55 pp**。
- 统计：class10 有效 GT 239、ignored GT 137、预测 4,338、
  TP 219、FP 3,983、ignored 预测 136；class11 有效 GT 230、
  ignored GT 110、预测 3,368、TP 179、FP 3,086、ignored 预测 103。
- **JSON 标注与 GT 缓存映射已严格核验**：1,543/1,543 个 val 样本
  的 class、顺序、归一化几何全部一致（atol=2e-4）；没有未核对
  的 `sigs[j]` 假设。
- 仍然：**0.4318/0.3432 是“现有 C0 检测器在 10 MHz 子集上的
  AP”，不是 10 MHz SRRC 调制分类精度。** 低 AP 主要反映检测/定位
  与窄框长尾，不是分类器本身。

## 6. 标签质量：有效 Phase A′ 证据（修订）

**已撤销的旧 Phase A 全量数字不再作为当前标签质量证据。** 旧版
因数字下变频参考错误（约 100% 的 SRRC 帧测量频带偏移）已正式
撤销；其 95.6% 无符号率线、63% 近纯音、56–57% 单音、fill10
中位 0.022、kurt_time 中位 2.02 等只能作为历史记录，详见
[`stage0_errata.md`](stage0_errata.md)。

有效的 Phase A′ 结果（正确频率坐标）：

- 10 个真实 class10/11 样本中，**5/10 通过 SRRC-like 结构门限**
  （8.333 MHz 符号率音 + RC 形 PSD + W50≈R）。
- **6/10** 发现约 **8.333 MHz** 的符号率线，把滚降定标在 α≈0.2；
  10% 边缘宽度 9.42–9.80 MHz 支持 RC 边沿。
- class10 clean 帧 m4 ≈ 1.01–1.14（贴 QPSK 参考）；
  class11 clean 帧 m4 ≈ 1.39–1.79（方向符合 16QAM，但受低 SNR/
  非 QAM 星座影响）。
- 未通过门限的帧包括 overlap 污染、平坦 10 MHz 带/无符号率音、
  内容偏心/触捕获端等；**未通过门限不等于标签错误**。
- **10 个样本不足以估计全量标签质量**；目前**没有有效的全量
  结构恢复率统计**。

**核心结论**：SpaceNet 的 SRRC 类别已在部分真实 IQ 中观察到
可恢复的单载波调制结构；全量可恢复率和标签可靠性尚未验证，
不能认定大多数标签错误。

## 7. 是否足够监督训练与文件分组验证？

**数量上足够做试点；标签质量与交叠仍是限制。**

- fit 10 MHz：1,824 实例 / 1,572 文件；val 469 / 404 文件；
  类别平衡。文件级 train/val 与 5-fold 文件分组 CV 可行。
- 有效独立单位是**文件**，不是实例；同文件多实例相关。
- 约 80% 交叠、46–53% 强交叠；clean 子集需分层。
- 标签可靠性未验证：10 个样本的 Phase A′ 只能证明“部分可恢复”，
  不能给出全量通过率；在全量筛查前，不能把 1,824 当作 1,824 个
  可信调制样本，也不能反过来认定大多数标签错误。
- 建议：先做 **class10 vs 11 二分类试点**（不是 14 类），
  以 clean 子集为主、overlap 子集为分层；同时跑 label-shuffle
  与窄框排除对照。若二分类都接近随机，优先做标签/数据审计，
  而不是加大网络。

## 8. 产物

- `srrc_data_audit.json`：逐类/逐口径计数、带宽分布、时长、
  交叠、full/old/new AP、映射核验、TP/FP/ignored 统计
- `srrc_instances.csv`：fit+val 全部 class10/11 实例清单
- 脚本：`scripts/baseline3srrc_stage0_data_audit.py`、
  `scripts/srrc_ignore_ap.py`、`scripts/test_srrc_ignore_ap.py`
- 修订记录：`stage0_errata.md`
- 关联审计：`reports/local_iq_round1/phase_ap_srrc.md`、
  `phase_ap_report.md`、`reports/spaceNet_class_ids_and_wifi_labels.md`
