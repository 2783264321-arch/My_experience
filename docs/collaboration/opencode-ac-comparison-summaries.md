# A/C 对照三轮追问汇总：主对照 · 家族级指标 · 被忽略结论

2026-11-20，OpenCode。项目负责人对已完成的 Group A（全频单图独立训练）与
Baseline-1（粗图 + 频率细片）对照连续三轮追问，本文把三次回答合并成一份记录。
**本轮全部为既有工件的再分析，没有新训练、没有新推理**；test split 继续封存，
全部数值来自固定 1,543 个验证样本、产生 0.6427 的同一套权重与验证口径。唯一的
新计算是家族级组过滤指标，已入库
[family_group_metrics_ab_c.json](../../reports/family_group_metrics_ab_c.json)。

四臂代号：**A** = 全频单图独立训练、coarse-only 推理（主口径 last=ep99 EMA，
best=ep29 另报）；**B** = 同一联合权重仅跑粗图；**C** = 同一联合权重跑
粗图 + 细片全视图。A 取自 worktree `stage5_val{,_best}`，B/C 取自回归门复现
`runs/regression/c_coarse`、`runs/regression/c_all`（B 0.5794、C 0.6427 与
历史口径逐位一致）。

## 一、主对照复述

### 总体

| 指标 | A last | A best | B coarse | C all |
| --- | ---: | ---: | ---: | ---: |
| mAP@0.5（严格 14 类） | 0.5404 | 0.5763 | 0.5794 | **0.6427** |
| AP50 agnostic（只看定位） | 0.8020 | 0.8208 | 0.8178 | **0.9046** |
| localization recall | 0.8570 | 0.8789 | 0.8773 | **0.9580** |
| classification accuracy | 0.8328 | 0.8270 | 0.8278 | **0.8433** |
| predictions | 16,173 | 30,981 | 25,409 | 65,025 |
| val views | 1,543 | 1,543 | 1,543 | 24,975 |

关键差值：**C−A last +10.23 pp**（论文主口径，末轮对末轮）；**C−A best
+6.64 pp**（更强单图参照）；**C−B +6.33 pp**（同权重加细片的净收益）；
B−A best +0.31 pp（单 seed 下应视为无差异，见三.9）。

### 类别组与带宽组（A best / B / C）

| 类别组 | A best | B | C | C−A best |
| --- | ---: | ---: | ---: | ---: |
| WiFi 20MHz 0/1/2 | 0.3839 | 0.4212 | 0.4020 | +1.81 pp |
| WiFi 40MHz 3/4/5 | 0.4285 | 0.4550 | 0.4415 | +1.30 pp |
| SRRC 10/11 | 0.6399 | 0.5908 | 0.6395 | −0.04 pp |
| 窄带 9/12/13 | 0.5002 | 0.4838 | **0.7804** | **+28.02 pp** |

带宽组：narrow ≤0.25 MHz 同上（+28.02 pp）；medium 0.3–2 MHz A best 0.8917
> C 0.8388（−5.29 pp，**归因已在三.6 修正**）；wide / very wide / ultra 上 C
仅 +0.91 / +1.81 / +1.30 pp，且三组均为 B 最高。明细见
[group_a_results.md](../../reports/group_a_results.md) 与
[group_a_narrowband_check.json](../../reports/group_a_narrowband_check.json)。

### 预算与成本

- 等预算：A 100 轮 × 1 视图 vs C 50 轮 ×（1 粗 + 1 细），图像呈现数同为
  595,700，optimizer steps 37,300 vs 37,250；墙钟 A 1.62 h（另有 2,050 s /
  11.63 GiB BF16 缓存）vs C 2.69 h；峰值显存均 10.1 GiB。
- 延迟（128 样本 × 3 次重复）：A 1 视图端到端 p50 0.327 s（构建 0.317 s +
  模型 0.010 s）；C 19.10 视图 0.509 s（0.422 + 0.086）。model-only 8.64×，
  端到端仅 **1.56×**——贵的不是切片，是底谱图；每加一个细片的边际成本约
  6 ms 构建 + 4.5 ms 推理。

## 二、家族级（"WiFi 算一大类"）与类别无关指标

两种口径先分开：**组内小类 AP 平均**（仍要求 14 类小类正确）与**家族坍缩
AP**（GT 与预测限制在组内类别后按类别无关评测定框，等价于"把组当一个大类
只看定位"）。后者统一用组过滤评测器的 `ap50_agnostic`，四臂 × 8 家族数值入库
[family_group_metrics_ab_c.json](../../reports/family_group_metrics_ab_c.json)。

### 家族坍缩 AP@0.5（只看定位）

| family | n GT | A best | B | C |
| --- | ---: | ---: | ---: | ---: |
| WiFi 全部 0–5 | 1,439 | 0.8245 | **0.8415** | 0.8327 |
| WiFi 20MHz 0–2 | 997 | 0.7859 | **0.8270** | 0.8165 |
| WiFi 40MHz 3–5 | 442 | **0.8814** | 0.8630 | 0.8554 |
| BLE/Zigbee 6–8 | 2,847 | 0.9532 | 0.9519 | **0.9548** |
| LoRa 9 | 1,411 | 0.6427 | 0.6308 | **0.9484** |
| SRRC 10–11 | 716 | **0.8901** | 0.8791 | 0.8795 |
| AM/FM 12–13 | 749 | 0.4375 | 0.4223 | **0.6982** |
| 窄带 9/12/13 | 2,160 | 0.5847 | 0.5727 | **0.8695** |

家族定位召回同趋势：C 的增益全部落在 LoRa（0.7477→0.9780）、AM/FM
（0.5007→0.7810）、窄带（0.6792→0.9162）；WiFi 家族召回 B=C=0.9680，仅比
A best 0.9555 高 1.25 pp。

### 家族坍缩 vs 组内小类：gap 定位瓶颈

| family | A best（坍缩 / 小类 / gap） | B | C |
| --- | --- | --- | --- |
| WiFi 0–5 | 0.8245 / 0.4062 / **41.8 pp** | 0.8415 / 0.4381 / 40.3 pp | 0.8327 / 0.4217 / 41.1 pp |
| BLE/Zigbee 6–8 | 0.9532 / 0.9499 / 0.3 pp | 0.9519 / 0.9499 / 0.2 pp | 0.9548 / 0.9490 / 0.6 pp |
| SRRC 10–11 | 0.8901 / 0.6399 / 25.0 pp | 0.8791 / 0.5908 / 28.8 pp | 0.8795 / 0.6395 / 24.0 pp |
| AM/FM 12–13 | 0.4375 / 0.4289 / 0.9 pp | 0.4223 / 0.4103 / 1.2 pp | 0.6982 / 0.6965 / 0.2 pp |

匹配框内组内分类准确率：WiFi ≈ 0.47–0.48（B 与 C 精确相同），BLE ≈
0.98–0.99，LoRa = 1.0000，SRRC ≈ 0.66–0.68，AM/FM ≈ 0.98–1.00，窄带 ≈
0.95–0.98。

结论：**"大类之间好分、大类内部小类难分"成立**，集中表现为 WiFi 家族
~41 pp 与 SRRC ~24–29 pp 的坍缩−小类 gap；AM/FM 的瓶颈相反，是定位（坍缩≈
小类，一旦定位到小类几乎必对）；BLE/Zigbee 大小类都接近 0.95。C 的
agnostic 增益来自 LoRa / AM/FM / 窄带，**不来自 WiFi**。

## 三、被忽略的结论（十项）

### 1. oracle 分解：分类单独损失 22.1 pp，且全部集中在 WiFi/SRRC

`stage5_val/oracle_class_ap.json`：把每个 IoU≥0.5 匹配框直接改成正确类别，

| 口径（C） | mAP@0.5 |
| --- | ---: |
| 实际 | 0.6427 |
| oracle：匹配框类别全对 | **0.8639**（+22.1 pp） |
| agnostic：只看定位 | 0.9046（再 +4.1 pp 为去重/排序残余） |

分家族的分类损失：WiFi 0–5 每类 **33–49 pp**（oracle 0.81–0.88 vs 实际
0.33–0.52）；SRRC 10/11 为 19 / 31 pp；AM/FM、LoRa、BLE 均 ≤1.2 pp。这是
"分类是主瓶颈"的最硬量化，也是后来复数前端研究的原始动机。

### 2. 混淆矩阵：跨家族 ≈ 0，问题全在家族内部

actual 混淆矩阵几乎完全块对角（WiFi GT 误到 BLE/LoRa/AM 的次数 0–2 / 约
300）。家族内失败模式：**WiFi 16QAM 是主要失败点**（W20-16：160 次→QPSK
> 123 次正确；W40-16：67→QPSK vs 52 正确）；W20-64 基本可分（153 正确）、
W40-64 勉强（58 vs 48+27）；BLE 内 LE2M↔Zigbee 互误 12/25；SRRC Q↔16QAM
互误 82/133；**AM/FM 误成的是 LoRa（6/8 次）而不是彼此**。

### 3. 同位置多类堆叠：3,800 组，同类重复 0

`stage5_val/error_analysis.json`：同一位置检出多张不同类别框的组 3,800 个，
同类重复 0（class-wise NMS 已清干净同类）。分类不确定性具体表现为跨类堆叠框，
class-wise NMS 无法去除。推论：位置级去重（agnostic NMS / 每位置保留最高分
类别）是**上界 ~4 pp** 的工程杠杆；更好的分类特征是 **~22 pp** 的杠杆，可叠加。

### 4. 细片分支 = 窄带专家 + 宽带噪声源

[view_source_attribution.md](../../reports/view_source_attribution.md)、
`stage5_val/view_source_attribution.json`：

| 类别 | coarse-only | fine-only | both |
| --- | ---: | ---: | ---: |
| WiFi 0–5 | 0.37–0.55 | **0.0000（六类全部）** | 0.33–0.52（低于 coarse-only） |
| SRRC 11 | 0.610 | 0.091 | 0.571（−3.9 pp） |
| SRRC 10 | 0.572 | 0.340 | 0.708 |
| LoRa 9 | 0.631 | 0.952 | 0.948 |
| AM 12 / FM 13 | 0.509 / 0.312 | 0.738 / 0.657 | 0.739 / 0.654 |
| BLE 6 / 7 / 8 | 0.951 / 0.955 / 0.943 | 0.827 / 0.430 / 0.395 | 0.959 / 0.947 / 0.941 |

- fine-only 总体 mAP50 仅 **0.3165**；fine 视图 45,629 条预测中 **40%
  （18,496）是 AP=0 的 WiFi 预测**。
- 细片对 WiFi 匹配集零贡献：B 与 C 的 WiFi 家族 cls acc / recall 精确相同
  （0.4752 / 0.9680）。
- 边界过滤删掉的 202,131 条原始检出中 **84% 来自 0–5 + 10/11**（WiFi
  139,287），窄带仅 2.1%——它主要在清理细片在看不见的类别上产生的垃圾。
- 这正是 fine 0–5 过滤 +0.7 pp（0.6427→0.6497）的机理。

### 5. 真正的定律是相对带宽 ~0.5–2% 交叉

粗图 2048 行上信号只占 rel×2048 行：LoRa / AM / FM 中位 0.3–0.4% → **约
6–8 行**；BLE 2.5–5% → 51–102 行；WiFi 67–80%。

| rel_bw | n GT | A best | C | C−A best |
| --- | ---: | ---: | ---: | ---: |
| ≤0.5% | 1,405 | 0.2527 | **0.6728** | **+42.0 pp** |
| 0.5–2% | 1,025 | 0.8356 | 0.8600 | +2.4 pp |
| 2–5% | 1,680 | **0.7987** | 0.7237 | −7.5 pp |
| 5–20% | 1,074 | **0.7128** | 0.7014 | −1.1 pp |
| >20% | 1,978 | 0.4837 | 0.4968 | +1.3 pp |

**切片框架是分辨率补丁，不是通用检测增强**：粗图行数 <10 的目标靠细片救回；
rel≥2% 粗图已足够、细片不加分甚至减分。

### 6. 修正：medium −5.29 pp 是 57 个 SRRC 16QAM 的小子集效应

medium 0.3–2 MHz 组 = BLE/Zigbee + 窄带宽 SRRC 实例。组内分解
（`signal_characteristics.json`）：

| medium 组内 | n GT | A best | B | C |
| --- | ---: | ---: | ---: | ---: |
| BLE/Zigbee 6/7/8 | 2,847 | 0.952/0.954/0.944 | 0.952/0.955/0.943 | 0.959/0.947/0.941 |
| SRRC QPSK 10（0.3–2 MHz） | 137 | 0.843 | 0.392 | **0.859** |
| SRRC 16QAM 11（1.6–2 MHz） | 57 | **0.766** | 0.496 | 0.488 |

BLE 三臂完全相同；差距**全部**由 1.6–2 MHz SRRC 16QAM 的 57 个实例贡献
（细片 cls11 AP 0.091 拖累 C），同组 SRRC QPSK 上 C 反而最高。因此"A best
在 medium 更好 / 单图在中等带宽更强"应改述为**细片分支在 1.6–2 MHz SRRC
16QAM 上失效的小子集效应**，不是系统性带宽规律。留痕见第五节。

### 7. 超短信号是完全未解决的区间（时间维对偶问题）

| 区间 | n GT | A best | C | C 的 agnostic recall |
| --- | ---: | ---: | ---: | ---: |
| dur ≤ 5 ms | 514 | 0.262 | 0.242 | 0.881 |
| time_ratio ≤ 10% | 585 | 0.256 | 0.244 | 0.904 |
| dur > 60 ms | 925 | 0.426 | 0.513 | 0.977 |

14 类都有短实例（非 WiFi 混淆假象），**找得到（recall 0.88）但分不出（AP
~0.24）**，匹配框置信度仅 ~0.53，细片框架不解决（C 甚至略低于 A）。这是与
频率分辨率问题对偶的**时间分辨率**问题；若再做"切片"，时间维比再加频率视图
更值得。

### 8. SAHI 管道本身几乎零损耗，框质量缓降

`stage5_val/stage_recall.json`（C）：

| 阶段 | overall | narrow ≤0.25 | medium 0.3–2 |
| --- | ---: | ---: | ---: |
| 视图模型输出（pre-boundary） | 0.9645 | 0.9222 | 0.9921 |
| 边界过滤后 | 0.9606 | 0.9222 | 0.9829 |
| 全局 NMS 后 | 0.9583 | 0.9171 | 0.9809 |
| 每样本 top-100 后 | 0.9578 | 0.9162 | 0.9806 |

边界过滤 + NMS + 截断合计 **−0.68 pp**；漏检都在视图内模型上（narrow
pre 0.9222、rel≤0.5% 0.9025、dur≤5ms 0.8872）。唯一真实管道代价是 medium
BLE 的 −0.92 pp（骑切片边界被删）。IoU 扫描（`iou_sweep.json`）衰减平缓：
class-aware 0.6427→0.5945@0.75→0.5096@0.9，agnostic 0.9046→0.8543@0.75
→0.7303@0.9，无断崖。

### 9. 方法学警示

- **两条学习曲线不同相位**：A ep29 见顶后过拟合 −3.59 pp（train loss
  66.5→13）；C 到 ep49 仍在涨（0.6388→0.6427）。等预算对比略微低估 C 的
  天花板；`best_corrected.pt` 即 ep49 last EMA，无 checkpoint 不对称；旧口径
  早停会把 C 误停在 ep29/34，错过 0.6427。
- **噪声地板**：A 自身 best−last 波动 3.59 pp > B−A best 0.31 pp，"A best≈B"
  是 n=1 的无差异结论，不能读成单图训练追平联合训练。
- **成本叙事**：端到端只有 1.56×；单视图 STFT 构建 0.317 s 占 A 端到端的
  97%，模型只有 10 ms。"A 便宜 19×"只对 model-only 成立。
- **BLE/Zigbee 三臂全 0.94–0.95 已饱和**，不适合作未来对照的判别指标；判别
  应看 WiFi / SRRC / 窄带。

### 10. 由本轮引出、尚未做的动作

| 动作 | 依据 | 量级 |
| --- | --- | --- |
| 位置级去重（agnostic NMS / 每位置 top-1 类别） | 3,800 组跨类堆叠 | 上界 ~4 pp，纯工程 |
| 时间维视图 / 更长观测（dur≤5ms、time≤10%） | AP ~0.24、recall 0.88 | 7% GT，未探 |
| 评估口径以 WiFi / SRRC / 窄带为主 | BLE 饱和 | 只影响口径 |

（早前设想的"C3-fine-only"已被后来的 C3 三前端试点及其留出复核取代并收尾，
见第四节。）

## 四、与当前路线的关系（2026-11-20）

- **复数前端线已收尾**：[C3 三前端 pilot](opencode-c3-frontends.md) 与
  [1,159 留出复核](opencode-c3-holdout.md) 显示 zero3 无复数对照总体最好
  （mean 0.5156）、dense 的 3/4/5 残差不复现。本文 oracle 暴露的 **22 pp
  分类瓶颈因此仍未解决**；若重开复数线，需满足留出复核写明的前置条件
  （多 seed、预注册 checkpoint 规则、活容量对照）。
- **窄带找回已关闭**：[网格级复核](opencode-narrow-recovery.md) 显示粗图
  漏检的窄带目标对比度足够（35.8 dB，高于已检出组 32.6 dB），差在**薄**
  （2 行 vs 12 行）。这与本文 rel_bw ≤0.5% 定律、细片=窄带专家的结论互证；
  若继续，方向是检测器侧薄框分配 / 回归容差，且先做只读证据。
- 与已关闭路线不冲突、仍未做的：位置级去重、时间维短信号、评估口径调整
  （三.10）。

## 五、修正留痕（2026-11-20）

"medium 0.3–2 MHz 上 A best 比 C 高 5.29 pp"的原表述见于三处，本轮均已加注
指向本文三.6：

- [Baseline-0.md](Baseline-0.md) 待复核项；
- [opencode-group-a-fullband-baseline.md](opencode-group-a-fullband-baseline.md)
  带宽组表与解读段；
- [group_a_results.md](../../reports/group_a_results.md) Interpretation 第 2 条。

原始数值本身无误（0.8917 vs 0.8388，
[group_a_narrowband_check.json](../../reports/group_a_narrowband_check.json)）；
被修正的是**归因**：从"单图在中等带宽更强"改为"细片分支在 1.6–2 MHz SRRC
16QAM（57 GT）上失效的小子集效应"。

## 六、产物与口径

- 新入库：`reports/family_group_metrics_ab_c.json`（A last / A best / B / C
  × 8 家族 × 组过滤指标；B 取 `runs/regression/c_coarse`、C 取
  `runs/regression/c_all`、A 取 worktree `stage5_val{,_best}`；由
  `scripts/class_group_metrics.py` 的组过滤评测器计算，家族坍缩 AP = 组过滤
  口径 `ap50_agnostic`）。
- 主要引用（`runs/` 未入库，为服务器本地路径）：
  `runs/yolo11s_c1_directreg_2048x512/stage5_val/` 下 `oracle_class_ap.json`、
  `error_analysis.json`、`stage_recall.json`、`view_source_attribution.json`、
  `signal_groups.csv`、`signal_characteristics.json`、`iou_sweep.json`；
  以及 [view_source_attribution.md](../../reports/view_source_attribution.md)、
  [stage5_results.md](../../reports/stage5_results.md)、
  [group_a_results.md](../../reports/group_a_results.md)、
  [group_a_training.md](../../reports/group_a_training.md)、
  [group_a_latency.json](../../reports/group_a_latency.json)。
- test split 封存；本轮无新训练、无新推理，全部结论可由上述工件复算。
