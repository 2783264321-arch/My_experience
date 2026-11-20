# Baseline-3 · 局部原始 IQ 调制结构可分性（第一轮：物理核查 + 小规模配对对照）

2026-11-20，OpenCode。依据 [`97ed926`](https://github.com/2783264321-arch/My_experience/commit/97ed9266bcb2928b1c120094afeb0a77c5557f43)
的 A/C 归因、既有调制可分性报告与项目负责人的 Baseline-3 任务书建立。

## 研究问题

给定目标区域，能否从局部原始 IQ 恢复比功率图更有判别力的调制结构？
判别任务是组内小类：WiFi `0/1/2`（20 MHz 内 QPSK/16QAM/64QAM）、
`3/4/5`（40 MHz 内同三阶）、SRRC `10/11`（10 MHz 共有带宽档内
QPSK/16QAM）。不用带宽猜类别：两组的带宽在组内均无判别力，问题纯在
调制阶数。

## 与既有路线的关系

- 保留 Baseline-1 C1 粗图 + 细片定位系统；不重训 YOLO、不改线上推理、
  test 继续封存、不动原始 IQ/JSON。
- 旧 C3（复数系数当图像通道喂端到端）与窄带峰值/质心路线已关闭。本轮
  做的是**显式物理同步后的相位不变统计**（符号幅度矩、对齐 FFT 后的
  子载波幅度统计），判别器是轻量统计分类器，不落在这两条关闭路线的
  否定范围内。
- 证据链：oracle 分解显示分类单独损失 22.1 pp，全部集中在 WiFi
  （每类 33–49 pp）与 SRRC 10/11（19/31 pp）；[可分性报告]
  (opencode-modulation-separability-result.md) 的收尾结论是"局部细功率
  +2.3/+2.7 pp、real/imag 失败；若强分类器对照仍失败，下一步做局部
  IQ/星座/循环特征"——本轮接在该结论之后。

## 帧与划分（预注册）

- **WiFi 0–5**：复用可分性研究的同批 `(sample_id, GT box)` 帧
  （`reports/local_rep/proposal_local_win0100.npz` 清单：3,754 框 /
  2,503 样本，类别计数 705/716/732/530/517/554，IoU≥0.75 中位 0.984），
  overlap 标志沿用该批标注（clean / WiFi overlap / 跨类 / 强重叠≥25%）。
- **SRRC 10/11**：fit split 全部真值框，带宽档取 9.5–10.5 MHz
  （两类共有档；0.3–2 MHz 窄带档本轮不做）。overlap 标志按同一
  口径补算（any / 同类 / 跨类、面积比）。
- 所有表示同一几何：**GT 框 + 10% 物理边距**。IQ 从真值框直接取；
  同框功率基线（coarse 0.7 ms、local_power 0.1 ms）按同一几何重建，
  旧 proposal-box 数字只作文献参照，不作配对主对照。
- 划分：按 IQ 样本 20% holdout、seed 1729（沿用
  `split_by_sample` 同一实现）；参数与门限标定只允许在 train 侧，
  holdout 只评估一次。先低交叠后交叠。

## 预注册规则

1. **同步参数防泄漏**：候选网格只来自合成验证与 label-blind 质量指标。
   - WiFi 候选：SCS 78.125 kHz；FFT 256（20 MHz）/512（40 MHz）；
     CP ∈ {0.8, 1.6, 3.2, 6.4} µs。备选扫描 T_u ∈ {64,128,256,512}
     样本（信号临界采样率），选择依据是 CP 自相关峰的归一化强度，
     阈值在合成数据上按 SNR 10 dB 标定。标准 HE 参数只当候选，不当
     数据集真值。
   - SRRC 候选：α ∈ {0.15,0.20,0.25,0.30,0.35,0.40,0.50}，
     R = GT框带宽/(1+α)（带宽来自真值框，非类别）；匹配滤波后按
     平方信号在候选符号率的定时音选定时相位；选择依据是定时后星座
     紧致度，逐样本盲选。
   - 类别标签只进汇总评测，不进任何逐样本选择。
2. **同步失败弃权协议**：质量门限在合成数据上标定；真实帧按门限判
   synced/failed，失败率按类报告；增益报两个口径——synced-only 混淆、
   以及"失败回退功率基线"后的总账。
3. **判别器**：IQ 特征为每实例特征向量 + logistic/GBDT（3 seed）；
   功率基线沿用 SmallCNN 同配方（100 epoch、batch 64、class weights、
   同划分、seed 1729/1730/1731）。不再引入易过拟合的大容量 CNN。
4. **增益门限**：IQ 特征对同框最强功率基线的 holdout 配对增益
   ≥ +3 pp 且 3 seed 方向一致 → 过门；任一组同步失败率 > 30% → 该组
   判不可用。过门才提出现有预测框上的第二轮验证方案。
5. **卡点归因清单**（若真实数据信号不清，按此定位）：① WiFi 无 CP /
   非标准 numerology → 卡在对齐；② 中心频偏不在整数子载波格点 →
   ICI，卡在频率对齐；③ SRRC 符号率盲估在重叠帧失效 → 卡在匹配
   滤波；④ 每帧可用符号太少（短观测）；⑤ SNR 低于矩统计工作点。

## 阶段

- **Phase A · 物理核查（无标签）**：按类×带宽档计数表 → SRRC 通带/
  符号率盲估 → WiFi 合成验证整条代码路径 → 真实 CP/符号周期/对齐
  质量检查 → 同步成功率与参数分布。合成信号同时给出判别统计量随
  SNR 的敏感度包络，用于真实数据失败归因。
- **Phase B · 小规模配对对照**：同帧三档表示——coarse 功率、
  local_power（0.1 ms）、同步后符号/子载波统计；另加未同步 IQ 统计
  作为中间档，验证"同步本身是不是信息来源"。先低交叠，再交叠。
- **Phase C · 判定写回**：本文件追加结果与继续/停止判断；工件入
  `reports/local_iq_round1/`。

## 状态

- [x] Phase 0 交接记录（本文件）与 README 路线更新。
- [x] Phase A 物理核查（2026-11-20）：**标注调制在 IQ 中不存在**。
      WiFi 无 CP/12.8 µs 签名（HE 候选失败率 71.2%，剩余帧 r32 为窄带
      相干；子载波 m4 不落理论值），SRRC 无符号率线（三候选失败
      82.5–95.6%）；0–5/10–11 框内是窄带音样成分+底噪（fill10 比对照
      小 30–100×，20–37% 框基本全空，无突发、无掩埋台基、无镜像）；
      对照类 BLE/Zigbee/LoRa/AM/FM 用同一管线提取完全正常。
      [详细报告](../../reports/local_iq_round1/phase_a_physics.md)
- [x] Phase B 配对对照（2026-11-20）：IQ 物理特征三组全在机会水平
      （012 0.334 / 345 0.365 / 1011 0.526 vs 机会 0.333/0.333/0.500；
      单变量 AUC ≤0.11），同帧功率裁块 0.482/0.535/0.559。
      margin=0 事后诊断否定边距假说：功率基线读的是框内非调制细纹理
      （去边距 WiFi coarse 反升 +11 pp）。
      [配对报告](../../reports/local_iq_round1/comparison.md)
- [x] Phase C 判定写回（本节）。

## 结果与判定（Phase C，2026-11-20）

**判定：停止"从局部原始 IQ 恢复调制结构"路线。**预注册增益门三组
全部未过：IQ 特征对同帧最强功率基线的配对增益 012 **−13.6 pp**
（McNemar p=4.2e-05）、345 **−16.4 pp**（p=9.5e-06）、1011 **−5.0 pp**
（p=0.22，不显著）。不提出预测框上的第二轮验证——增益为负。

按预注册卡点清单归因：**卡点①命中，且从"参数不匹配"加深为
"结构缺失"**。WiFi 无 CP-OFDM 结构、SRRC 无脉冲成形单载波结构；
失败不是 CP/SCS/滚降参数取错，而是调制本身不在数据里，原计划的
合成门限标定因此失去意义。判定依据的证据链：

1. 管线自证：对照类（BLE/Zigbee/LoRa/AM/FM，各 25 帧）在同一解调/
   带限/普查下满带且带宽与标签一致；0–5/10–11 框 fill10 仅 0.002–0.034，
   无突发（占空比>med+6 dB≈0）、无掩埋宽带台基（框内下 40% 分位功率
   ≤捕获底噪）、无实值镜像（rms(Q)=rms(I)）。
2. 剩余类别信息不是调制：功率裁块在**紧框**（margin=0）下反而更高
   （WiFi coarse +9.6~+11.1 pp），说明其读的是框内时频细纹理；结合
   Phase A，该纹理只能是与类别相关的**生成伪迹**（逐类随机实现的
   tone 模式/底纹），SmallCNN 能提取、37 维 label-blind 汇总特征不能
   （逐特征 AUC ≤0.11）。
3. 方法学副产品：裁块几何配方（proposal 框±10% 发表口径 0.4085 →
   GT 框±10% 0.482 → GT 紧框 0.581）对 WiFi coarse 的影响（~17 pp）
   远大于可分性研究的表示间差异（+2.3 pp）。既往裁块分类结论应按
   几何配方口径比较。

对 [`97ed926`](https://github.com/2783264321-arch/My_experience/commit/97ed9266bcb2928b1c120094afeb0a77c5557f43)
归因的含义：oracle 的 22.1 pp 分类瓶颈（WiFi 每类 33–49 pp、SRRC
19/31 pp）获得**数据侧物理解释——瓶颈在生成层（类别标签与物理调制
脱钩），不在表示层**。任何表示工作（功率图、复数、符号域、循环谱）
都无法恢复数据中不存在的调制阶数；已关闭的 C3 复数前端与窄带
峰值/质心路线的失败与此一致。

探索性观察（事后子集，不作判定依据，留痕供未来参考）：SRRC 组
any-clean 的 64 个 holdout 帧上 IQ 特征 0.703 vs 功率 0.578，与
10/11 类音宽中位差（1.13/1.57 MHz）方向一致；但 82% 的 SRRC 帧有
某种重叠，重叠下 IQ 回机会水平（0.488）。若重开须预注册条件路线。

**继续/停止建议**：

- 停止本路线（按预注册门）。红线全部遵守：未重训 YOLO、未改线上
  推理、test 全程封存、原始 IQ/JSON 未修改。
- 若能获取 advanced 包的生成规则、或推动重新生成含真实调制的包
  （真实 OFDM/SRRC 波形），本路线可在新数据上重启；Phase A 的
  物理核查协议（`scripts/baseline3_frames.py` / `baseline3_census.py`
  及对照类校验）可直接复用，先于任何训练对照执行。
- 现有数据上，分类上限的可行动作只剩与表示无关的工程杠杆（位置级
  去重 ~4 pp 上界，A/C 汇总已记录）；"更好的调制分类特征"在调制域
  不可行，除非接受追逐生成伪迹（对真实数据无泛化意义，不建议）。

产物索引：`reports/local_iq_round1/` 下
[phase_a_physics.md](../../reports/local_iq_round1/phase_a_physics.md)、
[comparison.md](../../reports/local_iq_round1/comparison.md)、
[frame_counts.json](../../reports/local_iq_round1/frame_counts.json)、
[census_summary.json](../../reports/local_iq_round1/census_summary.json)、
[census_controls_summary.json](../../reports/local_iq_round1/census_controls_summary.json)、
[census_all_summary.json](../../reports/local_iq_round1/census_all_summary.json)、
[comparison.json](../../reports/local_iq_round1/comparison.json)、
[power_arms.json](../../reports/local_iq_round1/power_arms.json)、
[power_arms_m0.json](../../reports/local_iq_round1/power_arms_m0.json)；
大体积 npz 已 gitignore。运行脚本：`scripts/baseline3_{frames,census,
power_reps,power_arms,compare,dsp}.py`。

---

## 勘误与新状态（2026-11-21，频率坐标审计 + Phase A′）

**以上"结果与判定"一节中的停止判定已被推翻，保留原文仅作方法论
记录。**推翻链条：

1. **频率坐标错误（项目负责人质疑、独立审计证实）**：原始 IQ 的
   FFT 零频对应观测中心 `(f_lo+B/2)`，旧 `demod_to_center` 用
   `f_c−f_lo`，系统性偏移半个观测带宽。四层证据：原始 FFT 地标与
   JSON 标注仅在"零频=中心"下吻合（LoRa/BLE 窄带音逐一命中）；
   生产 STFT 标签映射代数上只与该约定自洽；正确/错误公式同帧对比
   （真实内容只在正确公式下出现）；生产粗图行号实测对齐。
2. **Phase A"标注调制不存在"撤销**：约 85% WiFi-20 / 67% WiFi-40 /
   100% SRRC 帧的普查测在错误频带；"控制类验证"是噪声混淆的循环
   论证（fill10 0.27–0.45、r32 0.43–0.83 均可由带限噪声复现）；
   旧 CP 检测器（幅度梳状）灵敏度比标准复数累加低约三个数量级。
   详见 `phase_a_physics.md` 顶部勘误。
3. **Phase B 的 IQ 特征臂无效**（同样建在错误频带上）；**功率基线臂
   继续有效**（走生产管线，几何正确）：coarse 0.482/0.503/0.552、
   local_power 0.459/0.535/0.559、margin=0 0.581，留作 Phase B′ 对照。
   详见 `comparison.md` 顶部勘误。
4. **Phase A′（正确坐标下的小规模复核）已完成，证据方向反转**：
   - 提取链路经 26 项合成单元测试验证（`.bin` 往返、5 种 B、
     负/偏心/边缘频率、能量保持 1.000、带外抑制 158 dB）。
   - **WiFi 17/30 帧检出强 CP-OFDM，全部 (T_u,CP)=(3.2,0.8) µs**
     （SCS 312.5 kHz，legacy 风格；HE 候选 12.8 µs 从未胜出），
     子载波 m4 与标签阶数 10/17 一致、QPSK vs 16/64 部分分离。
   - **SRRC 5/10 帧通过结构门限，6 帧带精确 8.333 MHz=10/1.2
     符号率线（α=0.2）**；class-10 恢复符号 m4≈1.01–1.14（QPSK 样），
     class-11 1.39–1.79（方向与 16QAM 一致、值偏高）。
   - 负对照（合成噪声、真实诱饵带）全部 0 过，判定门限全部由
     合成标定预注册。
   - 主报告：[phase_ap_report.md](../../reports/local_iq_round1/phase_ap_report.md)；
     三轨明细：[extract](../../reports/local_iq_round1/phase_ap_extract.md) ·
     [wifi](../../reports/local_iq_round1/phase_ap_wifi.md) ·
     [srrc](../../reports/local_iq_round1/phase_ap_srrc.md)。
5. **Phase B′（配对分类对照）尚未开始**，是否进入由独立审阅决定。
   若继续，预注册顺序：全量 5,578 帧结构普查（脚本已就绪）→ 以
   真实 numerology 为同步锚的 WiFi/SRRC 特征 + 轻分类器，对照沿用
   上述有效功率基线；重叠帧与失配帧单独分层；SRRC 专家需带定时
   弃权门（±0.25 符号定时错足以抹平阶数差异）。

红线不变：test 封存、原始 IQ/JSON 未修改、未训练新模型、
线上推理未动。
