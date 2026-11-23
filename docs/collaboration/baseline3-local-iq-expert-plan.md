# Baseline-3 局部 IQ 调制专家执行方案

2026-11-22。供项目负责人、Codex 和 OpenCode 共同审阅。后续决策和实验结果通过本目录交接。OpenCode 可结合 AutoDL 的实际数据和候选框独立复核。若建议改动，记录证据及能够区分方案的对照。

## 当前路线与交付节点

**先让冻结的 YOLO 走完局部 IQ 专家的完整闭环。** 沿用阶段 0 推理保存的 YOLO26l 最佳权重、完整 val 和粗细片后处理。专家把预测为 SRRC 的 10 MHz 框分成 QPSK（10）和 16QAM（11），仅更新类别。用同一批候选和同一套 AP 评估比较前后表现。框坐标、YOLO 原分数与已经完成的 NMS 固定。D-FINE-L 的接入排在 YOLO 路线走通以后。

1. **拿到第一张系统 AP 对照表。** 使用冻结 YOLO 的真实预测框，裁局部 IQ，运行现有专家及一个新表示。固定框、分数、NMS，报告 class10/11 AP、全 14 类 mAP、净改对、改坏和未匹配框被改。即使 AP 持平或下降，也完成了第一版，对照结果用来找下一步最该改的地方。
2. **在真实预测框上提高专家。** 用系统错例指导数据表示、模型和训练。优先把 fit 上与真值对齐的真实 YOLO 预测框纳入训练，让网络看到定位偏差和混叠。用相同的冻结 YOLO 完整 val 对照 GT 训练和添加 Pred 样本后的结果。逐窗诊断只在它能解释一个实际错误时使用。
3. **定型并复验系统收益。** 选好专家和改类规则后，锁定配置并在独立数据上复验完整 AP，同时记录两类收益和对原本正确结果的影响。确认方案有效后，将同一专家接口迁移到更强的检测器。正式 test 留给定型后评估。

### 第一版用什么表示

三个输入共用现成的框内搬频、带限、统一 20 MS/s、RMS 归一化和 1D ResNet。现有 `8×1024` 只负责整理成固定张量，本轮使用同一个取样预算。

- **幅度** `|z|`。把 QPSK 与 16QAM 可能不同的幅度分布直接交给网络学习。
- **Raw I/Q**。保留归一化后的两个复数分量，作为保留原始波形信息的参照。
- **幅度与相邻复相位变化** `[|z[n]|, Re(z[n]·conj(z[n−1])), Im(z[n]·conj(z[n−1]))]`。新表示同时给网络幅度及局部复数变化。相邻复乘消掉常量载波相位旋转，并保留幅度权重。分类判决由网络学习完成。

仅用预测类别 10/11、预测带宽 9.5–10.5 MHz、YOLO 分数至少 0.01 来决定是否取框。报告两条预定规则，即路由后用专家最高分的类别，以及专家置信度至少 0.70 才更新类别。两条规则沿用同一缓存和同一批候选。匹配真值只用于训练监督和评估计数。

### 第一版 AutoDL 命令

从该分支的仓库根目录执行。已有 20 轮幅度与 I/Q 权重可以直接复用。新表示单种子训练 20 轮，一次裁剪供三路评估。

~~~bash
pilot=/root/autodl-fs/runs_local/srrc-gt-pilot
run="$pilot/yolo26-label-only-v1"
reference="$pilot/seed1729_e20"

if [ ! -s "$reference/magnitude_last.pt" ] || [ ! -s "$reference/iq_last.pt" ]; then
  reference="$run/reference"
  python -m scripts.baseline3srrc_gt_pilot train \
    --cache "$pilot/fit_8win.npz" --out-dir "$reference" \
    --device cuda --arm both --epochs 20 --batch 8 --seed 1729
fi

python -m scripts.baseline3srrc_gt_pilot train \
  --cache "$pilot/fit_8win.npz" --out-dir "$run/new-representation" \
  --device cuda --arm amplitude_phase_change --epochs 20 --batch 8 --seed 1729

python -m scripts.baseline3srrc_yolo_expert \
  --expert magnitude="$reference/magnitude_last.pt" \
  --expert iq="$reference/iq_last.pt" \
  --expert amplitude_phase_change="$run/new-representation/amplitude_phase_change_last.pt" \
  --out "$run/val"
~~~

验证入口默认读取阶段 0 的 `val_full_best`。缓存搬迁后可用 `--pred-dir` 指向原始完整预测归档。每个结果有完整 mAP@0.5、mAP@0.5:0.95、逐类 AP50 和逐候选 CSV。真实 YOLO 参照是约 0.659601 mAP@0.5。第一张表连同采纳和损伤计数写回共享目录，以此结束第一阶段。

## 第一版完成与第二版执行

**第一版已交付**。完整冻结 YOLO26l val 上，654 个路由候选都进入了真实 IQ 裁剪。幅度与相邻相位变化在专家置信度 0.70 的规则下，mAP@0.5 从 **0.659601** 到 **0.663122**（+0.352 个百分点），class10/11 AP50 变化 **−0.42 / +5.35 个百分点**。57 个改对，14 个改坏，另有 54 个未匹配框被改。[第一张 AP 表](opencode-srrc-yolo-first-ap.md)

**第二步先加入真实预测框训练，沿用这个输入和现有 1D 骨干。** 14 个改坏全是把正确的 class10 改成 class11，其中 val 的 `train/1255` 第 1 号预测框，YOLO 分数 0.956 且类别正确，专家以 0.808 的置信度改错。另一个 val 例子是 `train/1175` 第 4 号框，匹配不到真值，专家仍以 0.938 的置信度改类。GT-only 训练没有直接教过网络如何处理这两种真实错误。离线只改匹配框约 +0.468 个百分点，只改未匹配框约 −0.092 个百分点。两种拆分只用于解释 AP，真实改类动作由预测可见的信息决定。[逐候选记录](../../reports/baseline3_srrc_yolo_first_ap/amplitude_phase_change_candidates.csv)

按同一次完整实验跑到底。

1. **准备 fit 预测框。** 复用同一固定 YOLO26l 权重和 filtered_0_5 后处理的 fit 预测缓存。若还没有，将下方导出命令跑一次，直接用于训练。按源文件分组沿用原 fit 内的训练和开发划分。
2. **训练一版会保留原类别的专家。** 复用 V1 的 8 窗幅度＋相邻相位变化表示和骨干作为起点。GT 裁剪与 fit 中可靠一对一 IoU 匹配的 10 MHz SRRC 预测框提供 class10/11 标签。fit 的未匹配、重复框和非目标框给出「保留 YOLO 原类别」的监督。这个标签对应改类动作，允许候选框里仍有信号。在原骨干上加一个保留类别的输出即可。训练做均衡取样或加权，避免拒判样本数主导损失。真值参与 fit 样本赋标与 val 的事后计数。
3. **直接出第二张完整 AP 表。** 用同一 1,543 样本、同一冻结 YOLO 预测与同一候选路由，比较纯 YOLO、V1 和 V2。把「保留」译为原 YOLO 类别，框、分数、NMS 和候选集合保持一致。延用两条已记录的判决口径，直接按最高分行动，以及 0.70 置信度的改类规则。特别看 class10 AP、14 个改坏这类事件是否减少、未匹配改动是否下降，同时核对 class11 收益是否还在。复用裁剪缓存，整条链路跑完再做下一次方向判断。

用下列入口导出 fit 预测。此模式调用同一冻结 YOLO 的推理和合并，直接写预测供训练使用。

~~~bash
pilot=/root/autodl-fs/runs_local/srrc-gt-pilot
yolo_run=/root/autodl-tmp/worktrees/formal-yolo26/runs/yolo26l_c1_2048x512

python -m scripts.validate_yolo26 --split fit \
  --checkpoint "$yolo_run/best.pt" --output "$pilot/yolo26-fit"
~~~

`fit_sample_ids.json` 与 `fit_predictions_filtered_0_5.npz` 按预测归档的索引一一对应。第一轮训练按服务器上实际可用的候选缓存和 I/O 跑同口径全 val，结果写回共享目录。

**V2 已执行（2026-11-23，OpenCode）**：fit 5,957 样本 / 96,051 views / 23.2 min；2,589 个候选，train/dev 2,081/508；V2 三分类 20 epoch / 113.8 s，dev balanced 0.459（keep 召回 0.380）。完整 val 上，括号内为实际改对/改坏/未匹配改动。V2 argmax **+0.386 pp**（96/33/87），V2 0.70 **+0.149 pp**（13/0/17），V1 0.70 **+0.352 pp**（57/14/54）。V1 与 V2 的 0.70 规则进入第三步独立复验。 [第二张 AP 表](opencode-srrc-yolo-v2.md)

**GT＋Pred 混训已执行（2026-11-23，OpenCode）**：3,539 训练实例（1,458 GT + 2,081 Pred），Pred-dev 508；80 轮 / 35,440 更新 / **116.9 s**，best epoch 15（Pred-dev balanced CE 1.0109）。0.70 规则完整 val：mAP50 **+0.149 pp**（0.661092），AP10 0.720204、AP11 0.617152，**13 改对 / 0 改坏 / 6 未匹配**。与 V2 0.70 同 mAP、改坏从 42 降到 0；V1 0.70 仍是最大 AP 主候选（+0.352，57/14/54）。V2 0.70 被混训支配。 [第三张 AP 表](opencode-srrc-yolo-mixed.md)

## 当前停点与正式 GT＋Pred 混训

现有 V1 和 V2 各跑了 20 轮，V2 从 V1 初始化后只用 Pred 样本训练。先在 fit 上完成一条 GT＋Pred 混训主线，并在固定 val 上出完整 AP 表。第三步外测放到这张表之后。

### 数据和预算

- 从已有 GT 缓存取 **1,458** 个 fit 训练实例，从 V2 缓存取 **2,081** 个 fit 训练候选，共 **3,539** 个训练实例。GT 和可靠匹配的 Pred 提供 class10/11，未匹配等候选提供保留原类。开发侧沿用 V2 的 **508 个真实预测框候选**，同源组只放一侧。
- 从 V1 末轮权重初始化，与 V2 一样扩成三输出。沿用幅度＋相邻相位变化、1D ResNet 和既有观察预算。训练用同一个种子 1729、batch 8、AdamW、按训练类别频数加权的交叉熵。
- **固定 80 轮**，约 **35,440 次更新**。峰值 LR **5e-4**，前 5 轮从 1e-4 热身到峰值，后续余弦退火到 **5e-5**。按已有 V1/V2 训练时长和更新数估，训练约 **13–25 分钟**，排期留约半小时，另计同一份 val 的 AP 评估。两份训练缓存和 val 预测框裁剪直接复用。
- 每轮自动记录训练损失、Pred-dev 三类召回和等权类别交叉熵。**选权重只看 fit 内 Pred-dev 的等权类别交叉熵**，保存最低值权重，同时留末轮作曲线复盘。80 轮到点结束，报告末段走势。唯一的人工停点是训练完成后的完整 val 对照表。

### AutoDL 执行

从仓库根目录训练一次。随后将自动选定的权重接到冻结 YOLO26l，在原 1,543 个 val 样本上按原 0.70 规则出表。

~~~bash
pilot=/root/autodl-fs/runs_local/srrc-gt-pilot
cached="$pilot/yolo26-label-only-v2"
mixed="$pilot/yolo26-label-only-mixed"
v1="$pilot/yolo26-label-only-v1/new-representation/amplitude_phase_change_last.pt"
yolo_val=/root/autodl-tmp/worktrees/formal-yolo26/runs/yolo26l_c1_2048x512/val_full_best

python -m scripts.baseline3srrc_v2 train \
  --cache "$cached/fit_candidates.npz" --gt-cache "$pilot/fit_8win.npz" \
  --v1 "$v1" --out "$mixed" --epochs 80 --batch 8 --lr 0.0005 --seed 1729

python -m scripts.baseline3srrc_v2 eval \
  --cache "$cached/val_candidates.npz" --v1 "$v1" \
  --checkpoint "$mixed/amplitude_phase_change_3class_best.pt" \
  --pred-dir "$yolo_val" --out "$mixed/val" --gated-only
~~~

主表四行为纯 YOLO、V1、V2 和新混训版。确认纯 YOLO 参照复现 0.659601 mAP50，一起报 class10/11 AP50、mAP50/mAP50:95、改对、改坏和未匹配改动。结合曲线及系统结果决定候选去留。延长训练和调整 LR 也属于这版方案，比较结果用于判断整体方案效果。把训练汇总和最终 AP 表写回共享目录。

## 第三步冻结复验（在混训评估后）

先根据混训的固定 val 结果选定候选。冻结纯 YOLO26l 与入选专家，在同一份获准使用的独立数据上一次跑完。继续用原框、原分数、原 NMS 和同一候选路由，交付全量 mAP50/mAP50:95、class10/11 AP50、实际改对、改坏与未匹配改动。正式 test 留给方案锁定后的外测。 [此前 V2 复核](opencode-srrc-yolo-v2.md#42-第三步独立复验)

## 前期记录

旧实验给了两个具体教训。

- SRRC 的 `B0_unsync` 在完整内部留出集上为 **61.54%**，在 213 个同步成功的实例上为 **79.34%**。原恢复门限只通过 36.9% 的 fit 实例。新模型的主比较覆盖完整留出集，再按标注框的重叠程度报告。 [SRRC 局部 IQ 结果](https://github.com/2783264321-arch/My_experience/blob/2c5e815/reports/baseline3_srrc_local_iq/srrc_local_iq_results.md)
- 旧 SRRC 纠错器把匹配框准确率从 52.0% 提到 61.7%，级联后全 14 类 mAP@0.5 从 0.649725 变为 0.647348。旧 Wi-Fi 的 GT+Pred 混合微调相对冻结专家约 +0.235 个百分点，配对区间包含零。新路线要测的是直接读波形后，分类覆盖、迁移和完整候选结果有没有改变。 [SRRC 级联结果](https://github.com/2783264321-arch/My_experience/blob/2c5e815/reports/baseline3_srrc_yolo_offline/srrc_yolo_offline_results.md#L27-L81) · [Wi-Fi 级联与不确定性](https://github.com/2783264321-arch/My_experience/blob/2c5e815/reports/baseline3_wifi_cascade/cascade_adaptation_comparison.md#L203-L223)

## 前期前端设计记录

### 观测频宽、目标频宽和采样率

分别记录原始观测宽度 $B_{\rm obs}$、待裁目标宽度 $B_{\rm sig}$、原始复采样率 $F_{\rm in}$ 和统一后的复采样率 $F_{\rm out}$。原始数据的频率坐标以观测中心为零频，项目采用 $F_{\rm in}=B_{\rm obs}$。每对 I/Q 算一个复采样点。

以 80 MHz 观测中的 10 MHz SRRC 为例，原始 IQ 是 80 MS/s，数字频带约为 −40 到 +40 MHz。目标搬到基带后约占 −5 到 +5 MHz。首轮统一到 **20 MS/s**，数字频带约为 −10 到 +10 MHz，给带限滤波的过渡区和预测频率偏差留出空间。设成目标总带宽的两倍是这个试点的工程选择。奈奎斯特公式里的最高频率与标注里的双边总带宽需用不同符号。

### 4 sps 放进后续对照

每符号采样点数为 $F_{\rm out}/R_s$，其中 $R_s$ 是符号率。若 10 MHz 接近波形的理论总带宽，且理想 SRRC 的滚降系数 $\alpha$ 约为 0.2，则从 $B=(1+\alpha)R_s$ 得到名义 $R_s\approx8.33$ MBd。此时 4 sps 对应约 33.33 MS/s，20 MS/s 对应约 2.4 sps。4 sps 提供规则的采样网格，符号采样相位需要单独估计。现有小规模波形证据支持把 8.33 MBd 当作试验先验。 [SRRC 前端先验](https://github.com/2783264321-arch/My_experience/blob/2c5e815/scripts/srrc_frontend.py#L19-L30) · [带宽与滚降关系](https://www.ee.iitm.ac.in/uday/notes/wsd/pulse.html)

直接 I/Q 基线只做按框裁剪、频率搬移、带限、重采样和归一化。后续若收益值得追，依次测名义符号率网格、SRRC 匹配滤波或符号定时对齐，每次改一个因素。比较 20 和 33.33 MS/s 时要对齐可见的**物理时长**和取窗策略。相同的 1,024 点在两者下分别是 51.2 和 30.7 µs。若有同步失败，记录覆盖率，在完整留出上评估可用的回退路径。

Wi-Fi 按相同的实验逻辑走。20 MHz 先做 0/1/2，40 MHz 另做 3/4/5，各组选能容纳该频带的局部采样率。先测直接 I/Q，再把 CP 定位、OFDM 符号对齐和子载波特征作为可拆分的增量，按各自的覆盖率和端到端增益判断。

## 实现起点

~~~text
原始 IQ + GT 或预测时频框
  → 按框读时间片，留出滤波所需的边缘上下文
  → 以观测中心为零频，将框中心搬到基带
  → 带限和抗混叠重采样到固定 20 MS/s
  → 在实例不同时间位置取 8 个窗口，每窗 2 × 1,024 点
  → 同一小型 1D ResNet 处理各窗
  → 平均窗口 logits，按实例计算分类损失
~~~

- 复用经修正的 `read_segment` 和 `demod_to_center`。换频使用框中心减观测中心。过去把观测下沿作为零频，导致所有提取偏移半个观测带宽。用已知频率样例和少量真实 IQ 验收中心、正负方向、带边、时间索引及 Pred 中心偏差。 [IQ 读取与坐标代码](../../scripts/baseline3_dsp.py#L35-L77) · [频率坐标勘误](opencode-baseline3-local-iq-round1.md)
- 滤波可从既有名义总宽度 10.2 MHz 起试，核对真实过渡带、邻频抑制和抽取后的混叠。若预测框偏移截断目标，比较一个固定的更宽保护带。训练和推理用同一套前端，预测输入从预测框几何得到，GT 只用于监督和诊断。
- 对窗口做 RMS 归一化，每个实例等权。训练取窗可以轮换位置，评估固定位置。先用等权汇总避免只挑最强窗口。长点窗、幅度协变量、复杂聚合或增强留到结果能说明要解决的具体问题时再做。
- fit 内按源 IQ 或已核实的重复源分组。同源 GT、Pred、所有窗口和增强版本同侧。10 MHz 档可复核既有 9.5–10.5 MHz 定义。原审计计数为 fit **1,824** 例、val **469** 例。按原定义记录交叠，报告无、弱、强三层，明确比率的分母。结构峰、定时失败和弱信号作为诊断与分层字段。窄带变体另做工程评估并按带宽报告。 [SRRC 数据审计](https://github.com/2783264321-arch/My_experience/blob/2c5e815/reports/baseline3_srrc_stage0/srrc_data_audit.md#L14-L85)

## 前期阶段划分

| 阶段 | 对照与工作 | 如何用结果决策 |
| --- | --- | --- |
| 0. 前端与系统空间 | 验收坐标、数据清单和取窗。复用最强检测器的冻结候选，对确能路由且可靠匹配的 10 MHz SRRC 框做离线真值改类，固定框和分数；对 Wi-Fi 20 MHz 做同类诊断。记录路由覆盖、错类、重复和未匹配。 | 真值改类只用于量化现有候选的可纠正空间。若 SRRC 端到端空间小，优先完成试点，正式系统预算放到空间更大的组。 |
| 1. GT 基线 | 比较原版 `B0_unsync`、同窗口预算下的 B0、幅度 CNN（复数模长）、同骨干双通道 I/Q CNN。先单种子验证小跑，测输入、标签、吞吐和缓存，再复验有希望的结果。报告实例级平衡准确率、各类召回及按 IQ 文件聚类的配对区间。 | 用同一完整留出集回答深度模型和复数输入各自贡献多少。看收益落在哪类交叠样本上，以及干净组有没有退步。 |
| 2. 预测框迁移 | 用同一份 GT 权重对同一目标的 GT 与 Pred 裁剪成对测试。需要适配时，从相同 GT 权重比较等更新步的 GT 继续训练与 GT+Pred 训练，预测框靠可靠几何匹配赋标签。若要主张预训练本身有效，加同总训练预算的混合数据从头训练。 | 区分预测框域差、额外训练和真实 Pred 样本带来的收益。记录 Pred 来源、匹配和模型可见过的源。若使用原 val 做适配，两系统都在相同剩余留出上复评。 |
| 3. 级联与评分 | 先固定候选、框和分数做仅改类对照，再比较检测器自身校准与加入局部 IQ 的组内融合。部署只依赖检测类别、预测几何和可获得的模型分数。阈值在开发集选择并锁定。 | 报告家族 AP、14 类 mAP@0.5 和 mAP@0.5:0.95、改对/改错、未匹配候选被改、耗时及覆盖率。根据全候选结果决定集成。 |

阶段 0、1 与窗数对照已完成，当前进度以文首的 YOLO 闭环路线为准。复用现有报告和脚本的位置，保留配置、提交和权重来源。后续分歧写进同一交接，不把尝试过的阈值或采样率写成既定规格。官方 test 继续留给方案冻结后的同口径评测。

**阶段 0 实测（2026-11-22，OpenCode）**：前端/数据核验通过；YOLO26l 冻结候选真值改类 oracle 显示 10 MHz SRRC 匹配准确率 48.9%、全部改对 +3.59 pp、路由后 +3.08 pp；8 窗中位覆盖 1.95%；GPU 被 D-FINE 占用，GT 小跑未执行。结果、调整建议和预算见 [阶段 0 记录](opencode-srrc-local-iq-phase0.md)。

## 外部对照与论文口径

ZoomSpec 已在同名 SpaceNet 数据上结合候选框取 IQ、频移与滤波、双域特征融合以及边界精修。论文报告官方测试的 mAP@0.5:0.95 为 **78.1**，D-FINE 加 LS-STFT 为 **74.3**。这是重要对照，也说明局部 IQ 和时频信息有联合验证的价值。其测试时长表写到 200 ms，本项目已审计的数据包观测到 150 ms。核对数据版本、测试文件、匹配协议和检测上限之后再做正式横向比较。 [ZoomSpec 方法与实验](https://arxiv.org/html/2604.13568v1#S4)

论文最需要回答的是同一检测器、同一候选下，局部 I/Q 能否减少调制错类，GT 到预测框迁移是否留下稳定收益，以及物理先验在同口径对照中贡献了多少。整体 AP 的提升用于衡量这部分研究对系统的价值。
