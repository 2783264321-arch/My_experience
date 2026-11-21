# SRRC Stage 0 · 实验协议（预注册草案，未训练）

2026-11-21。本文件只定义下一阶段的实验协议；**本轮没有启动任何
SRRC 训练**。test 继续封存。

## 1. 研究问题

在 SpaceNet advanced 的 class 10（10 MHz SRRC QPSK）与 class 11
（10 MHz SRRC 16QAM）上：

1. 直接 IQ 网络能否在**文件分组**的 val 上区分 10 vs 11？
2. 若能，增益是否在 clean / overlap 分层后仍存在？
3. 直接 IQ 是否优于手工 SRRC 统计特征（m4/m6/PSD/符号率）？
4. AMC-Net/AWN 的预训练表示在适配 10 MHz 输入后是否优于从零
   训练的简单 CNN？
5. 标签质量（窄框、近纯音、失配帧）是否是主要瓶颈？

## 2. 数据纳入/排除（预注册）

- 只使用 `split_new ∈ {fit, val}`；test 封存。
- 目标类：class 10/11；主口径带宽 `9.5–10.5 MHz`。
- 排除：带宽 <9.5 或 >10.5 MHz 的 class10/11 框（窄框长尾），
  除非做“窄框敏感性”对照。
- 按 `sample_id` 分组；同一文件的所有实例同侧。
- fit：1,572 文件 / 1,824 实例；val：404 文件 / 469 实例。
- 不重新随机切分 fit/val；如需 dev，fit 内做 5-fold 文件分组 CV。
- 所有样本清单见 `srrc_instances.csv`。

## 3. 输入管线（固定）

1. 用 GT 框中心做数字下变频：`demod_to_center(x, fs, fc, f_lo)`；
   GT 只用于离线训练/评估；推理时用 YOLO 框。
2. 带限到 10 MHz × 1.02（edge 5%），与 prior SRRC 审计一致。
3. 重采样到固定率 **20 Msps**（主） / 40 Msps（敏感性）。
4. 取框中心 1024 点（主） / 2048 点（敏感性）；不足则零填充/
   反射填充（预注册一种，推荐零填充并记录 mask）。
5. 每窗 RMS 归一化；保留原始幅度/SNR 代理作为协变量，不进入
   主模型输入。
6. 不做符号同步、不做 CP/OFDM 前端、不做手工 m4/m6 作为主输入。
7. 数据格式：`float32 [2, N]`（I/Q），不转 STFT/图像。

## 4. 模型与对照

| 编号 | 模型 | 输入 | 初始化 | 目的 |
| --- | --- | --- | --- | --- |
| B0 | 逻辑回归/手工特征（m4,m6,PSD 宽度,符号率线） | 统计量 | 从零 | 手工基线 |
| B1 | 简单 1D CNN / 小 ResNet | 2×1024 | 从零 | **首个主基线** |
| B2 | AMC-Net | 2×128 | 从零/改长度 | 架构对照 |
| B3 | AWN（2018.01a 2×1024） | 2×1024 | 预训练→微调 | 长窗预训练对照 |
| B4 | 同 B1 + label shuffle | 2×1024 | 从零 | 标签信息控制 |
| B5 | 同 B1 + 窄框/近纯音排除 | 2×1024 | 从零 | 标签质量敏感性 |

- B1 必须是简单、可解释、可复现的基线；不先上大模型。
- B2/B3 的预训练权重只作为初始化；输入长度/采样率适配后必须
  重训/微调，不能直接零样本套用。
- 所有模型使用相同 train/dev/test 文件划分、相同 epoch/early-stop
  预算；3 seeds（1729/1730/1731）。

## 5. 训练协议

- 任务：先二分类 10 vs 11；类别平衡（fit 941 vs 883）。
- Loss：CE + label smoothing 0.05（可选）；类别权重按文件频次，
  不按实例频次（同一文件多实例降权）。
- Optimizer：AdamW，lr 1e-3（从零）/ 1e-4（微调），wd 1e-4；
  cosine 或 plateau；early stop 按 dev macro-F1，patience 10。
- Batch：按文件分组采样，避免同文件多实例跨 batch 泄漏。
- 增强：仅使用不改变标签的 IQ 增强（随机相位、幅度缩放、
  小频偏、时间裁剪）；不使用会改变脉冲形状的增强。
- 不根据 val 调超参；val 只做最终留出。dev 从 fit 内文件分组
  CV 产生。
- 训练前先跑 label-shuffle 控制；若真实标签与 shuffle 差距
  <5 pp，停止扩大模型。

## 6. 评估与统计

- 主指标：val 文件级 accuracy、macro-F1、每类 P/R、混淆矩阵。
- 分层：clean（any-overlap=0）、overlap、strong≥25%、窄框排除。
- 统计：按 `sample_id` 文件聚类 bootstrap（≥2,000 次）报告
  accuracy/macro-F1 的 95% CI；不把实例当独立样本。
- 校准：报告 confidence 分布/ECE（可选）。
- 不报告单一 overall accuracy 作为唯一结论；必须同时给
  clean/overlap 分层与类别 P/R。
- 若使用 YOLO 框推理：固定 C0 权重，不重训 YOLO；报告 GT 框与
  Pred 框两种条件，并追踪未匹配预测。

## 7. 停止/继续判据

**停止并转数据/标签审计**，如果：

- 二分类在 clean 子集仍接近随机（macro-F1 <0.55 或 CI 覆盖
  0.5）；或
- label-shuffle 与真实标签差距 <5 pp；或
- 窄框排除后有效 clean 样本 <300 实例。

**继续到 Stage 2**，如果：

- B1 在 clean 子集显著优于 B0/随机；且
- 至少 2/3 seeds 方向一致；且
- overlap 分层后的下降可解释，而不是全由标签伪影驱动。

## 8. 已知风险

1. **标签有效性**：prior 审计仅 5/10 帧通过 SRRC 门限；全量普查
   95.6% 无预期符号率线、63% 近纯音。若标签本身不含 SRRC 调制，
   任何监督模型都只能学伪影。
2. **交叠**：80% any-overlap、46–53% strong；GT 框内可能混入
   其他信号，clean 子集小。
3. **采样率/脉冲形状**：SpaceNet 10 MHz SRRC 与 RML2016/2018
   的生成域不同；预训练权重不能零样本迁移。
4. **样本量**：1,824 fit / 469 val 10 MHz 实例，文件级独立单位
   更少；W40 式的小类问题不适用，但 overlap 会进一步缩小有效集。
5. **YOLO 定位**：10 MHz 子集 AP50 仅 0.3045/0.2377；用 Pred 框
   裁剪会引入定位误差，需与 GT 框条件分开报告。
6. **test 封存**：本阶段不接触 test；最终结论只基于 val/文件分组
   CV。

## 9. 交付物（Stage 1，尚未开始）

- `srrc_stage1_data_screen.md`：标签质量筛查结果
- `srrc_stage1_baseline_results.md`：B0/B1/B4/B5
- `srrc_stage1_pretrained_results.md`：B2/B3（若适配完成）
- 逐文件预测 JSON、bootstrap CI、混淆矩阵
- 不提交原始 IQ/权重；只提交脚本、清单、聚合结果
