# SRRC Stage 0 · 文献与代码复现审计

2026-11-21。只做资料审计，不训练、不下载大数据、不使用 test。
目标：class 10（10 MHz SRRC QPSK）/ class 11（10 MHz SRRC 16QAM）
的局部 IQ 分类。所有外部信息以 2026-11-21 可访问的仓库/论文为准。

## 1. 候选基线总表

| 模型/论文 |  venue | 官方代码 | 预训练权重 | 输入 | 数据集/类别 | SNR | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **AMC-Net** | ICASSP 2023 | 是，MIT | 是，Google Drive/Baidu（不在 git） | 2×128 | RML2016.10a/b；11/10 类 | −20…+18 dB | ACM+MSM+FFM；本轮优先审阅 |
| **AWN** | TCCN 2023 | 是，MIT | 是，三数据集外部链接 | 2×128 / **2×1024** | RML2016.10a/b、RML2018.01a；11/10/24 类 | 2018：26 档（常见 −20…+30 dB） | 自适应小波；长窗 1024 权重 |
| VTCNN2 / O'Shea CNN | 2016 | 是（radioML/examples） | 否 | 2×128 | RML2016.10a；11 类 | −20…+18 dB | 最简基线；CC BY-NC |
| ResNet / CLDNN | 2017 | 分散/示例 | 否 | 2×128/1024 | RML2016/2018 | 同上 | 经典强基线 |
| MCLDNN | IEEE WCL 2020 | 是 | 是（10a） | 2×128 | RML2016.10a/b；11/10 类 | −20…+18 dB | TF1/Keras；三流 1D/2D/LSTM |
| MCNet | IEEE CL 2020 | MATLAB 代码 | 否 | 2×1024（2018） | RML2018.01a；24 类 | 2018 档 | 20 dB 24 类 93.59% |
| UN-GCPDS/AMC | DSP 2022 | 是 | 否 | 多种 | RML2016/2018/HisarMod | 各数据集 | 14 个模型的统一 Keras 实现 |
| TorchSig XCiT | 2026（工具） | 是，MIT | 宣称提供 | 2×N | 24 类窄带 | 可配置 | 支持 SRRC 生成；需核权重下载 |

**代码/权重的“是”指官方 README 提供入口；预训练权重多为外部网盘，
不在 git 内。** 本轮没有下载/执行外部权重，因此不能声称已复现其
数值；Stage 1 需要逐个锁定 commit、权重 SHA256 和推理脚本。

## 2. AMC-Net（优先审阅）

- 论文：J. Zhang, T. Wang, Z. Feng, S. Yang, “AMC-Net: An Effective
  Network for Automatic Modulation Classification,” ICASSP 2023,
  arXiv:2304.00445。
- 仓库：<https://github.com/zjwfufu/AMC-Net>，commit
  `38bedf5a41f779de355466b199e7630dfcb99cd0`，MIT。
- 代码完整性：`main.py`、`models/model.py`、`data_loader/`、
  `util/training.py`、`config/*.yml` 均在；`checkpoint/` 只有
  `.gitignore`，权重通过 README 的 Google Drive/Baidu 链接下载。
- 输入：`[N, 2, 128]`；config `sig_len: 128`，`num_classes: 11`
  （10a）或 10（10b）。
- 数据：RML2016.10a 220k / 10b 1.2M；11/10 类；SNR −20…+18 dB，
  2 dB 步进；GNU Radio 生成，RRC 成形（`sps=8`, excess BW 0.35），
  动态信道/频偏/噪声；生成器对每个向量做 `sum(abs)` 归一化。
- 预处理：原始 I/Q，无 STFT/手工特征；模型在 ACM 后做 L2 norm。
- 训练：Adam 1e-3；100 epochs；batch 64；每 (mod, SNR) 内随机
  6:2:2 划分（**不是文件分组**）；early stop 10 epochs。
- 报告：10a OA 62.51% / macro-F1 0.6483；10b OA 64.63% /
  macro-F1 0.6487（全 SNR 平均）。
- 采样率注意：官方生成脚本的 `dynamic_channel_model` 以
  `200e3` 实例化（RRC `sps=8`），数据集本身不带采样率字段；
  文献中常引用 1 MSps，但**应以生成脚本为准**，不要把它当成
  10 MHz 采样。对网络而言关键是 128 点归一化波形，而不是绝对 Hz。

**适配风险**：AMC-Net 的 ACM/FFM 以 `sig_len=128` 固定全连接/
注意力维度；把输入改成 1024 需要改结构并重训，不能直接加载
128 点预训练权重。10 MHz SRRC 需要先下变频/重采样到固定率，
再切 128/1024 点；128 点在 20 Msps 下只有 6.4 µs（约 53 个
8.333 Msym/s 符号），上下文偏短。

## 3. AWN（长窗预训练对照）

- 论文：J. Zhang et al., “Towards the Automatic Modulation
  Classification with Adaptive Wavelet Network,” IEEE TCCN 2023。
- 仓库：<https://github.com/zjwfufu/AWN>，commit
  `0b3d2b2d3d5d4761e73eb91e223844b4b8585b53`，MIT。
- 代码/权重：完整训练/评估/可视化；RML2016.10a/b + RML2018.01a
  三套预训练权重（Google Drive/Baidu）。
- 输入：2016 为 2×128；**2018.01a 为 2×1024，24 类，2.5M 样本，
  26 个 SNR 档**（常见 −20…+30 dB）。
- 预处理：原始 I/Q；自适应小波变换在模型内；数据加载/归一化需
  在复用前核对。
- 优势：1024 点权重比 AMC-Net 128 点更接近 10 MHz 长窗；包含
  QPSK/16QAM；官方代码/权重齐全。
- 风险：24 类合成数据、采样率/脉冲形状与 SpaceNet 不同；小波
  网络比简单 CNN 更难解释/裁剪；仍不能直接当 10 MHz 模型。

## 4. 其他成熟基线

- **VTCNN2 / O'Shea CNN**：2×128，RML2016.10a；代码在
  `radioML/examples`（commit `6c9ac602...`），无预训练权重，
  CC BY-NC。作为“从零训练”的最低成本对照非常合适。
- **MCLDNN**：2×128；官方代码
  <https://github.com/wzjialang/MCLDNN>，提供 10a 权重；TF1.14/
  Keras 2.2.4，环境老旧。
- **MCNet**：RML2018.01a 24 类，20 dB 93.59%；官方仓库只有
  MATLAB 代码，无权重；适合作为“轻量 CNN”参考，不适合直接复用。
- **UN-GCPDS/AMC**：14 个模型的统一 Keras 实现，覆盖 CNN1/2、
  MCNet、MCLDNN、CLDNN、PET-CGDNN 等；无预训练权重，TF1.14；
  适合做文献横向对照，不适合作为首个可执行基线。
- **TorchSig / torchsig-models**：支持 60+ 信号类型与 SRRC 生成，
  `XCiTClassifier` 24 类，宣称提供预训练模型；代码 MIT。适合
  生成与 SpaceNet SRRC 匹配的合成预训练/增强数据，但需在 Stage 1
  核实权重下载、输入长度和训练配置。

## 5. 七个问题的回答

### Q1：哪篇论文最适合 SRRC QPSK/16QAM 的局部 IQ 分类？

**分两层回答：**

- **作为架构/代码主参考：AMC-Net 最合适。** 官方 MIT 代码、
  预训练权重、输入/训练协议清楚，ACM+MSM+FFM 专门为 IQ 设计，
  且 QPSK/16QAM 在其类别内。缺点是 2×128、RML2016 生成域，
  与 10 MHz 长窗不匹配。
- **作为 10 MHz 长窗的预训练起点：AWN（RML2018.01a, 2×1024）
  更匹配。** 1024 点、24 类、官方权重，能提供更长的时域上下文。
- **作为本任务的首个主基线：不建议直接套任一预训练模型。**
  先用简化的 direct-IQ CNN/ResNet（2×1024，从头训练）建立
  可复现下限；AMC-Net/AWN 作为第二/第三对照，输入长度和
  采样率适配后重训或微调。

### Q2：是否提供完整官方代码及预训练权重？

| 模型 | 代码 | 预训练权重 | 权重位置/许可 |
| --- | --- | --- | --- |
| AMC-Net | 完整 | 有 | Google Drive/Baidu；MIT；不在 git |
| AWN | 完整 | 有（三数据集） | Google Drive/Baidu；MIT |
| MCLDNN | 完整 | 有（10a） | GitHub release；仓库许可需核 |
| MCNet | MATLAB | 无 | 仅代码 |
| VTCNN2 | Notebook/示例 | 无 | CC BY-NC |
| UN-GCPDS/AMC | 完整 | 无 | Keras/TF1.14 |
| TorchSig XCiT | 完整 | 宣称有 | 需 Stage 1 核实下载/许可 |

### Q3：论文输入长度、采样率、SNR、类别、预处理？

| 项 | AMC-Net | AWN (2018) | VTCNN2 | MCLDNN |
| --- | --- | --- | --- | --- |
| 输入 | 2×128 | 2×1024 | 2×128 | 2×128 |
| 采样率 | 官方生成器 200 kHz 量级；文献常写 1 MSps，需以脚本为准 | 1024 点；官方数据集卡未给单一率，常见 1 MSps 说法 | 200 kHz 量级 | 200 kHz 量级 |
| SNR | −20…+18 dB，2 dB | 26 档，常见 −20…+30 dB | −20…+18 dB | −20…+18 dB |
| 类别 | 11（10a）/10（10b） | 24 | 11 | 11/10 |
| 预处理 | 原始 I/Q；生成器 sum(abs) 归一；模型 ACM 后 L2 | 原始 I/Q；模型内小波；加载器归一化需核 | 原始 I/Q | 原始 I/Q |

### Q4：SpaceNet 局部 IQ 与论文输入有哪些必须适配的差异？

1. **带宽/采样率**：SpaceNet 捕获 B=20/30/40/50/80 MHz，
   复采样率 = B；SRRC 目标带宽 10 MHz。论文是 200 kHz/1 MSps
   量级的窄带单信号。必须先数字下变频到框中心，再重采样到固定
   率（建议 20 或 40 Msps），不能直接把原始捕获丢进网络。
2. **输入长度**：论文 128/1024 点；SpaceNet 目标持续 3–125 ms。
   需选固定窗（如 1024/2048 点）并中心裁剪；窗长与符号率、
   过采样率一起预注册。
3. **多信号/交叠**：SpaceNet 同一捕获含多个信号，约 80% 目标
   框有交叠、46–53% 强交叠；论文是单信号合成。需 clean/overlap
   分层，并考虑框外干扰。
4. **脉冲形状**：RML2016 用 RRC excess BW 0.35；SpaceNet SRRC
   的 prior 审计提示 α≈0.2、R≈8.333 MHz（10/1.2）。脉冲形状
   不匹配会改变星座/谱形。
5. **频率/定时偏移**：SpaceNet 有真实 LO/采样偏移和多信号；
   论文有合成 CFO/SRO/信道，但无宽频捕获上下文。
6. **类别空间**：论文 11/24 类；本任务先做 10 vs 11 二分类，
   不要把 14 类检测标签直接当 24 类 AMC。
7. **数据量/划分**：论文 220k/2.5M 随机划分；SpaceNet 10 MHz
   档 fit 1,824 / val 469，必须按 `sample_id` 文件分组，不能
   随机切样本。
8. **标签质量**：见数据审计；必须先筛查，不能假设 10/11 标签
   都对应可恢复的 SRRC 调制。

### Q5：fit/val 样本数、交叠、时长、YOLO AP？

见 [`srrc_data_audit.md`](srrc_data_audit.md) 的表。核心数字：

- fit 10 MHz：**1,824 实例 / 1,572 文件**（class10 941/868，
  class11 883/824）。
- val 10 MHz：**469 实例 / 404 文件**（class10 239/220，
  class11 230/209）。
- 时长中位 20–25 ms，q05 3–5 ms。
- any-overlap 78–82%，强交叠 45–53%。
- YOLO C0 AP50：all_labels 0.7079/0.5711；**10 MHz GT 子集
  0.3045/0.2377**。

### Q6：是否有足够样本监督训练与文件分组验证？

**数量够做试点，质量不足以直接宣称可靠分类。**

- 1,824 fit / 469 val 10 MHz 实例，1,572 / 404 文件，类别平衡；
  文件级 train/val 与 5-fold 文件分组 CV 可行。
- 但 prior SRRC 结构审计只 5/10 帧通过；全量普查 95.6% 无预期
  符号率线、63% 近纯音、56–57% 单音；标签有效性未知。
- 约 80% 交叠、46–53% 强交叠；clean 子集小。
- 建议先做 **class10 vs 11 二分类试点** + label-shuffle/窄框
  排除/clean-vs-overlap 对照；若二分类接近随机，先修标签/数据，
  不要加大网络。

### Q7：是否应该用直接 IQ 网络作为首个主基线，而不是重新设计符号同步与手工统计特征？

**是，首个主基线应是直接 IQ 网络。** 理由：

1. SRRC 是单载波脉冲成形信号，直接 IQ CNN 可同时学习脉冲形状、
   星座/幅度分布和残余频偏，不依赖脆弱的符号定时。
2. Prior SRRC 审计显示符号定时对 m4 很敏感（±0.25 符号可使
   QPSK m4 从 ~1.02 膨胀到 ~1.26），手工同步/统计特征容易把
   定时误差误判为调制差异。
3. SRRC 门限在真实帧上通过率仅 ~50%；手工同步会丢弃大量样本，
   在标签质量未知时进一步缩小有效集。
4. 直接 IQ 模型 + label-shuffle 控制是检验“标签是否含可恢复
   信息”的更干净探针；手工特征容易引入研究者自由度。
5. 但首个模型应**简单**：2×1024/2048 的 1D/2D CNN 或小 ResNet，
   从头训练；AMC-Net/AWN 作为适配后的第二/第三对照，不要一上来
   堆预训练大模型。

## 6. 推荐路线（Stage 1 预注册要点）

1. **数据**：只用 9.5–10.5 MHz class10/11；排除窄框长尾；按
   `sample_id` 分组；fit 内 5-fold 文件分组 CV，val 只做最终
   留出；test 封存。
2. **输入**：框中心下变频 → 重采样到 20 Msps（预注册；可做
   40 Msps 敏感性）→ 中心 1024/2048 点 → 每窗 RMS 归一化；
   同时保留 clean/overlap 标记。
3. **模型**：首个主基线 = 简单 direct-IQ CNN/ResNet（2×1024）；
   第二 = AMC-Net（128 点，重训/改长度）；第三 = AWN 2018
   （2×1024，微调）；全部同 split、同预算。
4. **任务**：先二分类 10 vs 11；再做 10/11 vs 噪声/窄框对照；
   不直接做 14 类。
5. **控制**：label shuffle、窄框排除、clean-only、overlap-only、
   随机噪声、全零/常数输入；报告每项的 accuracy/macro-F1。
6. **统计**：文件级 bootstrap CI；每类 P/R/混淆矩阵；clean/
   overlap 分层；不把实例当独立样本。
7. **停止条件**：若二分类在 clean 子集仍接近随机，或 label
   shuffle 与真实标签差距 <5 pp，停止扩大模型，转标签/数据审计。

## 7. 来源

- AMC-Net: <https://github.com/zjwfufu/AMC-Net>（commit
  `38bedf5a41f779de355466b199e7630dfcb99cd0`）；
  <https://arxiv.org/abs/2304.00445>
- AWN: <https://github.com/zjwfufu/AWN>（commit
  `0b3d2b2d3d5d4761e73eb91e223844b4b8585b53`）；
  <https://ieeexplore.ieee.org/document/10058977>
- MCLDNN: <https://github.com/wzjialang/MCLDNN>
- MCNet: <https://github.com/ThienHuynhThe/MCNet>
- UN-GCPDS/AMC: <https://github.com/UN-GCPDS/AMC>
- VTCNN2/RadioML examples: <https://github.com/radioML/examples>
  （commit `6c9ac6029ab1d0803442da7de8b7be04714bdebb`）
- RadioML generator: <https://github.com/radioML/dataset>
- TorchSig: <https://github.com/TorchDSP/torchsig>；
  <https://github.com/TorchDSP/torchsig-models>
- DeepSig datasets: <https://www.deepsig.ai/datasets/>
- SpaceNet class mapping: `reports/spaceNet_class_ids_and_wifi_labels.md`
- SRRC prior audit: `reports/local_iq_round1/phase_ap_srrc.md`、
  `phase_a_physics.md`
