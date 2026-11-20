# SpaceNet 第一场 baseline 总结（2026-11-18）

项目根：`/root/autodl-tmp/workspace/baseline`
运行：`yolo11s_c1_directreg_2048x512`（50 epoch，已完成）

## 结论（先看这里）

- 模型：YOLO11s 主体、C=1、14 类、`reg_max=1`、`end2end=False`
  （DirectReg，非原版 YOLO11s）。
- 输入：`2048x512`；IQ -> 0.7 ms Hann STFT（512 列）-> log power
  -> 1 张全频 coarse + 多张 2048x512 fine 切片。
- 数据：冻结 split `fit 5957 / val 1543 / test 2500`；test 未打开。
- **修正后完整 val（1543 源、24,975 views、全部 fine+coarse）**：
  - strict class-aware **mAP@0.5 = 0.6427**
  - class-agnostic AP@0.5 = 0.9046
  - localization recall@0.5 = 0.9580
  - matched classification accuracy = 0.8433
  - `best_corrected.pt` = last EMA（epoch 49）；`best.pt`（epoch 29）为 0.6388
- 训练耗时 2.694 h（RTX 4090，bf16，物理 batch 8 源=16 views，
  AdamW lr0=1e-3，cosine，warmup 3 epoch）。

## 关键修正（必须知道）

训练期 val 的 `_map_view_dets` 没有先把 fine 框裁到视图边界，导致部分
DirectReg 框映射到原始频带之外，增加了 FP、压低了 class-aware AP
（训练期最好 0.5429）。Stage 5 发现后按计划加上 `[0,512]x[0,2048]`
裁剪并重跑完整 val，得到 0.6427，并重新选择 checkpoint。
旧结果保留在 `stage5_val_unclipped/`；训练期 `val_history.json` 只作为趋势。
详细说明：`reports/stage5_results.md`。

## 结果要点

强类（AP50）：6=0.959、7=0.947、9=0.948、8=0.941、12=0.739、10=0.708。
弱类：1=0.329、0=0.354、5=0.405、4=0.427、3=0.492；主要剩余问题是
0/1 和 3/4/5 的类别混淆。按带宽：5MHz=0.824、20MHz=0.738、30MHz=0.707、
40MHz=0.635、50MHz=0.630、80MHz=0.652。

错误分析：TP 6,490 / FP 58,535 / FN 672；同一位置多类预测组 3,800
（class-wise NMS 的固定规则）；202,142 个 fine 边界候选被单独扣留。
部分 FN 是近重复 GT（IoU 0.996）在 one-to-one matching 下只能匹配一个，
这是“不静默去重/不 same-box ignore”的直接结果。详见
`reports/stage5_results.md` 和 `stage5_val/error_analysis.json`。

## 诊断补充（2026-11-18）

- 三级损失分解：R_agn **0.958** → AP_agn **0.905** → AP_cls **0.643**。
  边界过滤、全局 class-wise NMS、每样本 100 框上限合计只损失约 **0.7 pp**
  recall；不是瓶颈。
- Oracle-class AP（匹配框分类完美）为 **0.864**，分类 gap **22.1 pp**；
  主要集中在 0/1/2、3/4/5、10/11。6/7/8/9/12/13 已接近 oracle：
  窄带 12/13 的问题是小框定位/排序，不是分类。
- AP75 = 0.5945，AP50:95 = **0.5764**（class-agnostic 0.8543 / 0.8203）。
  框较紧；建议 **AP50 继续作为主指标/选模指标**，AP75/AP50:95 作为二级诊断，
  按类别/带宽组报告，不用 AP50:95 替换 AP50。
- 8 epoch 低 LR 纯续训：best **0.6423**（epoch 3）vs baseline **0.6427**，
  final 0.6405；差异在噪声内，判定**已收敛**，不再靠加 epoch 提升。
- 详细表：`reports/stage5_diagnostics.md`、`stage5_val/signal_characteristics.json`、
  `oracle_class_ap.json`、`stage_recall.json`、`iou_sweep.json`。

## 产物

- `runs/yolo11s_c1_directreg_2048x512/`：config、train_log.csv、
  val_history.json、best.pt、last.pt、best_corrected.pt、summary.json、
  training_curve.png。
- `runs/yolo11s_c1_directreg_2048x512/stage5_val/`：修正后完整预测
  （val_predictions.npz）、per_bt.csv、error_analysis.json、summary.json。
- `runs/yolo11s_c1_directreg_2048x512/stage5_examples/`：10 张
  coarse 视图 GT/预测叠加图。
- `stage5_inference.json`：单样本推理耗时。
- 报告：`reports/stage0_audit.json`、`stage1_smoke.md`、
  `stage2_resources.md`、`stage3_overfit.md`、`stage4_training.md`、
  `stage5_results.md`。

## 耗时/资源

- 训练 epoch 0：393 s（启动 + 最后一次 10-view batch 的 cuDNN autotune）；
  之后 155-164 s/epoch。
- 完整 val：约 3-6 min（24,975 views；本次修正复算 4.3 min）；每 5 epoch 一次。
- 训练显存 reserved 10.1 GiB；val 1.32 GiB。
- 单样本推理：5MHz/20ms 0.10 s；40MHz/80ms 0.79 s；
  80MHz/150ms 1.66 s（CPU STFT 约 1.2 s）。

## 已完成 / 可选后续

1. **GitHub push 已完成**：`main` 已推送到
   `https://github.com/2783264321-arch/My_experience`，使用仓库级
   write deploy key；本地与远程 `main` 同步。
2. **test 仍封存**，未做任何 test 评估或调参。
3. 分类加权 fine-tune（`runs/yolo11s_c1_directreg_2048x512_clsft/`）
   已完成，但**未超过 baseline**：修正后 mAP 0.6366 vs 0.6427，未采纳；
   仍以 `best_corrected.pt` 为交付。详见 `reports/stage5_finetune.md`。
4. 若继续提升，优先处理 0/1 与 3/4/5 的类别混淆（类别加权/更大模型/
   轻度增强），而不是再压缩频率轴或减少切片覆盖。

## Group A 全频单图基线（补充）

- 独立训练、每源只跑 1 张全频粗图、100 epoch；与 C 的 50 epoch 在
  595,700 张图像 / 37,300 vs 37,250 步上对齐。
- **A best（ep29）= 0.5763**，A last（ep99）= 0.5404；
  B 仅粗图 = 0.5794，C 全视图 = 0.6427。
- B−A best **+0.31 pp**，C−A best **+6.64 pp**，C−B +6.33 pp。
- A last 明显过拟合；未来实验启用 4 次验证事件早停（约 14,920 步），
  主 checkpoint 用 best EMA。
- 推理 views/sample：A 1 vs C 19.10；model-only 8.64×，端到端 p50 1.56×。
- 详见 `reports/group_a_results.md`、
  `docs/collaboration/opencode-group-a-fullband-baseline.md`。

## C3 三套粗图前端 pilot（补充）

- 同一原版 C1 功率图，1536 fit / 384 val、20k steps、best EMA；C1/overlap/
  sparse 各 2 seed，dense 1 seed。
- C1 mean **0.4999**；overlap mean 0.5078（+0.79 pp）；sparse mean 0.4998
  （−0.01 pp）；dense 0.5072（+0.73 pp）。
- **0/1/2 没有稳定收益**：overlap 0.3019→0.2473，sparse 0.2819→0.2336；
  seed1729 的增益未复现。
- **3/4/5 是唯一复现的收益**：overlap 两 seed 0.4012/0.3940（mean +4.01 pp）；
  dense 单 seed 0.4646（+10.70 pp，错类 79→69）；sparse 不稳。
- dense 是 3/4/5 专用，不是全局替换；sparse 总体 +2.51 pp 是 seed 波动。
- 仍有 1 通道 C1 vs 3 通道 C3 的容量混杂，需 C3-zero-complex 才能因果归因。
- 详见 `docs/collaboration/opencode-c3-frontends.md`、`reports/c3_pilot_results.md`。

### C3 复核：dense seed1730 + zero3 对照

- zero3 = 3 通道 `power+0+0`，同架构/head，无复数信息；不是活容量对照。
- best mean：zero3 **0.5156** > dense 0.5123 > overlap 0.5078 >
  sparse 0.4998。没有复数前端在总体上超过无信息对照。
- 0/1/2：zero3 0.2834 最好；dense 0.2619，复数信息未帮助 20 MHz。
- 3/4/5：dense **0.4543** vs zero3 0.4198，+3.45 pp（两 seed 同向）；
  common-box 3/4/5 分类 dense +2.95~+5.95 pp（4/4 同向），但 McNemar
  p 0.22–0.77，定位召回不升。属于需复核的调制判别信号，不足以全量扩大。
- 建议：3–5 seed、768+ val、power+power+power/随机活通道的活容量对照；
  只有 3/4/5 残差稳定后再扩全量。0/1/2/overall 不扩。
- 估计量：aggregate-first 同时抬高 GT/噪声，无弱信号分离优势，保留现版。
- 详见 `reports/c3_replication_results.md`、`docs/collaboration/opencode-c3-frontends.md`。

### C3 1,159 留出复核

- 冻结 dense/zero3 best+last，不重选、不调阈值；1,159 样本、290 个 3/4/5 GT。
- best 均值：dense mAP 0.4983 vs zero3 0.4990；3/4/5 AP 0.3874 vs 0.3489；
  common 3/4/5 分类 0.4804 vs 0.4454。
- last 均值：dense mAP 0.4583 vs zero3 0.4520；3/4/5 AP 0.3163 vs 0.3465；
  common 分类 0.4697 vs 0.5032。
- 逐 seed/ckpt 只有 s1730 best 显著偏向 dense（+10.45 pp，p=0.007）；
  s1729 best/last、s1730 last 都偏向 zero3。384 子集的 dense 4/4 优势未复现。
- 定位召回 dense 不高于 zero3；唯一显著点是分类，不是找到更多框；best/last
  排序翻转说明 checkpoint 选择不稳。
- **不扩全量训练。** 若继续，先做预注册 checkpoint 规则的多 seed 复核 +
  power+power+power/随机活通道的活容量对照。
- 详见 `reports/c3_holdout_replication.md`、
  `docs/collaboration/opencode-c3-holdout.md`。

### C3 holdout 分层诊断（B/T）

- 复用 1,159 留出预测，不重训/不重选/不调阈值；3/4/5 GT 只在 B=40/50/80。
- B 汇总四对比符号：40 MHz `- - + -`，50 MHz `- + + -`，80 MHz `- - + -`；
  没有 B 条件四对比全为 dense 正。
- T 仅 100 ms 一致，方向为 **zero3 更好**（`- - - -`）；其余 T 为 mixed。
- B×T 无 common≥10 且四对比全 dense 正；`40×80` 为 `- - + +`，不选为目标；
  `80×80 + + + 0`、`50×60 + + + -` 也不满足严格一致。
- **建议收尾这版稠密短窗前端，不扩全量。**
- 详见 `reports/c3_holdout_strata.md`、`docs/collaboration/opencode-c3-holdout.md`。

### 窄带找回假设：带内 peak/centroid 只读诊断

- 复用 B/C 冻结预测：窄带 9/12/13 中 both 1,437、**粗图漏掉/细片找到
  542**、neither 177；对照背景带 2,289、B 窄带误检 2,461。
- **绝对 peak 可分性极强**：c_only vs 背景 AUC **0.995**，vs B 误检
  **0.965**；分 B 全部 ≥0.964。
- **PAPR 不可靠**：总体 0.704，80 MHz 仅 0.502；class 12 甚至反向
  （c_only 10.6 vs both 40.2）。峰均比确有频点数/噪声混杂。
- **centroid 真实但较弱**：c_only |centroid| 中位 **0.057** vs both
  0.010 / 背景 0.012；AUC 0.62–0.84，80 MHz 最低。
- 建议只做一轮窄范围配对训练：同一 0.7 ms STFT 派生
  `band_peak` + `band_centroid` 两个低成本 2048×512 通道，功率通道原样；
  不加 PAPR、不加复数/短窗；预注册只用 last checkpoint。
- 详见 `reports/coarse_narrow_recovery.md`、
  `docs/collaboration/opencode-narrow-recovery.md`。
