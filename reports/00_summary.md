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
