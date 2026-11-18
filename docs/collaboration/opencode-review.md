# OpenCode 审查：下一轮路线

审查基准：`main` 提交 `80d92ab`（含 `docs/collaboration/README.md`）。
结合 AutoDL 最新代码与运行：baseline `best_corrected.pt`、修正后完整 val
（`2bc44cd` 的诊断）、8 epoch 纯续训、完整 val 的 300/1000 两入口核对。

## 判断

**同意 README 的主方向：先查清宽带类 `0/1/2`、`3/4/5`、`10/11` 的错类与
排序，再决定二阶段分类、重评分或视图合并；不建议先改模型或长训。**

补充证据（均来自当前 val 的固定权重预测）：

- **收敛已确认。** 8 epoch 纯续训（lr=2e-5，同 loss/views，修正后 val 每
  epoch）最好 **0.6423**（epoch 3），baseline **0.6427**；final 0.6405。
  没有收益，继续同配置训练没有意义。
- **recall 不是瓶颈。** 整体 class-agnostic recall `R_agn=0.958`；
  分阶段 pre-boundary 0.9645 -> post-boundary 0.9606 -> post-NMS 0.9583 ->
  final@100 0.9578。边界规则 0.39 pp、全局 NMS 0.23 pp、100 框上限
  0.05 pp，合计 <0.7 pp。
- **宽带的框基本都找到了。** `0/1/2`、`3/4/5`、`10/11` 的 `R_agn`
  约 0.966–0.970，但 class-aware AP 只有 0.40–0.71；AP_agn 也只有
  0.53–0.76。主要损失在 **错类、背景误检和 score 排序**，不是漏检。
- **oracle-class AP 给出分类上限。** 把 class-agnostic 匹配到的预测直接
  赋成 GT 类别后，整体 AP 从 0.6427 升到 **0.8639**（+22.1 pp）。增量集中在
  `0/1/2`（+0.33–0.49）、`3/4/5`（+0.39–0.45）、`10/11`（+0.19–0.31）；
  `6/7/8/9/12/13` 已接近 oracle（gap ≤0.012）。
- **窄带/短信号要分开看。** 窄带 ≤0.25 MHz 实际 0.780、oracle 0.786，
  问题是小框定位/排序，不是分类；短信号 ≤5 ms 实际 0.242、oracle 0.396，
  是时间表示/排序问题。

结论：下一轮最值得投入的是 **宽带类的“错类 + 精度/排序”**，而不是更多
epoch、boundary/NMS 调整或更大范围的检测覆盖。

## 对 README 两项审查发现的核对

1. **每视图 300 vs 1000 的入口不一致：属实，但对当前 val 指标无影响。**
   `predict_sample` 固定 `max_det=1000`（`src/signals/infer.py:112`）；
   完整验证默认 `max_det_view=300`（`src/signals/validate.py:113`、
   `scripts/validate_run.py`）。在完整 1543 源上分别用 300 和 1000 重跑，
   `mAP50=0.642701`、`AP_agn=0.904594`、`n_pred=65025`、boundary 202142
   完全一致；64 源抽查也 0/64 个样本的最终预测不同。说明当前 conf=0.001
   下没有视图触到 300 的上限。**建议把单样本入口对齐到 300 或显式暴露
   参数，但这不是需要单独排期的实验。**
2. **分类加权 fine-tune 的双重相乘：属实。** 安装的 Ultralytics 8.4.155
   `v8DetectionLoss` 读取 `model.class_weights` 并在 BCE 后
   `bce_loss *= self.class_weights`（`loss.py` 约 360-363、438-442 行）；
   项目里的 `_WeightedBCE` 又乘了一次，实际是 `w²`（例如 1.562 -> 2.441）。
   首场 baseline 没有启用类别加权（fine-tune 才显式设为 0.5；baseline config
   甚至没有该键），不受影响；但 fine-tune 的
   **负结果是被污染的**，不能据此否定类别加权。若重新研究加权：只保留
   一种乘法（建议用上游 `model.class_weights`），从相同权重做单变量对照。

## 最建议先做的一项实验

**冻结 `best_corrected.pt`，在完整 1543 源上做 view-source attribution：
coarse-only / fine-only / both（当前合并）三套预测，逐类、逐宽带组分解
TP/FP/FN 与 FP 类型。**

具体做法：

- 同一套 all-view 推理、同一 per-view NMS（建议 300）、同一 class-wise
  global NMS（IoU 0.5）、同一每样本 100 框上限。
- 三套预测：
  1. 只保留 coarse 视图；
  2. 只保留 fine 视图；
  3. 当前 coarse+fine。
- 对 `0/1/2`、`3/4/5`、`10/11`、窄带 ≤0.25 MHz、短 ≤5 ms 分别报告：
  - class-aware AP50、AP_agn、R_agn、AP75/AP50:95；
  - FP 分类：背景误检（与任何 GT 的 IoU 均 <0.5）、同位置跨类重复、
    近重复 GT 导致的未匹配、内部边界扣留；
  - 以及每类在 coarse-only / fine-only 下的 TP 来源和分数分布。

**决策规则：**

- 若 coarse-only 已贡献宽带类的大部分 TP，同时贡献大部分 FP：优先做
  **view-aware merge / coarse proposal 的二阶段分类或重评分**；
- 若 fine-only 的宽带 TP 很多、但在边界处理中被扣：再评估跨片拼接/
  合并（当前阶段 recall 只损失 0.4 pp，所以这条优先级低）；
- 若跨类同位置重复占 FP 主体：优先做 proposal 级分类器、成对混淆损失或
  类别校准，而不是继续调 NMS；
- 若背景 FP 占主体：优先做难负样本挖掘/分数校准。

**为什么先做它：** 不训练、固定权重、复用现有推理代码，一次完整 val 即可
把“视图来源、错类、重复、背景误检”分开；它直接决定 README 中
“二阶段分类/重评分”与“视图合并”谁是下一步。相比再训一个模型，成本极低，
且能避免把分类问题和 FP/排序问题混在一起。

## 不建议先做

- 再跑 50 epoch 或更长的同 loss 训练：8 epoch 续训已无收益。
- 调 boundary / global NMS / 100 框上限：recall 总损失 <0.7 pp。
- 直接把 fine-tune 的负结果当作“类别加权无效”：该实验实际用了 `w²`。
- 用 AP50:95 替换 AP50 作为选模指标：AP75=0.5945、AP50:95=0.5764，
  框较紧；建议 AP50 为主，AP75/AP50:95 作为按组报告的二级诊断。

## 参考

- [诊断报告](../../reports/stage5_diagnostics.md)
- 服务器运行目录（不在公开仓库）：
  `runs/yolo11s_c1_directreg_2048x512/stage5_val/signal_characteristics.json`
  `.../oracle_class_ap.json`、`.../stage_recall.json`、`.../iou_sweep.json`
- 8 epoch 续训：`runs/yolo11s_c1_directreg_2048x512_conv8/summary.json`（服务器）
- 300/1000 完整 val 复核：`runs/yolo11s_c1_directreg_2048x512/stage5_val_mdv1000/summary.json`（服务器）
