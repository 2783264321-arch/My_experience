# OpenCode 结果：共享 fine-0-5 过滤 + 粗图裁块分类器首轮

依据 `docs/collaboration/README.md`（`67a453a` / `b323bc5`）执行。
详细报告：[reports/proposal_classifier_experiment.md](../../reports/proposal_classifier_experiment.md)。

## 1. 共享后处理已接通并复核

`predict_sample` / `run_validation` 增加 `drop_fine_classes` 开关；
`validate_run.py --drop-fine-classes 0-5` 使用同一套逻辑。完整 val：

| 指标 | baseline | filtered |
| --- | ---: | ---: |
| mAP50 | 0.642701 | **0.649725** |
| AP50 agnostic | 0.904594 | 0.905977 |
| R_agn | 0.957973 | 0.957833 |
| n_pred | 65,025 | 47,784 |

与 b323bc5 的 0.6497250 / 47,783 一致（+1 框为 chunk 数值噪声）。这是本轮对照。

## 2. 粗图裁块分类器首轮：失败

- 冻结 `best_corrected.pt`；fit 上 3,418 个样本、coarse proposals，
  7,978 个 128×256 裁块；其中 4,289（53.8%）是“框对但原类别错”的样本。
- 012 训练 4,278 / holdout 457；345 训练 2,939 / holdout 304。
- 小型 4-block CNN，20 epoch，按样本划分 holdout：
  - 0/1/2 holdout acc **0.4311**；confusion（行 GT/列 pred）
    `[[51,54,60],[41,49,50],[15,40,97]]`
  - 3/4/5 holdout acc **0.4507**；confusion
    `[[52,16,4],[37,62,24],[56,30,23]]`
  - 都接近随机（0.33），class 5 只有 23/109 正确。

固定 val，只改最终 0–5 框的类别，保留原框和原分数，再 class-wise NMS + 100：

| 指标 | filtered before | after classifier | 差值 |
| --- | ---: | ---: | ---: |
| 总 mAP50 | **0.649725** | 0.589391 | **-6.03 pp** |
| AP50 agnostic | 0.905977 | 0.905551 | -0.04 pp |
| R_agn | 0.957833 | 0.951689 | -0.61 pp |
| n_pred | 47,784 | 46,083 | -1,701 |
| TP / FP / FN | 6493 / 41291 / 669 | 6049 / 40034 / 1113 | -444 / -1257 / +444 |
| FP@0.5 错类/重复/背景 | 959 / 10 / 315 | 893 / 9 / 310 | -66 / -1 / -5 |

分组 AP：

| group | before | after | 差值 |
| --- | ---: | ---: | ---: |
| 0/1/2 | 0.4212 | 0.2522 | **-16.90 pp** |
| 3/4/5 | 0.4550 | 0.3425 | **-11.25 pp** |
| 10/11 | 0.6395 | 0.6395 | 0 |
| narrow 9/12/13 | 0.7804 | 0.7804 | 0 |

分类器改了 2,996 / 4,844 个 0–5 框；高分错类只从 959 降到 893，但 TP 掉了
444、FN 增了 444。**不能采纳这一轮分类器。** 额外耗时：裁块+分类器+重 NMS
约 **4.79 s / 4,844 框（~0.99 ms/框）**；完整 eval 70 s（含 coarse view）。

## 3. 判断

- 小型 coarse-crop 分类器**不可用**；holdout 接近随机，val 掉 6.03 pp。
- 这**不能证明 coarse 表示已到上限**：分类器小、欠拟合、且对所有 0–5 框
  无条件改类；换更强 backbone / 检测器特征 / 置信度 gate 之前，不能下结论。
- fine-0-5 过滤仍是独立的 +0.70 pp，保留为新的 baseline。

## 4. 是否值得做同框原生 STFT 裁块对照

**值得，但应作为加强分类器之后的受控对照，而不是现在直接用同一弱模型跑。**

- 当前 coarse 分类器接近随机，直接做 native 会被分类器容量/训练问题混淆。
- native crop 有 1.43 kHz 频率格（coarse 为 B/2048，约 2.4–39 kHz/行），
  0/1/2、3/4/5 的调制/边缘细节可能正是 coarse 聚合丢掉的；这是
  near-chance 的一个合理解释。
- 实测（B=80/T=150，10% context）：20 MHz coarse 616×286 vs native
  16,800×286；40 MHz coarse 1,230×170 vs native 33,601×170。裁块+resize
  约 0.15–0.27 ms，ResNet18 前向约 0.07–0.15 ms/proposal；4.8k 框只多
  约 1–5 s，成本不是问题。
- 但 native 裁块比 coarse 高 7–27 倍，直接 resize 到 128×256 会压缩
  100–260 倍，可能把要比较的细节抹掉。公平对照应使用 256×256/512×512、
  1D 频率 profile 或 coarse+native-profile 双分支，并保持**同一 proposal
  框、同一物理 context、同一结构/划分/融合规则**。

## 5. 下一步建议

1. 保留共享 fine-0-5 过滤（0.6497）作为新 baseline。
2. 加强 0–5 分类器：ResNet18/ConvNeXt-Tiny 或检测器 P5/ROI 特征；更多 fit
   proposals/hard negatives、轻量增强、更长训练。
3. 加**置信度 gate**：只在分类器 margin 高时改类，否则保留检测器原类/分数；
   阈值在 fit holdout 上定，不在 val 上调。当前“全部改类”是掉点主因之一。
4. 若加强后的 coarse 分类器仍混淆 0/1/2 或 3/4/5，再做同框 native STFT
   crop 对照；10/11 仍按带宽/来源单独处理；6/7/8/9/12/13 不做分类。
5. test 继续封存。

服务器产物：`runs/yolo11s_c1_directreg_2048x512/proposal_classifier_experiment.json`、
`proposal_classifiers/training_report.json`、`stage5_val_filtered_shared/`。
