# OpenCode 结果：view-source attribution（冻结 best_corrected.pt）

依据 `docs/collaboration/README.md` 提交 `40e94aa` 的首个实验口径执行。
固定权重与完整 val（1543 源 / 24,975 views），一次 all-view 推理，全局合并前
按 coarse / fine / both 拆分；每组使用相同 class-wise NMS（IoU 0.5）和每样本
100 框上限。详细报告：[reports/view_source_attribution.md](../../reports/view_source_attribution.md)。

## 复现

| 指标 | baseline stage5_val | both 组 | 差值 |
| --- | ---: | ---: | ---: |
| mAP50 | 0.6427009 | 0.6427025 | +1.7e-6 |
| n_pred | 65,025 | 65,027 | +2 |
| boundary dropped | 202,142 | 202,131 | -11 |

逐类 AP 到 3 位小数一致。差异来自 chunk 组合的数值非确定性（原验证用跨样本
的全局 8-view buffer，本次按样本 8-view chunk）；不影响来源结论。
`predict_sample` 已顺手对齐到 `max_det_view=300`。

## 总体来源

| source | mAP50 | AP50 agnostic | R_agn | n_pred |
| --- | ---: | ---: | ---: | ---: |
| coarse only | 0.5794 | 0.8178 | 0.8773 | 25,409 |
| fine only | 0.3165 | 0.5013 | 0.5564 | 45,629 |
| both | **0.6427** | 0.9046 | 0.9578 | 65,027 |

## 宽带组（AP=组内类别 AP 均值；FP@0.5(m/d/b)=高分误检的错类/重复/背景）

| source | group | AP | R_agn | TP/FP/FN | FP@0.5(m/d/b) | GT(c/w/m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| coarse | 0/1/2 | 0.4212 | 0.9699 | 806/2783/191 | 464/2/22 | 437/530/30 |
| fine | 0/1/2 | 0.0000 | 0.0000 | 0/13503/997 | 0/0/0 | 0/0/997 |
| both | 0/1/2 | 0.4020 | 0.9699 | 805/15970/192 | 464/2/22 | 437/530/30 |
| coarse | 3/4/5 | 0.4550 | 0.9661 | 337/926/105 | 183/1/18 | 224/203/15 |
| fine | 3/4/5 | 0.0000 | 0.0000 | 0/4993/442 | 0/0/0 | 0/0/442 |
| both | 3/4/5 | 0.4415 | 0.9661 | 337/5787/105 | 183/1/18 | 224/203/15 |
| coarse | 10/11 | 0.5908 | 0.9679 | 630/1249/86 | 273/0/15 | 452/241/23 |
| fine | 10/11 | 0.2157 | 0.2430 | 165/5855/551 | 0/0/42 | 161/13/542 |
| both | 10/11 | 0.6395 | 0.9721 | 634/6934/82 | 273/0/51 | 456/240/20 |
| fine | 9/12/13 | 0.7821 | 0.9032 | 1942/10329/218 | 4/1/73 | 1925/26/209 |
| both | 9/12/13 | 0.7805 | 0.9162 | 1965/21039/195 | 10/1/94 | 1946/33/181 |

## 判断

1. **宽带 0/1/2、3/4/5 是 coarse 视图的类别。** coarse-only R_agn
   0.966–0.970；fine-only R_agn/AP=0（2.93 MHz 细片装不下完整 20/40 MHz 框）。
   both 的 GT 覆盖与 coarse 相同，但 fine 额外带来 13,210/4,883 个背景 FP，
   所以 both 的 AP 反而略低于 coarse（0.402 vs 0.421；0.442 vs 0.455）。
2. **高分误检来自 coarse 的错类，不是 fine 背景。** 0/1/2 的 FP@0.5 为
   coarse 464 错类 / 2 重复 / 22 背景，fine 0/0/0；3/4/5 为 coarse
   183/1/18，fine 0/0/0。fine 的 FP 绝大多数是低分背景。
3. **框找对了、类别经常错。** 宽带 GT 被错类覆盖：0/1/2 530/997、
   3/4/5 203/442、10/11 240/716；这些框的 IoU 多在 0.98–0.999。
   例：train/3254 GT1→pred2 0.981@0.991；train/520 GT1→pred2 0.977@0.999；
   train/3126 GT4→pred5 0.984@0.992；train/2777 GT11→pred10 0.974@0.983。
4. **10/11 是混合带宽组。** fine-only 能找到窄的 10/11 实例（R_agn 0.243），
   但带来 5,396 背景 FP 和 42 个高分背景 FP；both AP 0.640 > coarse 0.591，
   主要来自 class 10。不能对 10/11 一刀切丢掉 fine。
5. **窄带 9/12/13 是 fine 的领域。** fine-only R_agn 0.903、AP 0.782；
   coarse-only 0.667/0.484；both 约等于 fine。coarse 只增加约 23 个 TP 和
   约 1.07 万 FP。
6. **边界/近重复不是宽带主因。** 边界候选 202,131（宽带组
   81,421/57,866/30,809）；近重复 GT 的 FN 仅 27/11/3（wide groups），
   远小于错类覆盖 139/86/59 和漏检 26/8/20。

## 首选改动

**先做 coarse 候选框分类复核（lightweight proposal classifier）。**

- 冻结检测器；取 coarse 视图的候选框（或与 GT 匹配的 coarse proposals），
  在 fit 上训练轻量 14 类/家族分类器（hard negatives），在同一 val 上评估。
- 目标：0/1、1/2、3/4/5、10/11 的高分错类；oracle-class AP 已显示
  0.83–0.87 的上限，实际只有 0.40–0.47。
- 不建议先做检测器重训或全局类别加权（后者还有已核实的 `w²` 问题）。

**次级、低成本：view-aware merge。**

- 0/1/2、3/4/5：直接丢弃 fine 预测（fine-only TP=0；仅减少背景 FP，
  预计整体 +0.7 pp 量级）。
- 10/11：按带宽/来源门控，10 MHz 用 coarse，窄实例用 fine；不要一刀切。
- 9/12/13：优先 fine；both 已接近 fine-only。

详细逐类 AP、全部 FP/FN 例子和边界计数在服务器
`runs/yolo11s_c1_directreg_2048x512/stage5_val/view_source_attribution.json`；
可读表在 [reports/view_source_attribution.md](../../reports/view_source_attribution.md)。
