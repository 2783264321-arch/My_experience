# SRRC V2：真实预测框训练与第二张冻结 YOLO AP 表

2026-11-23。分支 `work/srrc-gt-pilot-prep`，基线提交 `09a1bec`。
V2 沿用 V1 的幅度＋相邻相位变化表示和 1D 骨干，把输出改为
**class10 / class11 / 保留原类别**，并加入 fit 的真实 YOLO
预测框训练。只做 GT/val 评估，不训练 YOLO，不使用 test。

## 0. 结论先行

- fit 预测导出完成：5,957 样本、96,051 views、**1,389.6 s（23.2 min）**；
  fit 路由候选 **2,589**（class10/11/keep 874/835/880），
  train/dev **2,081/508**；val 654 个候选复用缓存。
- V2 三分类 20 epoch、seed 1729，训练 **113.8 s**；dev 候选
  accuracy **45.9%**、balanced **45.9%**，三类召回
  0.433 / 0.563 / 0.380。keep 类学得不好。
- 第二张 AP 表：V2 argmax mAP50 **0.663458（+0.386 pp）**，
  AP10 0.682369、AP11 0.688111；但 195 改对 / 132 改坏 /
  165 未匹配改动，风险高。V2 confidence≥0.70 为
  **+0.149 pp**，55/42/30，低于 V1 的 +0.352。
- V2 argmax 的 matched-only 诊断 **+0.617 pp**，unmatched-only
  **−0.221 pp**；收益仍主要来自匹配框，未匹配框是主要损伤。
- **判断：V2 不采用为主规则。** V1 amplitude_phase_change +
  confidence≥0.70 仍是当前更稳的选择；V2 需要先修
  keep 类学习、class10 和未匹配抑制。

## 1. 数据与标签

- 检测器：冻结 YOLO26l `best.pt`，filtered_0_5；
- fit 预测：`/root/autodl-fs/runs_local/srrc-gt-pilot/yolo26-fit/`
  （`fit_predictions_filtered_0_5.npz`、`fit_sample_ids.json`）；
- 候选路由：预测 class10/11、score≥0.01、宽度 9.5–10.5 MHz；
- 标签：
  - 与 GT class10/11 一对一 IoU≥0.5 匹配 → class10/11；
  - 未匹配、重复、非目标 → 保留原 YOLO 类别；
- 按源文件 `group_id` 分组；已知组沿用原 fit train/dev 划分，
  新组用固定 hash 分到约 20% dev；
- 输入：8 窗 × 1,024 点，20 MS/s，幅度＋相邻复相位变化。

## 2. V2 训练

| 项 | 数值 |
| --- | ---: |
| fit 预测样本 / views | 5,957 / 96,051 |
| 导出耗时 | **1,389.6 s（23.2 min）** |
| fit 路由候选 | **2,589** |
| 标签 class10/11/keep | **874 / 835 / 880** |
| train / dev 候选 | **2,081 / 508** |
| 训练 | 20 epoch，batch 8，lr 1e-3，seed 1729 |
| 训练耗时 | **113.8 s** |
| dev accuracy / balanced | **0.4587 / 0.4586** |
| dev recall 0/1/2 | 0.433 / 0.563 / 0.380 |
| dev confusion | `[[77,45,56],[13,94,60],[33,68,62]]` |

- 骨干从 V1 `amplitude_phase_change_last.pt` 初始化，2 类头换成
  3 类头；前两类复制 V1 权重，keep 类零初始化。
- 类别权重按 train 频次平衡（约 0.997/1.038/0.967），但没有
  解决 keep 类学习；dev 上 keep 只召回 0.380。
- 说明 V2 的“保留”能力还很弱；argmax 规则仍改了 416/654
  个候选，远多于真实 keep 目标。

## 3. 第二张完整 AP 表

基线：mAP50 **0.659601**，mAP50:95 **0.597872**，
AP50 class10 **0.712889**，class11 **0.603582**。
同一 1,543 样本、同一 654 个候选、固定框/分数/NMS。

| 方法 | 规则 | mAP50 | Δ mAP50 | mAP50:95 | AP50 c10 | AP50 c11 | 改对 | 改坏 | 未匹配改动 | 净 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 纯 YOLO | — | 0.659601 | — | 0.597872 | 0.712889 | 0.603582 | — | — | — | — |
| V1 幅度+相位 | conf≥0.70 | 0.663122 | **+0.352 pp** | 0.601828 | 0.708683 | 0.657088 | 57 | 14 | 54 | +43 |
| V2 三分类 | argmax | **0.663458** | **+0.386 pp** | **0.602216** | 0.682369 | **0.688111** | 195 | 132 | 165 | +63 |
| V2 三分类 | conf≥0.70 | 0.661092 | **+0.149 pp** | 0.599213 | **0.718809** | 0.618545 | 55 | 42 | 30 | +13 |

补充：V1 argmax 为 **+0.0003 pp**（133/69/136），仍不如
V1 的 0.70 gate。V2 argmax 的总体 mAP 最高，但改动量和未匹配
改动远大于 V1；V2 0.70 更保守，但低于 V1 0.70。

### 匹配 / 未匹配拆分

| 方法/规则 | matched-only Δ | unmatched-only Δ |
| --- | ---: | ---: |
| V1 幅度+相位 ≥0.70 | **+0.468 pp** | −0.092 pp |
| V2 argmax | **+0.617 pp** | −0.221 pp |
| V2 ≥0.70 | +0.136 pp | +0.010 pp |

V2 argmax 的匹配收益最大，但未匹配损伤也最大；full mAP +0.386
是两者相抵后的结果，不能直接当成稳定系统收益。

## 4. 判断

### 4.1 这版说明了什么

1. **真实预测框训练没有自动解决 keep/未匹配问题。** V2 argmax
   仍改 416/654 个候选，其中 165 个未匹配；V1 0.70 只改 125 个。
2. **V2 的排序/匹配收益更高**（matched-only +0.617），但
   未匹配损伤也更大（−0.221）；总体 +0.386 并不稳。
3. **V2 的 keep 类学习很差**：dev 三类 balanced 只有 0.459，
   keep 召回 0.380；用 argmax 时几乎不会主动保留。
4. **V1 0.70 仍是更稳的规则**：+0.352，57/14/54，未匹配改动
   54；V2 0.70 只有 +0.149。
5. V2 的 class10 AP 在 argmax 下降到 0.682，仍存在
   QPSK→16QAM 偏置；0.70 gate 把 class10 拉回 0.719，但
   class11 收益缩小。

### 4.2 下一步

**不采用 V2 作为主规则，V1 0.70 保留为当前基线。**

下一步按优先级：

1. **修 V2 的 keep 类学习**：当前 3 类头从 2 类头初始化，
   keep 零初始化且只训 20 epoch；需要更长训练/更低学习率/
   分层采样，或显式 keep 负样本挖掘。
2. **部署可见的未匹配/重复框抑制**：V2 argmax 的 matched-only
   很强，但 unmatched-only 为负；需要置信度、几何稳定性、
   重复框关系等预测侧特征，而不是真值匹配。
3. **class10 校准**：继续看 QPSK→16QAM 的错误，先做类先验/
   温度校准，再决定是否用 V2。
4. **V2 只有在“匹配收益保留、未匹配损伤下降、class10 不退化”
   同时成立时才替换 V1；当前不接检测器。**

### 4.3 效率记录

- fit 导出：5,957 样本 / 96,051 views / **1,389.6 s**；
- fit 候选缓存：2,589 个候选，12 workers，**239 s**；
- val 候选缓存：654 个候选，8 workers，**112 s**（串行 V1 为
  608 s，约 5.4×）；
- V2 训练：**113.8 s**；
- V2 评估直接用 val 缓存，不再重复裁 IQ。
- 结论：候选缓存化 + 并行裁剪是有效的；下一次 fit 导出可
  benchmark 8 vs 16 views、12 vs 14 workers，再选最快配置。

## 5. 产物

- 新入口：`scripts/baseline3srrc_v2.py`
- fit 预测：`/root/autodl-fs/runs_local/srrc-gt-pilot/yolo26-fit/`
- fit/val 候选缓存：
  `/root/autodl-fs/runs_local/srrc-gt-pilot/yolo26-label-only-v2/{fit,val}_candidates.npz`
- V2 权重：`.../v2/amplitude_phase_change_3class_last.pt`
- 结果：`reports/baseline3_srrc_yolo_v2/{train_summary.json,val_summary.json,*_candidates.csv}`
- 本轮未训练 YOLO，未使用 test；只做冻结检测器上的仅改类 AP。
