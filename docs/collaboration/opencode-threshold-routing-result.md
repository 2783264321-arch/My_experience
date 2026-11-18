# OpenCode 结果：0–5 分类器阈值/路由复核

依据 `docs/collaboration/README.md` 与 `opencode-coarse-classifier-result.md`。
对照：共享 fine-0-5 过滤完整 val **0.649725**（AP_agn 0.905977，R_agn
0.957833，n_pred 47,784；0/1/2 0.4212，3/4/5 0.4550；177 s）。

## 1. 留出集

复用现有两个小分类器与 fit crops 的按样本 10% holdout（seed 1729）：

- 761 个 holdout crops：GT 0/1/2 457，GT 3/4/5 304。
- 检测器预测组：0/1/2 471，3/4/5 279，0–5 之外 11。
- 同一 GT/pred 组 701；**跨组 60（7.9%）**。
- 训练按真值类组取样；部署按预测类组路由。跨组框会被错误分类器处理，
  预测在 0–5 之外的框则不会改类。

## 2. 置信度门槛（holdout）

`changed / corrected / wrong / net / ratio / precision`

| routing | t | changed | corr | wrong | net | ratio | prec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT-routed（oracle 路由） | 0.0 | 452 | 150 | 302 | -152 | 0.594 | 0.332 |
| GT-routed | 0.5 | 132 | 44 | 88 | -44 | 0.173 | 0.333 |
| GT-routed | 0.70 | 16 | 6 | 10 | -4 | 0.021 | 0.375 |
| pred-routed same（干净部署子集） | 0.0 | 392 | 127 | 265 | -138 | 0.559 | 0.324 |
| pred-routed same | 0.5 | 104 | 32 | 72 | -40 | 0.148 | 0.308 |
| **pred-routed same** | **0.70** | **10** | **6** | **4** | **+2** | **0.014** | **0.600** |
| pred-routed all（含跨组） | 0.0 | 426 | 127 | 299 | -172 | 0.560 | 0.298 |
| pred-routed all | 0.5 | 117 | 32 | 85 | -53 | 0.154 | 0.274 |
| pred-routed all | 0.70 | 14 | 6 | 8 | -2 | 0.018 | 0.429 |
| cross-family only | 0.0 | 34 | 0 | 34 | -34 | 0.567 | 0.000 |
| cross-family only | 0.5 | 13 | 0 | 13 | -13 | 0.217 | 0.000 |

准确率：GT-routed 0.472→0.439；pred-routed same 0.512→0.444；
pred-routed all 0.472→0.409。margin sweep 也全负（GT-routed margin 0/0.1/
0.2/0.3 = -152/-83/-50/-30）。

## 3. 路由差异

- 7.9% 跨组；预测在 0–5 之外的还有 11 个。
- 跨组改动 **全部错误**（t=0 34/34；t=0.5 13/13；t=0.7 4/4）。
- 即使用 oracle GT 路由，净纠错仍为负；问题不只在路由，分类器本身也弱。
- 唯一正 net 是 t=0.70 同组：10 改 / 6 对 / 4 错，净 +2，覆盖 1.4%；
  不构成可靠净纠错。
- 现有两家族分别划分 holdout，跨组 pred-routed 数字只作指示；要干净估计
  需要共享样本 holdout 重训。

## 4. 决定：不锁定阈值，不跑第二组完整 val

没有门槛达到可靠净纠错。按约定，未锁阈值，也没有跑复核组完整 val；
对照仍是 **0.649725**。小型 coarse-crop 分类器不可用。

## 5. 下一步：先试更强的粗图模型，再考虑同框原生 STFT

**建议先做更强的 coarse 分类器，不先做 native crop。**

- 当前小 CNN 是主要混淆项：欠拟合（loss ~0.98/0.90）、holdout 接近随机；
  即使 oracle GT 路由也负。
- 路由错配只有 7.9%，不足解释失败；先排除分类器容量/训练问题。
- native crop 更高（7–27 倍）且 resize 会丢细节；等强 coarse 模型仍失败
  再做更合理。

下一轮分类器建议：

- 两家族共享样本级 holdout；6 类+background（处理 2↔3 跨组）或家族头+gate。
- ResNet18/ConvNeXt-Tiny（1 通道），或检测器 P5/ROI 特征；更多 hard
  negatives、轻量增强、更长训练。
- confidence/margin gate，阈值在共享 fit holdout 上定，不在 val 上调。
- 仍用全局合并前同一批候选框，只改类、保留框/分数，两组各合并一次。

若强 coarse 模型仍无可靠净纠错，再做同框 native STFT 对照：同 proposal
框/物理 context/结构/划分/融合规则；用 256×256/512×512 或 1D frequency
profile/双分支；实测裁块 0.15–0.27 ms、分类器 0.07–0.15 ms/proposal，
4.8k 框约多 1–5 s。

10/11 仍单独处理；6/7/8/9/12/13 不做分类；test 继续封存。

详细表：`reports/classifier_threshold_and_routing.md`、
`reports/classifier_threshold_sweep.json`。
