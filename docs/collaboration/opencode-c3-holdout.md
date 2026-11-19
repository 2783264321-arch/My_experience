# C3 1,159 样本留出复核（进行中）

起始提交：`949873a`
状态：**已完成**
test 继续封存。

## 目标

确认 dense 短窗能否让**已经定位正确的 40 MHz Wi‑Fi（3/4/5）**更容易判对
调制。使用完整 val 1,543 中剔除 384 样本 C3 子集后的 **1,159 个样本**；
不重新训练、不重新选权重、不调阈值。

## 冻结权重

- dense seed1729：`best.pt`、`last.pt`
- dense seed1730：`best.pt`、`last.pt`
- zero3 seed1729：`best.pt`、`last.pt`
- zero3 seed1730：`best.pt`、`last.pt`

best/last 均按原 384 子集选择，留出集只做评测。

## 缓存

- holdout power：`cache/c3_holdout/val_power_bf16.npy`（1,159×1×2048×512）
- holdout dense complex：`cache/c3_holdout_dense/val_complex.npy`
  （1,159×2×2048×512 float16）
- dense 算子/scale 不变；抽样核验功率与复数缓存。

## 评测

- 原有 conf=0.001、per-view NMS IoU=0.5、max_det_view=300、全局
  class-wise NMS IoU=0.5、max_det=100；不调阈值。
- 报告整体 mAP、AP_agn、定位召回、分组 AP。
- 共同框：两个模型都在 IoU≥0.5 匹配到同一个 3/4/5 GT 才计入 common；
  报告分类准确率、3×3+other 混淆、单方正确数量、McNemar p。
- best/last 分别报，不合并。

## 数据核对

- holdout **1,159** 样本，覆盖全部 31 个 B/T 组；
- GT **4,966** 个，其中 3/4/5 GT **290** 个（原 384 子集为 152 个），
  共同框统计功效应明显好于上一轮。

## 状态

- 2026-11-19 17:33：holdout 列表 1,159 已核对；power/dense 缓存构建中；
  冻结权重已确认存在。预计 18:00 左右完成评测。


## 结果与判断（2026-11-19 17:50）

- holdout 1,159 样本，3/4/5 GT 290 个；common 框 239–268 个/对比。
- **best 均值**：dense mAP 0.4983，zero3 0.4990；3/4/5 AP dense 0.3874，
  zero3 0.3489；common 3/4/5 分类 dense 0.4804，zero3 0.4454。
- **last 均值**：dense mAP 0.4583，zero3 0.4520；3/4/5 AP dense 0.3163，
  zero3 0.3465；common 分类 dense 0.4697，zero3 0.5032。
- 逐 seed/ckpt：
  - s1729 best：common dense 0.4308 vs zero3 0.4654（dense 更差，p=0.380）
  - s1729 last：0.4875 vs 0.5000（p=0.822）
  - s1730 best：**0.5299 vs 0.4254（dense +10.45 pp，p=0.007）**
  - s1730 last：0.4519 vs 0.5063（p=0.228）
- **结论：1,159 留出集没有复现 384 子集上 dense 4/4 的分类优势。**
  只有 s1730 best 一个显著点；s1729 两个 ckpt 和 s1730 last 都偏向 zero3。
  best 与 last 的排序还会翻转，说明 384 子集的 checkpoint 选择不稳定。
- 定位召回 dense 不高于 zero3；唯一显著结果是已定位框上的分类差异，
  不是找到更多框。
- **不建议扩大全量训练。** 若继续，只能做预注册 checkpoint 规则（如 last）
  的小规模多 seed 复核，并加 power+power+power/随机活通道的“活容量”对照；
  否则停止 C3-for-3/4/5。
- 完整表：`reports/c3_holdout_replication.md`；
  匹配明细：`reports/c3_holdout_matched_*.json`。
