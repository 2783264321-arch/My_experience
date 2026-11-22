# 三模型评测入口修正（D-FINE 轮，2026-11-22）

## 1. 运行版本与续训核查

- 进程 PID 7103，worktree `formal-dfine` @ `11aa7c1`，14:23 启动。
- 训练日志从 `epoch 0 step 1` 开始，**没有 `resumed` 行**；正式 D-FINE run
  **未发生续训**。当前 last.pt 的 epoch/step 与训练进度一致。
- `ema.updates` 未写入 checkpoint（旧格式）；因为未续训，EMA warmup 完整，
  对本轮无影响。已在 `train_dfine.py` 增加 `ema_updates` 保存/恢复：若从
  旧 checkpoint 续训，`updates` 会从 0 重新 warmup，需在报告中注明；建议
  非必要不续训。
- accept_dfine 的 smoke 曾续训（独立小跑），不进入正式结果。

## 2. 问题一：最终 AP@0.75 读取了缓存中不存在的 GT 字段

- 原 `validate_dfine.py` 的 `val_predictions_*.npz` 只有 `pred_*`；而
  `finalize_dfine.py` 的 `ap75()` 读取 `gt_boxes/gt_labels/gt_sample_index`
  → 会直接失败。
- 修复：验证 entry 现在同时保存 GT（`gt_*`）字段；finalizer 用同一缓存
  计算 AP@0.75。24 样本 probe 已验证 AP_agn@0.75 = **0.6510**。

## 3. 问题二：D-FINE 解码跳过 0.001 分数门槛

- 原 `validate_dfine.py` 直接把 postprocessor 的 top-300 全部送入合并，
  没有按公共 `conf=0.001` 过滤；YOLO 路径则由片内 NMS 的 conf 过滤。
- 修复：D-FINE infer 现在对每视图 top-300 施加 `score >= 0.001`。
- **同一 checkpoint（epoch 2）24 样本实测**：该 checkpoint 的 top-300 最低分
  为 0.0415/0.0618，**没有低于 0.001 的框**；阈值前后指标完全一致：
  - filtered mAP50 0.57216 / 50:95 0.47761；
  - unfiltered 0.57072 / 0.47622。
  所以本轮权重上无差异，但协议已一致；早期/随机或分数更低的 checkpoint 会
  体现差异。

## 4. 问题三：分组定位召回在匹配前按预测类筛框

- 原 `validate_dual._eval_set` 的分组评测先把预测限制在组内类别，再做匹配：
  一个定位正确但预测成组外类别的框会被计为**漏定位**。
- 修复：分组 `recall` 现在用**全部预测框**做类别无关匹配、只限制 GT 组；
  严格组 AP50 仍按组内预测类过滤（class-aware 的正确口径）。旧值保留为
  `recall_strict_predclass` 以便对照。
- 同一 probe（epoch 2，24 样本）实测：
  - SRRC 10/11：旧 **0.90** → 新 **1.00**（+10 pp）；
  - WiFi 0–5：0.9375 → 0.9375；窄带：0.7742 → 0.7742。
  说明该 bug 主要影响跨组错类的 SRRC/窄带样本，总体 mAP 不受影响。

## 5. 可用权重与复评范围

- D-FINE run 只保留 `best.pt`（按过滤 mAP50）与 `last.pt`（每轮覆盖）；
  **没有逐验证点权重快照**。训练结束后 finalizer 会用修正口径完整复评
  `best.pt` 与 `last.pt`，并在结果中给出对应 epoch；中间验证点只有
  `val_history` 指标，不能做权重级复评。
- YOLO11s（`best_corrected.pt`）与 YOLO26l（`best.pt`/`last.pt`）保留；
  三模型最终表统一用修正后的分组召回与 AP@0.75 口径。

test 继续封存。
