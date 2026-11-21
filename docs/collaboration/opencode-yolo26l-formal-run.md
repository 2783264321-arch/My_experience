# YOLO26l 正式训练：配置、验收与运行记录

2026-11-22。决策：唯一一场正式训练给 **YOLO26l**（依据见
[detector-upgrade](opencode-detector-upgrade.md)：同有效 batch 下约 2.05× 步时
优势、显存低约 2 GiB、原生链路已验证）。test 继续封存。

## 1. 正式配置

| 项 | 值 |
| --- | --- |
| run | `yolo26l_c1_2048x512` |
| 模型 | YOLO26l，单通道 `[N,1,2048,512]`，nc=14，one2many 推理 |
| 初始化 | 官方 `yolo26l.pt`，RGB stem 求和、两组分类头重初始化 |
| 数据 | 冻结 fit 5,957 / val 1,543；C1 log-power 粗图＋全部频率细片 |
| 物理 batch | 4 源 = **8 视图**；累积 **2** → 等效 16 视图 |
| 优化器 | AdamW，lr0 1e-3，wd 5e-4，warmup 3 轮，余弦到 1%，clip 10 |
| 精度 | BF16 autocast |
| EMA | 0.9999 / tau 2000 |
| 轮数 | 50；745 优化步/轮；每 5 轮一次完整 val（10 次） |
| 验证 | 24,975 视图；conf 0.001 / IoU 0.5 / top-300 片内、top-100 全局 |
| 主指标 | 细片 0–5 过滤后的 strict mAP@0.5；参照 **0.649725** |
| 次指标 | 无过滤 mAP50、mAP@0.5:0.95、不计类 AP、WiFi/SRRC 分组 |
| checkpoint | `best.pt` 按过滤 mAP50；`last.pt` 每轮含优化器/EMA/调度状态 |

## 2. 正式入口验收（已完成）

- **训练入口 5 步 smoke**：loss 325.2，**box 8.43 / cls 131.4 / dfl 0.23**
  ——box loss 非零，确认 TAL 正样本分配正常；两套检测头均有梯度。
- **checkpoint**：402 MB，首层 `[64,1,3,3]`，one2one 分类头 `[14,256,1,1]`；
  优化器 558 个 state、EMA 1,092 个键、criterion `{updates:1,o2m:0.7857}`。
- **续训验收**：从 `last.pt` 恢复后 epoch 1 step 0 正常，loss 144.5
  （box 5.76 / cls 57.67 / dfl 0.14），criterion updates → 2、o2m → 0.7714；
  优化器与 EMA 状态恢复成功。
- **验证入口**：按 checkpoint 重建、one2many；24 个分层样本（**每个带宽
  4 个，含 80 MHz**，共 412 视图）10.4 s 完成；过滤/无过滤、mAP50:95、
  分组全部产出。该 checkpoint 仅 6 步，指标接近 0，属链路 sanity。
- `bench_steps.py` 的 YOLO26 分支在 `main` 已是归一化 cxcywh（无需再改）；
  `bench_val.py` 的子集选择已修为每个带宽都保留（原先 24 样本会截掉 80 MHz）。

验收工件：`reports/detector_upgrade/accept_yolo26_smoke.json`、
`reports/detector_upgrade/accept_yolo26_val_subset.json`。

## 3. 首轮实测与预算

| 项 | 实测/估算 |
| --- | ---: |
| 首轮训练（745 步） | **306 s** |
| 稳定步时 | ~0.40 s/优化步 |
| 50 轮训练 | ~4.25 h |
| 10 次完整 val（双口径，单次推理） | ~1.2 h |
| **合计** | **~5.5 h** |

数据管线 8–12 workers 预取，端到端步时即模型主导；显存峰值约 14.9 GiB。

## 4. 监控与决策规则

- 第 5 轮（epoch 4）首次完整 val；关注过滤 mAP50、recall、box loss 与
  每轮训练 loss。若明显偏低，先查正样本（box loss 是否非零）、学习曲线
  和评估日志，**不因单次 AP 自动换模型**。
- 正式对照完成后同时报告：过滤/无过滤 mAP50、mAP50:95、不计类定位 AP、
  WiFi 0–5（含 0–2、3–5）与 SRRC 10–11 分组。
- test 继续封存；所有选择只基于 val。

## 5. 工件与命令

- 训练：`scripts/train_yolo26.py`（`--resume` 从 `last.pt` 继续）
- 验证：`scripts/validate_yolo26.py`（`--stratified 4` 为含 80 MHz 的子集）
- 双口径核心：`scripts/detector_upgrade/validate_dual.py`
- 配置快照：`reports/detector_upgrade/formal_yolo26_config.json`
- 运行目录：`runs/yolo26l_c1_2048x512/`（`train_log.csv`、`val_history.json`、
  `last.pt`、`best.pt`；`runs/` 不入库）
- 最终结果：训练结束后由 `scripts/finalize_yolo26.py` 生成
  `reports/detector_upgrade/formal_yolo26_results.{json,md}` 并回填本文。

## 6. 最终结果

（训练进行中；结束后回填过滤/无过滤 mAP50、mAP50:95、AP_agn、分组与
与 0.649725 的差值。）
