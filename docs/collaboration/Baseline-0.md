# Baseline-0：全频单图基线（Group A）

状态：**执行中**
会话：Baseline-0（OpenCode）
工作树：`/root/autodl-tmp/worktrees/baseline-0-group-a`
分支：`baseline-0/group-a`（基于 `main@87b6a22`）
日期：2026-11-18

## 目标

补做论文所需的独立训练「全频单图」基线 A：

- 每个 fit 样本只生成一张原生 `F×512` STFT 的 0.7 ms 全频聚合图 `2048×512`；
- 从与 B/C 相同的 `yolo11s.pt` 预训练起点独立训练 YOLO11s-C1-DirectReg；
- 验证/推理每样本只跑这一张粗图；
- 与 B（同一联合权重仅跑粗图，0.5794）和 C（粗+细，0.6427）对照。

A、B、C 主表统一使用**最后一轮 EMA**；A 的 `best` 另列。B 使用 C 的同一份末轮 EMA 权重。

## 已确认口径

| 项目 | 决定 |
|---|---|
| 训练预算 | 近似等预算：595,700 张训练视图；有效 batch 16；37,300 vs 37,250 步（+0.13%） |
| A epoch | 100（每轮 1 张粗图/源）；B/C 为 50（每轮 1 粗 + 1 细） |
| batch | A：16 源 = 16 views；B/C：8 源 = 16 views |
| warmup | A 6 epoch = 2,238 步；B/C 3 epoch = 2,235 步 |
| 验证节奏 | A 每 10 epoch = 3,730 步；B/C 每 5 epoch = 3,725 步 |
| seed | 1729 |
| 缓存 | fit 粗图 BF16 位模式，约 11.63 GiB；val 现场 float32 |
| 缓存核验 | 抽样位模式/张量/标签 + 同权重同批次前向、loss、梯度；有实质偏差回退在线 |
| 主 checkpoint | A/C 末轮 EMA；A best 另报 |
| 延迟 | A vs C 同一批 128 个分层 val 样本，GPU 空闲，3 次重复，p50/p95 |
| 排期 | 等 Baseline-1 当前 GPU/CPU 任务结束；A 正式训练独占 GPU |
| test | 继续封存 |

## 报告口径

- `B−A`：都只跑粗图时，训练加入细片带来的变化。
- `C−B`：同一联合权重加切片推理的变化。
- `C−A`：整套切片方案相对全频单图的收益。
- 窄带同时报：物理带宽 `≤0.25 MHz` 与类别组 `9/12/13`，均用同一脚本重算。
- 训练开销分列：缓存构建、训练 wall time、推理延迟。

## 代码计划

- `src/signals/data.py`：coarse-only dataset、BF16 缓存 dataset、`coarse_targets()`、`collate_coarse()`。
- `src/signals/validate.py`：`view_mode="coarse"`，每样本 1 view。
- `src/signals/infer.py`：`predict_sample(..., view_mode="coarse")` 与 `predict_views()`。
- `src/signals/train.py`：`--view-mode`、`--coarse-cache`，记录预算字段。
- `scripts/validate_run.py`：`--view-mode`。
- 新增 `scripts/build_coarse_cache.py`、`scripts/verify_coarse_cache.py`、`scripts/bench_view_modes.py`、`scripts/class_group_metrics.py`、`scripts/run_group_a.sh`。

## 状态日志

### 2026-11-18 20:29

- 创建独立 worktree 与 `runs/`、`cache/`，`.venv`/`yolo11s.pt` 软链到主工作树。
- 确认 `main@87b6a22`。
- Baseline-1 正在训练 `train_local_representation_classifiers.py`（GPU 12.4 GiB），并启动 `collect_local_proposal_crops.py --win-ms 0.35,0.7`（CPU 451%）。等待其结束，再建缓存/启动 A。
- 本阶段只做代码与文档，不占 GPU。

### 2026-11-18 21:05

- 代码提交 `d6aa382`：coarse-only 训练/验证/推理路径、BF16 缓存构建与核验、延迟基准、类别组指标、对照报告脚本；`view_mode` 默认仍为 `all`，不影响现有 B/C 路径。
- 已完成 CPU 侧 smoke：
  - `SpaceNetCoarseViews` 与 `build_sample_views(..., [])` 的框/标签逐元素一致；
  - 8 样本 BF16 缓存与在线路径逐元素一致，标签一致；
  - `FullViewDataset(view_mode="coarse")` 每样本 1 个 view。
- Baseline-1 的裁块任务已结束（输出 4.6 GB），但当前有 3 个 100-epoch 表示分类器任务占用约 20 GiB GPU。A 训练继续等待 GPU 独占。
- 全量 fit BF16 缓存已完成：`5957×1×2048×512`，11.63 GiB，构建 **2049.6 s**（约 34 min，1 线程；与 Baseline-1 分类任务并行）。32 样本核验：BF16 位模式逐元素一致（`max_abs_diff=0`），框/标签逐元素一致。
- 回归门通过，且与历史逐位一致：
  - C 全视图：mAP50 **0.6427008648**，AP_agn 0.9045937745，R 0.9579726333，n_pred 65,025，24,975 views。
  - C 仅粗图（B）：mAP50 **0.5793865644**，AP_agn 0.8177921708，R 0.8772689193，n_pred 25,409，1,543 views。
  - B 类别组 AP：0/1/2 0.4212，3/4/5 0.4550，10/11 0.5908，9/12/13 0.4838。
- 缓存训练路径核验通过（4 样本、同权重、同 batch）：在线重复噪声为 0；缓存 vs 在线 loss 差 0、preds 差 0、grad 最大差 0、grad rel-L2 0、cosine 1.0；图像 BF16 位模式一致，框/标签一致。
- 16 源 smoke 首次尝试 OOM：Baseline-1 同时启动了两个新 GPU 任务（12.14 + 8.29 GiB），GPU 仅剩 178 MiB；不是代码错误。等待 GPU 完全空闲后再跑 smoke 和 A 正式训练。

### 2026-11-18 21:57（节点 1：首个 val）

- Baseline-1 sweep 结束，wrapper 检测到 GPU 空闲；64 源 smoke 完成（80 updates，val mAP 0.0561，只作代码路径 smoke，不作质量门）。
- 正式 A 于 **21:45** 启动，GPU 独占 10.84 GiB；epoch 约 **50 s**，当前 epoch 11/100。
- 配置核对：37,300 steps、warmup 2,238、595,700 image views、有效 batch 16。
- **epoch 9 首个 val（3,730 步）**：
  - mAP50 **0.4405**
  - AP_agn **0.7590**
  - localization recall **0.8443**
  - predictions 53,380 / 1,543 views；val 77 s
- 参照：C 在 epoch 4（3,725 步）为 mAP50 0.3623、AP_agn 0.6789、R 0.843；A 在同 step 预算下更高，但 A 对每个源看了 10 次粗图，C 是 5 粗 + 5 细，不能直接等同为方法优劣。
- 下一个汇报节点：epoch 49 val（约 18.6k steps）；之后 epoch 99 完成。

### 2026-11-18 22:35（节点 2：epoch 49）

- val 曲线（EMA、coarse-only、1,543 源）：

| epoch | steps | mAP50 | AP_agn | R_agn | pred |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 3,730 | 0.4405 | 0.7590 | 0.8443 | 53,380 |
| 19 | 7,460 | 0.5546 | 0.8195 | 0.8843 | 37,063 |
| **29** | **11,190** | **0.5763** | 0.8208 | 0.8789 | 30,989 |
| 39 | 14,920 | 0.5745 | 0.8185 | 0.8746 | 27,277 |
| 49 | 18,650 | 0.5611 | 0.8177 | 0.8721 | 24,995 |

- `best.pt` 当前来自 epoch 29（0.5763）；epoch 49 的 last EMA 为 0.5611，且仍在下降。训练 loss 已降到 ~15，验证 mAP 在 29 后回落，呈现粗图单视图过拟合的迹象。
- 与 B（同一联合权重仅粗图）对照：A 当前 last 0.5611（-1.83 pp），A best 0.5763（-0.31 pp）；C 0.6427。
- 按约定完成 100 epoch 等预算训练，不用 val 提前停；主表仍用末轮 EMA，A best 另列。
- 训练速度约 50 s/epoch + 每 10 epoch 77 s val；预计 **23:25 左右**完成。
- 下一个节点：epoch 99 完成。

## 待确认 / 风险

- Baseline-1 的 GPU 任务占用 12.4 GiB，A 训练必须等它结束，避免 OOM/争抢。
- 缓存 11.63 GiB；磁盘当前剩余 36 GB，需持续监控。
- 单 seed 1729；如 A 与 C 差距很小，再补一对同 seed A/C 结果判断波动。
- 若 A 末轮仍在上升，补 8 epoch 低 LR 续训作为收敛诊断，不替换主结果。

## 提交记录

- 本文件首次提交：见 `git log --oneline -- docs/collaboration/Baseline-0.md`。
