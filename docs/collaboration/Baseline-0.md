# Baseline-0：全频单图基线（Group A）

状态：**已完成，Baseline-0 收尾**
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

## 后续实验早停策略（2026-11-18 两次训练复盘）

当前 A 这轮**不早停**，跑满 100 epoch，作为无早停对照。后续实验统一启用早停。

### 两次训练的实际曲线

C（50 epoch，训练期旧口径、未做视图裁剪）：

| epoch | 14 | 19 | 24 | 29 | 34 | 39 | 44 | 49 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mAP50 | 0.5174 | 0.5299 | 0.5389 | **0.5429** | 0.5423 | 0.5399 | 0.5355 | 0.5319 |
| AP_agn | 0.7395 | 0.7327 | 0.7388 | 0.7389 | 0.7415 | **0.7423** | 0.7410 | 0.7417 |

- mAP 的 95% 水平（0.5158）在 epoch 14 已达到；24–34 基本平台，之后回落。
- 但修正版完整 val 是 **best ep29 = 0.6388，last ep49 = 0.6427**。若用旧口径早停，会在 29/34 停，错过末轮最好结果。**后续早停必须用修正版指标。**
- 8 epoch 低 LR 续训的修正版 val：0.6384 / 0.6415 / 0.6406 / **0.6423** / 0.6414 / 0.6398 / 0.6403 / 0.6405；+3 之后进入平台。

A（100 epoch，修正版 coarse-only）：

| epoch | 9 | 19 | 29 | 39 | 49 | 59 | 69 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mAP50 | 0.4405 | 0.5546 | **0.5763** | 0.5745 | 0.5611 | 0.5557 | 0.5508 |
| AP_agn | 0.7590 | 0.8195 | **0.8208** | 0.8185 | 0.8177 | 0.8135 | 0.8114 |

- mAP 的 95% 水平（0.5475）在 epoch 19 已达到；AP_agn 的 95% 也在 19 达到；best 在 29，之后单边下降。
- 到 epoch 69 已比 best 低 2.55 pp，且 AP_agn、recall 同步下降，是明确的过拟合，不是单点噪声。

### 采用的参数

- 成熟 JDM 方案：`patience=15` epoch，`min_delta=0.1pp`。C 的验证间隔是 5 epoch，15 epoch 约等于 **3 次验证事件**。
- 按用户建议的 1.2N：3 × 1.2 = 3.6，取 **4 次验证事件**。
- `--early-stop-evals 4`、`--early-stop-min-delta 0.001`（0.1pp）、`--early-stop-min-evals 3`。
- 4 次事件约 **14,920 步**（A 约 40 epoch，C 约 20 epoch），与每 epoch 步数无关，A/C 可直接比较。
- 对 A：best ep29 之后在 39/49/59/69 四次未提升，会在 **ep69** 停，省 31 epoch（约 26 min，31%）。
- 对 C：best ep49 是末轮，4 次事件不会早停；3 次会在 ep44 停，早于已知的 ep49 最好结果，所以 3 次被否。
- 5 次事件会在 A 的 ep79 才停，只省 21%，而模型已比 best 低 4 pp 以上，偏浪费；4 次是“宽松但有效”的折中。
- 后续主 checkpoint 用 **best EMA**（早停后 last 通常不是最优），同时报 last、stopped_epoch、actual_optimizer_steps；所有对照臂使用同一早停规则。
- 代码：`src/signals/train.py` 新增 `--early-stop-evals/min-delta/min-evals`，默认 0 关闭；当前 A 这轮不受影响。

## 代码计划

- `src/signals/data.py`：coarse-only dataset、BF16 缓存 dataset、`coarse_targets()`、`collate_coarse()`。
- `src/signals/validate.py`：`view_mode="coarse"`，每样本 1 view。
- `src/signals/infer.py`：`predict_sample(..., view_mode="coarse")` 与 `predict_views()`。
- `src/signals/train.py`：`--view-mode`、`--coarse-cache`，记录预算字段。
- `scripts/validate_run.py`：`--view-mode`。
- 新增 `scripts/build_coarse_cache.py`、`scripts/verify_coarse_cache.py`、`scripts/bench_view_modes.py`、`scripts/class_group_metrics.py`、`scripts/run_group_a.sh`。

## 收尾与论文比较口径

训练已按原计划跑满 100 轮，完整评测使用固定的 1,543 个验证样本。A 是独立训练的全频单图模型。B 和 C 共用 Baseline-1 的末轮 EMA，分别使用仅粗图和粗图加细片推理。C 为未加细片类别过滤的结果。 [训练与预算](../../reports/group_a_training.md) · [完整结果](opencode-group-a-fullband-baseline.md)

| 论文用途 | 全频单图 A | 粗细联合 C | C−A mAP@0.5 |
| --- | ---: | ---: | ---: |
| 近似等预算主对照，均使用末轮 EMA | 0.5404（第 100 轮） | 0.6427（第 50 轮） | +10.23 个百分点 |
| 补充对照，给出更强的单图参照 | 0.5763（验证最佳，第 30 轮） | 0.6427（同上） | +6.64 个百分点 |

- **两行都报。** 末轮对末轮沿用事先约定的权重选择。两组各处理 595,700 张训练视图，更新步数相差约 0.13%。A 的验证曲线第 30 轮后回落，因此同时报告最佳 EMA。C 的 0.6427 是按修正版评测复核的末轮 EMA，修正版历史比较覆盖当时保存的两份权重。 [C 的复评记录](../../reports/stage5_results.md#L12-L28)
- **同权重的推理切片收益单列。** B 为 0.5794，C 为 0.6427，相差 6.33 个百分点。A 最佳和 B 相差 0.31 个百分点，两者都仅跑粗图，本轮单种子下表现接近。C 相比 A 最佳，窄带类别 `9/12/13` 的 AP 从 0.5002 升至 0.7804，贡献约九成的总体 mAP 差值。 [总表和分组](../../reports/group_a_results.md)
- **开销注明测量范围。** 在同一批 128 个分层样本上，端到端 p50 为 A 0.327 s、C 0.509 s。训练耗时和 A 的一次性缓存构建分开记。第 70 轮早停来自事后模拟，A 的实际训练与末轮权重仍以第 100 轮为准。 [延迟记录](../../reports/group_a_latency.json)
- **结果使用范围。** 当前是单种子的验证集对照，test 继续封存。未来若启用早停，配对实验在同一评测与权重选择规则下执行，同时报告最佳和末轮指标及实际训练步数。

Baseline-0 至此收尾。物理带宽组 `≤0.25 MHz` 的正文增益待 OpenCode 对照表格核正。下一阶段从 [Baseline-1 复数粗图的前端比较](opencode-complex-coarse-decision.md) 接续。

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

### 2026-11-18 23:30（节点 3/4：训练完成 + 评测）

- 100 epoch 完成，总 **1.62 h**；best **ep29 = 0.5763**，last **ep99 = 0.5403**。
- 后段 val：ep59 0.5557，ep69 0.5508，ep79 0.5468，ep89 0.5436，ep99 0.5403；训练 loss 继续下降，确认过拟合。
- A last 完整评测：mAP50 0.5404，AP_agn 0.8020，R 0.8570，pred 16,173。
- A best 完整评测：mAP50 **0.5763**，AP_agn 0.8208，R 0.8789，pred 30,981。
- 对照：
  - B（联合权重仅粗图）0.5794；C（全视图）0.6427。
  - B−A best = **+0.31 pp**；C−A best = **+6.64 pp**；C−B = +6.33 pp。
  - A best 在 10/11 0.6399（≈C）、medium 0.3–2 MHz 0.8917（>C）；窄带 9/12/13 0.5002（C 0.7804）。
- 延迟（128 样本）：A 1 view，端到端 p50 0.327 s；C 19.10 views，0.509 s；model-only C/A 8.64×，端到端 1.56×。
- 训练成本：A 1.62 h + 缓存 2,050 s（11.63 GiB）；C 2.69 h；峰值显存均 10.1 GiB。
- 早停复盘：4 次事件会在 ep69 停，省约 30 epoch（~26 min）；3 次会在 ep59 停但会误停 C（ep44）；5 次太晚。
- 详细结果：`docs/collaboration/opencode-group-a-fullband-baseline.md`、`reports/group_a_results.md`、`reports/group_a_training.md`。

## 复核边界

权重、缓存与原始运行产物保存在 AutoDL。共享目录保留配置、评测明细与结论，后续复用按本页「收尾与论文比较口径」执行。

## 提交记录

- 本文件首次提交：见 `git log --oneline -- docs/collaboration/Baseline-0.md`。
