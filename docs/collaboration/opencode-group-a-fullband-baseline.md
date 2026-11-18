# Group A：全频单图基线结果与对照

会话：Baseline-0（OpenCode）
工作树：`/root/autodl-tmp/worktrees/baseline-0-group-a`
运行：`yolo11s_c1_directreg_coarseonly_2048x512`
依据：`docs/collaboration/Baseline-0.md`、`reports/group_a_training.md`
日期：2026-11-18

## 结论先行

- 独立训练的 A（每样本只跑 1 张全频粗图）**best EMA = 0.5763**，last EMA = 0.5404。
- B（同一联合权重仅跑粗图）= **0.5794**；C（粗+细全视图）= **0.6427**。
- 以 best 比较：A 与 B 基本持平（B−A = **+0.31 pp**），C 比 A 高 **+6.64 pp**；以 last 比较：C 比 A 高 +10.23 pp，但 A 的 last 已明显过拟合。
- A best 并非全面落后：它在 **10/11（0.6399）** 与 **medium 0.3–2 MHz（0.8917）** 上优于 B/C；主要短板是 **窄带 9/12/13（0.5002 vs C 0.7804）**、`0/1/2`、`3/4/5` 以及 `wide/verywide/ultra`。
- 切片方案 C−B 的 **+6.33 pp** 主要来自窄带；宽带组 B 仅粗图反而更好，与 view-source attribution 的结论一致。
- A 的 last 曲线在 epoch 29 后单边下降（0.5763 → 0.5403），训练 loss 仍在下降，是明确的过拟合。按 4 次验证事件的早停规则，它会在 **ep69** 停，省约 30 epoch（~26 min）；后续实验统一启用早停并用 **best EMA** 作主 checkpoint。
- 推理成本：A 每样本 1 view，C 平均 19.10 views；模型部分 C/A = **8.64×**，端到端 p50 仅 **1.56×**（STFT 主导）。训练 A 1.62 h vs C 2.69 h；A 另有一次 34 min 缓存构建。

## 协议与预算

| 项目 | C（联合） | A（全频单图） |
| --- | ---: | ---: |
| 每源每轮视图 | 1 粗 + 1 旋转细片 | 1 粗 |
| epoch | 50 | 100 |
| 物理/有效 batch | 8 源 = 16 views | 16 源 = 16 views |
| 优化步数 | 37,250 | 37,300（+0.13%） |
| 训练图像呈现 | 595,700 | 595,700 |
| warmup | 2,235 步 | 2,238 步 |
| 验证节奏 | 3,725 步 | 3,730 步 |
| 初始化 | `yolo11s.pt` | `yolo11s.pt` |
| seed / 精度 | 1729 / BF16 | 1729 / BF16 |
| 训练输入 | 粗图 + 细片 | BF16 位模式粗图缓存 |
| 主 checkpoint | 末轮 EMA | 末轮 EMA（best 另列） |

A 的 100 epoch 不是增加训练量：单图每轮只有一半图像，按图像/步数对齐后与 C 的 50 epoch 等价。

## 总体结果

| 指标 | A last | A best | B coarse | C all |
| --- | ---: | ---: | ---: | ---: |
| mAP@0.5 | 0.5404 | 0.5763 | 0.5794 | **0.6427** |
| AP50 agnostic | 0.8020 | 0.8208 | 0.8178 | **0.9046** |
| localization recall | 0.8570 | 0.8789 | 0.8773 | **0.9580** |
| classification accuracy | 0.8328 | 0.8270 | 0.8278 | 0.8433 |
| predictions | 16,173 | 30,981 | 25,409 | 65,025 |
| val views | 1,543 | 1,543 | 1,543 | 24,975 |

差值：

| 对比 | mAP50 | AP_agn | R_agn |
| --- | ---: | ---: | ---: |
| B − A last | +3.90 pp | +1.58 pp | +2.02 pp |
| B − A best | **+0.31 pp** | −0.30 pp | −0.17 pp |
| C − B | +6.33 pp | +8.68 pp | +8.07 pp |
| C − A last | +10.23 pp | +10.26 pp | +10.09 pp |
| C − A best | **+6.64 pp** | +8.38 pp | +7.90 pp |

A best 的 AP_agn 和 recall 略高于 B，但 class-aware AP 低 0.31 pp，说明定位/召回不是瓶颈，差异在逐类排序与分类。

## 类别组（mean per-class AP@0.5）

| group | A last | A best | B | C | B−A best | C−A best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0/1/2 | 0.4061 | 0.3839 | 0.4212 | 0.4020 | +3.73 pp | +1.81 pp |
| 3/4/5 | 0.3947 | 0.4285 | 0.4550 | 0.4415 | +2.65 pp | +1.30 pp |
| 10/11 | 0.4899 | **0.6399** | 0.5908 | 0.6395 | **−4.91 pp** | −0.04 pp |
| 9/12/13 | 0.4599 | 0.5002 | 0.4838 | **0.7804** | **−1.64 pp** | **+28.02 pp** |

A best 在 10/11 上追平 C，并比 B 高 4.91 pp；但 9/12/13 仍远低于 C，说明全频单图无法替代细片对窄带信号的频率细节。

## 信号物理带宽组（AP@0.5）

| group | A last | A best | B | C | B−A best | C−A best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| narrow ≤0.25 MHz | 0.4599 | 0.5002 | 0.4838 | **0.7804** | −1.64 pp | **+28.02 pp** |
| medium 0.3–2 MHz | 0.6844 | **0.8917** | 0.7474 | 0.8388 | **−14.43 pp** | **−5.29 pp** |
| wide 2.1–10 MHz | 0.3973 | 0.4611 | **0.5055** | 0.4702 | +4.44 pp | +0.91 pp |
| very wide 10.1–20 MHz | 0.4061 | 0.3839 | **0.4212** | 0.4020 | +3.73 pp | +1.81 pp |
| ultra >20 MHz | 0.3947 | 0.4285 | **0.4550** | 0.4415 | +2.65 pp | +1.30 pp |

A best 的分布很清晰：**medium 与 10/11 强，0–5 宽类和窄带弱**。它不是 C 的均匀降级版，而是不同带宽区间的取舍。

## 收敛与早停

| A epoch | 9 | 19 | 29 | 39 | 49 | 59 | 69 | 79 | 89 | 99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mAP50 | .4405 | .5546 | **.5763** | .5745 | .5611 | .5557 | .5508 | .5468 | .5436 | .5403 |
| AP_agn | .7590 | .8195 | **.8208** | .8185 | .8177 | .8135 | .8114 | .8100 | .8084 | .8019 |
| R_agn | .8443 | **.8843** | .8789 | .8746 | .8721 | .8690 | .8657 | .8641 | .8615 | .8569 |

- mAP 与 AP_agn 的 95% 水平都在 epoch 19 达到；best 在 29；之后单边下降。
- 训练 loss 从 66.5 降到 ~13，val 反而下降，是典型过拟合；last 比 best 低 **3.59 pp**。
- 按 4 次验证事件的早停（`min_delta=0.001`，约 14,920 步）应在 **ep69** 停；best 仍是 0.5763，节省约 30 epoch（~26 min）。
- 3 次事件会在 ep59 停（省 40 epoch），但 C 会在 ep44 停，早于其已知 ep49 最好结果，故不采用；5 次事件会在 ep79 才停，省得少且模型已明显退化。
- **后续实验**：启用早停，主 checkpoint 用 best EMA；所有对照臂同一规则，并报 stopped_epoch / actual_optimizer_steps / last EMA。

## 推理成本

128 个分层 val 样本、3 次重复、GPU 空闲：

| mode | views/sample | build p50 | model p50 | end-to-end p50 | end-to-end p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A coarse | 1.00 | 0.317 s | 0.0099 s | 0.327 s | 0.778 s |
| C all | 19.10 | 0.422 s | 0.0859 s | 0.509 s | 1.159 s |

- views C/A = **19.10×**
- model-only p50 C/A = **8.64×**
- end-to-end p50 C/A = **1.56×**

端到端差距远小于视图数差距，因为 0.7 ms STFT（CPU）占主导；切片主要增加 GPU 前向/NMS 成本。

## 训练成本

| item | C | A |
| --- | ---: | ---: |
| 计划/实际 epoch | 50 / 50 | 100 / 100 |
| 优化步数 | 37,250 | 37,300 |
| 训练 wall time | 2.69 h | **1.62 h** |
| 峰值 reserved | 10.1 GiB | 10.1 GiB |
| 额外数据准备 | 无 | BF16 缓存 2,050 s（11.63 GiB） |
| 含缓存的 A 总成本 | — | **约 2.19 h** |

A 的 wall time 更低，部分来自缓存与 coarse-only 验证更便宜（~77 s vs ~180 s/次），不是模型计算量更少：两者训练图像呈现数和优化步数基本一致。

## 解释与论文口径

1. **A 作为单图基线是有效的，但没有超过 B。** A best 0.5763 vs B 0.5794，差 0.31 pp；考虑单 seed，基本持平。它证明“仅全频粗图”能训练出可用的检测器，但不能证明其优于联合训练权重只跑粗图。
2. **C 的增益主要是窄带。** C−B +6.33 pp，其中 9/12/13 +29.66 pp；宽带 0/1/2、3/4/5、wide、verywide、ultra 反而 B 更高。C−A best 的 +6.64 pp 也几乎全部来自窄带（+28.02 pp），其余组 C 与 A 接近或略低。
3. **A 的 last 不适合作为论文主结果而不加说明。** 按事先约定主表用 last，但 A last 已过拟合；建议论文同时报 A best（等价于早停模型）和 A last（敏感性），并把 last 的下降作为早停必要性的证据。
4. **推理切片不是“免费”的。** views 19.1×、模型计算 8.64×；端到端只有 1.56× 是因为 STFT 主导。论文必须同时报 views/sample、model-only 和 end-to-end。
5. **训练预算公平性成立。** 595,700 张图像、37,300 vs 37,250 步、有效 batch 16；A 的 100 epoch 是单图每轮一半图像的等价预算。
6. **后续实验采用早停。** 4 次验证事件、min_delta 0.001、min_evals 3；主 checkpoint 用 best EMA。当前 A 这轮保留为无早停对照。

## 局限

- 单 seed 1729；A 的 best/last 差异较大，若论文需要置信区间，建议补一对同 seed 的 A/C 重复。
- A 的早停是**事后**用 val 曲线模拟的（ep69），实际这轮跑满了 100 epoch；后续实验才会真正早停。
- B/C 的 C 权重是历史 `best_corrected.pt`；B 是用同一权重重新跑 coarse-only，不是重新训练。
- 缓存用 BF16 位模式；训练路径核验 loss/preds/grad 全 0，最终验证用现场 float32。
- test 继续封存；所有选择只在 val 上完成。

## 勘误与待复核（2026-11-18）

- `reports/group_a_results.md` 正文曾把物理带宽 `≤0.25 MHz` 的 C−A best 写成 **+5.29 pp**。
- 原始分组结果核正：A best `0.5001996`，C `0.7804442`，**C−A = +28.0245 pp**。
- `+5.2914 pp` 是 A best 在 `medium 0.3–2 MHz` 相对 C 的优势（`0.8917338` vs `0.8388196`），不是窄带。
- 正文已改为 `+28.02 pp`，并保留 medium 的 `+5.29 pp` 表述；核验明细见 `reports/group_a_narrowband_check.json`。
- **待复核**：请 Codex / Baseline-1 抽查上述两处与原始 `signal_characteristics.json` 是否一致，并确认文中没有其他把 `+5.29` 归到窄带的表述。

## 产物

- 训练：`runs/yolo11s_c1_directreg_coarseonly_2048x512/`
  - `summary.json`、`train_log.csv`、`val_history.json`
  - `best.pt`（ep29）、`last.pt`（ep99）
  - `training_curve.png`
- A last 评测：`runs/.../stage5_val/`
- A best 评测：`runs/.../stage5_val_best/`
- 对照：`reports/group_a_results.md`、`reports/group_a_results.json`
- 延迟：`reports/group_a_latency.json`
- 训练报告：`reports/group_a_training.md`
- 曲线：`reports/group_a_training_curve.png`
- 类别组：`reports/group_a_class_groups_stage5_val*.json`
