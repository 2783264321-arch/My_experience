# D-FINE-L QRE-full v1：query 级局部证据 + pairwise 残差

2026-11-26。分支 `work/dfine-qre-full`，基于 `work/spacenet-first-principles@3fa39f1`。
固定 e25 query，使用已有局部功率 + 30 维物理特征 + 原分数/metadata，
训练 query 级组内残差头，最后用 Soft-NMS 评估。未使用 test。

## 0. 结论先行

- **QRE-full v1 没有超过 Soft-NMS 基线**：A 为 `0.765636/0.723509`；
  最好 β=0.25 为 `0.765727/0.723556`（+0.009/+0.005 pp），β=0.5/1.0 反而下降。
- Wi‑Fi20 −0.01、Wi‑Fi40 +0.01、SRRC +0.06 pp，全部噪声量级；
- 候选级正类准确率为 0.431/0.462/0.696（012/345/1011），
  并不优于上一轮逐候选 C（0.438/0.467/0.727）；
- **不保留 QRE-full v1。** 局部功率 + 物理摘要 + pairwise 仍不能把局部证据
  转成系统 AP。

## 1. 设置

- 固定 e25 预测框/候选；
- 输入：局部功率 128×256、30 维物理特征、组内原分数、score、带宽、时长、view；
- 模型：ResNet18 + 特征 MLP + metadata MLP → 组类 + keep；
- 训练：20 轮，CE + pairwise（正类 query 排在 keep/负样本前）；
- 推理：只更新候选自己的类分数，one-sided 残差，keep/置信度门控 0.5；
- 合并：Soft-NMS Gaussian σ=0.5；
- 完整 val，未用 test。

## 2. 系统 AP

| 模型 | mAP50 | mAP50:95 | Wi‑Fi20 | Wi‑Fi40 | SRRC | FM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A Soft-NMS | 0.765636 | 0.723509 | 0.5551 | 0.6621 | 0.7827 | 0.6467 |
| QRE-full β=0.25 | 0.765727 | 0.723556 | 0.5550 | 0.6622 | 0.7833 | 0.6467 |
| QRE-full β=0.50 | 0.765523 | 0.723110 | 0.5549 | 0.6613 | 0.7834 | 0.6467 |
| QRE-full β=1.00 | 0.765417 | 0.722654 | 0.5543 | 0.6619 | 0.7827 | 0.6467 |

β=0.25 的 +0.009/+0.005 pp 是噪声；组级没有一项超过 0.06 pp。

## 3. 候选级训练

| 组 | 正类准确率 | keep recall | best epoch |
| --- | ---: | ---: | ---: |
| Wi‑Fi20 0–2 | 0.431 | 0.735 | 9 |
| Wi‑Fi40 3–5 | 0.462 | 0.866 | 6 |
| SRRC 10–11 | 0.696 | 0.900 | 10 |

pairwise 损失没有提高正类准确率，反而略低于上一轮 C 的
0.438/0.467/0.727；说明当前局部功率/物理摘要的判别上限就在这里。

## 4. 判断

**QRE-full v1 不保留。** 局部功率 + 30 维物理摘要已经包含包络/HOS/
CP/子载波统计，但仍不能超过 Soft-NMS 基线。继续在同一输入上调
pairwise/β 没有意义。

下一步要么：

1. 做一次**原始 IQ 1D 编码**的表示诊断，确认波形是否提供局部功率/
   摘要没有的额外信息；只有候选级准确率明显高于当前 0.43/0.46/0.70，
   才值得做 QRE-full v2；
2. 或者停止局部证据线，转 SAHI 2.0 的跨视图关联/FM 几何。

当前最确定的收益仍是 **Soft-NMS +0.96 pp 严格 mAP**，应优先保留和
工程化。

## 5. 产物

- 训练：`scripts/train_local_iq_abc.py --pairwise-weight 0.5`
- 评估：`scripts/eval_qre_full.py`
- 模型：`/root/autodl-fs/runs_local/dfine_local_iq/models_qre_full/{012,345,1011}/best.pt`
- 汇总：`reports/detector_upgrade/dfine_qre_full_v1.json`
- 本轮未使用 test，未改框回归。
