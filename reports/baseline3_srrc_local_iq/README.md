# Baseline-3 · SRRC 10/11 局部 IQ 分类（fit-only）

2026-11-21。独立实验记录。**只使用 fit；未使用 val/test；未接 YOLO；
未重训 Wi-Fi/YOLO/专家模型。**

## 文件

| 文件 | 内容 |
| --- | --- |
| `independent_review.md` | 实验前独立审查、风险、预注册判据 |
| `srrc_local_iq_results.md` | 结果、配对比较、净收益、独立判断 |
| `srrc_local_iq_results.json` | 机器可读数字（CI、混淆、分层、MLP） |
| `srrc_local_iq_predictions.csv` | holdout 逐实例预测/状态 |
| `srrc_local_iq_split.json` | 文件级 70/30 划分与 SHA256 |

## 一句话结论

**未同步局部 IQ 的幅度统计已经能到 79.3%；恢复符号幅度只多
+0.94 pp（CI 含 0）；相邻相位差只多 +1.41 pp（CI 含 0，MLP 反而
更低）。恢复成功率 36.9% 是主瓶颈。保留简单方案，不把相位差
列为带回 Wi-Fi 的线索，不扩大模型。**

## 关键数字

| 项 | 值 |
| --- | ---: |
| fit 10 MHz 实例/文件 | 1,824 / 1,572 |
| 文件级 train / holdout | 1,278 / 546 实例 |
| holdout 恢复成功 | 213（class10 119 / class11 94） |
| 全局恢复成功率 | 36.9% |
| clean / any / strong 恢复率 | 52.0% / 32.9% / 28.2% |
| B0_power（成功子集） | 58.2% |
| B0_unsync（成功子集） | 79.3% |
| R_amp（成功子集） | 80.3% |
| R_amp+phase（成功子集） | 81.7% |
| B0_unsync → R_amp | +0.94 pp，CI [−4.25,+6.16] |
| R_amp → R_amp+phase | +1.41 pp，CI [−1.44,+4.67] |
| 全实例 R_amp+fallback | 62.8%（+1.28 pp，CI [−0.72,+3.23]） |
| 全实例 R_amp+phase+fallback | 63.4%（+1.83 pp，CI [−0.18,+3.95]） |

## 判断

- **幅度**：有物理依据，class11（16QAM）明显受益；但未同步幅度
  统计已接近恢复幅度，恢复并非必要。保留简单方案。
- **相位差**：总体小幅正向但不显著，MLP 上为负，class10/11 方向
  相反；不满足可重复判据。不列为 Wi-Fi 线索。
- **瓶颈**：恢复覆盖率/交叠，而不是 clean 成功样本的特征判别
  （clean 成功子集分类 93–97%）。
- **后续**：若继续，先解决恢复/交叠/标签可靠性；本轮不扩大模型，
  不接 YOLO，不使用 val/test。

## 复现

```bash
PYTHONPATH=. .venv/bin/python scripts/baseline3srrc_local_iq.py
PYTHONPATH=. .venv/bin/python -m unittest scripts.test_srrc_frontend
```

原始 IQ/权重未入库；`run.log` 为本地日志。
