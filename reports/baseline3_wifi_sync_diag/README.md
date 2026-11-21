# Baseline-3 · Wi‑Fi 同步失败短段诊断（复核修正版）

> **试了什么：** 在 adaptation-train/calibration-dev 的 267 个 class3/4/5 Pred 框整段失败配对（及 12 个未匹配候选）上，复核现行整段 CP 检查与同一中心 20 ms 内的 2 ms 短段检查，并用冻结 W‑Shared 专家对最佳短段评分。
>
> **发现什么：** 修正 `gt_idx` 全局/样本内错位后，短段找回 42/267（15.7%）、40,835 个符号；匹配集 14 改对 / 7 改错（净 +7），Pred 高其他信号重叠从 42/42 降为 **8/42**，14 个改对中 **0 个**与其他信号同类；但收益全在 adaptation-train（13/6），calibration-dev 为 **1/1（净 0）**，class3 对照净 **−3**。
>
> **值不值得继续：** **不值得现在改规则或接回推理；本轮收尾，短段保留为待验证线索**——保留现行整段规则，不重训、不调阈值，final-dev/test 继续封存；若以后有更多独立开发数据，再做预注册复核。

## 文件

| 文件 | 内容 |
| --- | --- |
| `independent_review.md` | 实验前独立审查与预注册规则 |
| `wifi_sync_diag_results.md` | 复核修正后的完整结果与判断 |
| `errata.md` | `gt_idx` 目标排除错误的根因、旧/新对照与撤回范围 |
| `sync_segment_results.json` | 更正后汇总（含 old/new 对照与 p 值） |
| `sync_segment_cases.csv` | 更正后的 544 个 case；含 `target_signal_id`、`v1_*` 旧值 |
| `sync_segment_results_v1.json` | `ca7e9b0` 原始汇总（保留审计） |
| `sync_segment_cases_v1.csv` | `ca7e9b0` 原始逐框表（保留审计） |
| `scripts/baseline3wifi_sync_diag.py` | 原诊断脚本（已修正目标排除） |
| `scripts/baseline3wifi_sync_diag_recheck.py` | 本次 CPU 只读复核 |

## 关键数字

| 项 | Pred 框 |
| --- | ---: |
| 整段失败样本 | 267 |
| 短段找回 | **42/267（15.7%）** |
| 新符号 / blocks | 40,835 / 164 |
| 专家改对 / 改错 / 净 | **14 / 7 / +7** |
| 高其他信号重叠 | **8/42**（旧口径误报 42/42） |
| 专家类别=其他信号 | **4/42**（旧口径误报 30/42） |
| “改对”中专家=其他信号 | **0/14**（旧口径误报 14/14） |
| adaptation-train 净 | +7（13 对 / 6 错） |
| calibration-dev 净 | **0（1 对 / 1 错）** |
| class3 对照净 | **−3（0 对 / 3 错）** |
| 整体精确 p | 0.189 |

GT 框同批失败：265 个找回 41（15.5%），15 对 / 8 错（净 +7），
8/41 高重叠，3/41 专家=其他信号。

## 判断

- **上一版“短段主要捡其他信号”结论被推翻**：目标未从重叠统计中
  排除导致假高重叠；修正后 0/14 改对与其他信号同类。
- **短段确实能恢复部分目标结构**：42 个 Pred 框、40,835 符号；
  但收益只在 adaptation-train，calibration-dev 净 0，class3
  对照净 −3；整体 p=0.189，证据不足以改规则。
- **建议**：保留现行整段规则；不接回推理、不调阈值。短段作为
  待验证线索，在更多独立开发数据上做小规模预注册复核；当前
  不扩大、不进入 final-dev/test。

## 复现

```bash
# CPU-only 复核；不重训、不跑 GPU
PYTHONPATH=. .venv/bin/python scripts/baseline3wifi_sync_diag_recheck.py
```

原始 IQ/权重未入库；`run.log` 为本地日志。
