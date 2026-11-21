# Baseline-3 · SRRC 10↔11 YOLO 框离线复核

> **试了什么：** 在 fit 上复现冻结的 B0_unsync，并把它作用到 YOLO 自己画的 10/11 框（预测宽度 9.5–10.5 MHz、score≥0.01、置信度≥0.70），在 val 上只核验一次。  
> **净改对多少：** val 匹配集 17 个改对、14 个改坏（净 +3），另有 34 个未匹配框被改；全 14 类 mAP50 从 0.649725 降到 0.647348（−0.238 pp）。  
> **建议继续还是收尾：** **收尾**——不改推理流程；Pred-crop 分类并不比 GT-crop 差，瓶颈不是框几何，而是净收益太小且 AP 为负。

## 文件

| 文件 | 内容 |
| --- | --- |
| `independent_review.md` | 实验前独立审查、风险、预注册规则 |
| `srrc_yolo_offline_results.md` | 完整结果与判断 |
| `rule_lock.json` | fit 预测-only 锁定的候选/采用规则 |
| `reproduction.json` | B0_unsync 复现核对 |
| `fit_candidate_diagnostics.json` | fit 候选/置信度分布（无 GT） |
| `fit_fit_eval.json` / `fit_changed_candidates.csv` | fit 事后诊断 |
| `val_eval.json` / `val_changed_candidates.csv` | val 最终核验 |

## 关键数字

| 项 | val |
| --- | ---: |
| SRRC 原始预测 | 7,706 |
| 通过宽度+分数候选 | 691（9.0%） |
| 匹配到 GT 10/11 | 367 |
| 未匹配 | 324 |
| 实际改类 | 65 |
| 匹配集改对 / 改坏 | **17 / 14（净 +3）** |
| 未匹配框被改 | 34（重复 29 / 低 IoU 3 / 无 GT 2） |
| class10 AP50 | 0.7079 → 0.6881（−1.98 pp） |
| class11 AP50 | 0.5711 → 0.5576（−1.35 pp） |
| 全 14 类 mAP50 | 0.649725 → 0.647348（**−0.238 pp**） |

## 判断

- **框几何不是瓶颈**：val 同一批匹配目标，GT-crop 58.5% vs
  Pred-crop 61.7%，同实例一致率 88.0%。
- **B0 有 raw 潜力但采用规则不能转化**：val 匹配候选 raw 61.7%
  vs YOLO 52.0%，但 0.70 置信度只改 65 个；净 +3，AP 反而下降。
- **fit 不能当真值调参**：fit 是训练数据，YOLO 匹配 acc 92.0%；
  val 只有 52.0%。fit 事后 rule 净 −48（12 对/60 错），若用 fit
  真值调阈值会严重误导。
- **收尾**：不改推理；不把 B0/相位/符号恢复接回 YOLO；不扩大模型。

## 复现

```bash
# 冻结 YOLO，仅推理 fit（不重训）
PYTHONPATH=. .venv/bin/python scripts/baseline3c_yolo_pass.py \
  --split fit --out runs_local/baseline3c/yolo_fit_cache.npz

PYTHONPATH=. .venv/bin/python scripts/baseline3srrc_yolo_offline.py --stage prepare
PYTHONPATH=. .venv/bin/python scripts/baseline3srrc_yolo_offline.py --stage features
PYTHONPATH=. .venv/bin/python scripts/baseline3srrc_yolo_offline.py --stage diagnose
# 写 rule_lock.json（fit 预测-only）
PYTHONPATH=. .venv/bin/python scripts/baseline3srrc_yolo_offline.py --stage fit
PYTHONPATH=. .venv/bin/python scripts/baseline3srrc_yolo_offline.py --stage val
```

原始 IQ/权重/YOLO 缓存未入库；`*.log` 为本地日志。
