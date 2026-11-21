# Errata · `ca7e9b0` Wi‑Fi 短段诊断的目标排除错误

2026-11-21。独立复核发现 `ca7e9b0` 的重叠统计把**全局 GT 编号**
当成**样本内下标**，导致目标本身没有被排除。本文件记录错误、
影响范围与更正后的证据；原提交不重写，原始逐框表/汇总保留为
`sync_segment_cases_v1.csv` / `sync_segment_results_v1.json`。

## 1. 根因

- `adaptation_pairs.json` 的 `gt_idx` 是跨样本的**全局** GT 编号
  （最大 7,157；val 只有 1,543 个样本）。
- `baseline3wifi_sync_diag.py` 的 `other_signal_risk(sigs, target_gt_idx, ...)`
  却把它和 `enumerate(sigs)` 的**样本内下标**比较。
- 因此绝大多数样本里 `j == target_gt_idx` 永远不成立，目标自己
  被当成“另一条信号”，`other_frac≈1.0` 是目标自身的重叠。

## 2. 更正方式

1. `local_idx = gt_idx − gt_offsets[sample_idx]`（`gt_offsets` 来自
   `yolo_val_cache.npz` 的 `gt_counts` 累积和）。
2. 用 `local_idx` 从 JSON manifest 取目标信号，核对：
   - `class` 与配对/缓存的 `gt_label` 一致；
   - `start_frequency / end_frequency / start_time / end_time` 与
     配对中的 GT 时频坐标一致；
   - 由 JSON 重建的 GT 框与缓存 `gt_boxes[gt_idx]` 一致
     （atol=2e-4）。
3. 用稳定的 `signal_id` 排除目标，而不是用全局编号或枚举下标。
4. 只对已选出的最佳短段（`best_t1/best_t2`）重算 Pred/GT 框与
   其他标注信号的重叠；不重跑同步、不重跑专家、不调阈值。

## 3. 影响（Pred 框，42 个找回样本）

| 指标 | `ca7e9b0` 旧口径 | 更正后 |
| --- | ---: | ---: |
| 高其他信号重叠（`other_frac≥0.5`） | 42/42 | **8/42** |
| 专家类别 = 其他信号类别 | 30/42 | **4/42** |
| “改对”中专家 = 其他信号 | 14/14 | **0/14** |
| “改错”中专家 = 其他信号 | 1/7 | **3/7** |
| 改对 / 改错 / 净 | 14 / 7 / +7 | 14 / 7 / +7（不变） |

GT 框同批失败（41 个找回）：高重叠 41/41 → **8/41**；
专家=其他信号 28/41 → **3/41**；改对/改错仍为 15/8（净 +7）。

## 4. 撤回与保留

**撤回：**

- “42/42 个找回段都与框内另一条标注信号高重叠”；
- “14/14 个改对等于其他信号类别”；
- 由此得出的“短段主要是在捡框内另一条信号、净 +7 不能归因于
  目标恢复”这一结论。

**保留：**

- 短段找回 **42/267（15.7%）** Pred 失败框、40,835 个符号；
- 匹配集 **14 改对 / 7 改错（净 +7）**；
- 但收益全部来自 `adaptation-train`（13/6），
  `calibration-dev` 为 **1 改对 / 1 改错（净 0）**；
- class3 对照 **0 改对 / 3 改错（净 −3）**；
- 仍有 **8/42** 高其他信号重叠、**3/7** 改错与其他信号同类；
- 整体精确二项/McNemar p=0.189，证据不足以改规则。

## 5. 更正后的判断

短段**能**让一部分整段失败的框通过同步，且多数“改对”不能归因于
其他信号；但分类收益在两个开发划分之间不一致，class3 对照为负，
高重叠风险仍存在。因此：

- 保留现行整段推理规则；
- 不接回推理、不调阈值、不重训；本轮收尾；
- 短段作为**待验证线索**；若以后有更多独立开发数据，再做
  预注册复核，而不是直接扩大或进入 final-dev/test。

## 6. 产物

- `sync_segment_cases.csv`：更正后的逐框表（主列已更正，保留
  `v1_*` 旧值）；
- `sync_segment_cases_v1.csv`：`ca7e9b0` 原始逐框表；
- `sync_segment_results.json`：更正后汇总（含 old/new 对照与 p 值）；
- `sync_segment_results_v1.json`：`ca7e9b0` 原始汇总；
- `scripts/baseline3wifi_sync_diag_recheck.py`：CPU-only 复核脚本。
