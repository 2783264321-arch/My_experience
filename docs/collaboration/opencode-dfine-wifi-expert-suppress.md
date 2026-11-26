# D-FINE Wi-Fi 专家高分误检抑制：方法、复现与最终 test 计划

2026-11-26。分支 `work/dfine-raw-iq-diag` / 共享
`work/spacenet-first-principles`。本文件记录第一个经阈值扫描和
bootstrap 验证的后处理正收益。

## 0. 一句话

Soft-NMS 之后，对 **score≥0.3 的 Wi-Fi 最终检测**运行已有
W-Shared 3-seed OFDM Δφ 专家；专家 **conf≥0.9** 且调制阶数与
检测器不同时，**删除该检测**。val 从 **0.765636/0.723509** 提到
**0.770916/0.728623（+0.528/+0.511 pp）**。

## 1. 为什么是“删除”而不是“改类”

逐 GT 误差分解（742k 固定候选）：

- 98.4% 的 GT 在 IoU≥0.5 下已经有正确类检测；漏检仅 1.5%；
- 最高分重叠检测是错类的比例：Wi-Fi20 **50.0%**、Wi-Fi40
  **38.6%**、SRRC **32.3%**、窄带 1.4%；
- Wi-Fi/SRRC 的 FP 分数质量主要在 **wrong_class**，重复 FP 很小；
- 之前 QRE 的 per-candidate 改类/加分/抑制失败，因为修正大多是
  同一 GT 已有正确检测的重复项，破坏是高分且唯一的正确检测。

所以正确动作是：**删除高分的错类 FP**，让已有的正确类检测排上去。

## 2. 专家与权重

- 前端：CP/FFT/CFO 同步 + 带限 + `(Tu, CP)` 多假设；
- 特征：每 128 symbol 的 50×50 幅度-Δφ 直方图，最多 4 block；
- 模型：`HistCNN`（2,454,467 参数），3 类 QPSK/16QAM/64QAM；
- 权重：W-Shared 3 seeds（1729/1730/1731），本地文件，不入库。

SHA256：

| seed | SHA256 |
| ---: | --- |
| 1729 | `e9193a0fef7afb80cdc6cacf40d04fed9a76d002c4460db1efd366f6c9c69c53` |
| 1730 | `d3e54c41a2d28ad39f28e9adeab2215f8e38a0dc95155f23f87c861c17807e38` |
| 1731 | `74b322f677ff3e52d2f33996fe362f1ab76f110c2ffe2f6f770a41439777aca8` |

默认路径：`/root/autodl-fs/runs_local/dfine_wifi_expert/weights/`。

## 3. val 结果

| | mAP50 | mAP50:95 |
| --- | ---: | ---: |
| Soft-NMS 基线 | 0.765636 | 0.723509 |
| 专家抑制（conf 0.9） | **0.770916** | **0.728623** |
| Δ | **+0.528 pp** | **+0.511 pp** |

逐组：

| 组 | AP50 | AP50:95 |
| --- | ---: | ---: |
| Wi-Fi20 0–2 | 0.5551 → **0.5701 (+1.50 pp)** | 0.5201 → **0.5349 (+1.48 pp)** |
| Wi-Fi40 3–5 | 0.6621 → **0.6718 (+0.97 pp)** | 0.6396 → **0.6486 (+0.90 pp)** |
| SRRC 10–11 | 0.7827 → 0.7827 | 0.7526 → 0.7526 |
| 窄带 9/12/13 | 0.8553 → 0.8553 | 0.7596 → 0.7596 |

0–5 六类全升；类 1（Wi-Fi20 16QAM）+2.81 pp AP。

诊断：

- 3,438 个 score≥0.3 检测，sync 18.8%（646）；
- matched 630：检测器 0.375，专家 0.719；
- 专家 conf≥0.9 分歧 165 个：**163 修正 / 1 破坏，精度 99.4%**；
- 只删除 167 个检测。

阈值扫描：0.6–0.95 全正；0.6 最好 0.772100/0.729889
（+0.646/+0.638），0.9 预注册 0.770916/0.728623。0.6 是 val
选择，不作最终配置。

200 次按样本 paired bootstrap：mAP50 +0.534 pp
（95% CI [+0.381,+0.728]），mAP50:95 +0.508 pp
（[+0.359,+0.688]），P(diff≤0)=0.000。

## 4. 集成

新增：

- `src/wifi_expert_suppress.py`：`suppress(records, preds, ...)`，
  输入最终合并检测，返回过滤后的 `ImagePredictions` 和统计；
- `scripts/apply_wifi_expert_suppress.py`：固定候选缓存 →
  Soft-NMS/NMS 合并 → 抑制 → AP/保存预测。

val 复现：

```bash
cd /root/autodl-tmp/worktrees/dfine-raw-iq-diag
PYTHONPATH=.:scripts:scripts/detector_upgrade \
/root/autodl-tmp/workspace/baseline/.venv/bin/python \
scripts/apply_wifi_expert_suppress.py \
  --split val --merge soft --score-thr 0.3 --conf-thr 0.9 --workers 16 \
  --out /root/autodl-fs/runs_local/dfine_local_iq/wifi_expert_repro_val.json
```

## 5. 最终 test 计划（只跑一次）

test 只在配置锁定后确认一次，不用于调参。配置：

- e25 checkpoint：
  `/root/autodl-tmp/worktrees/formal-dfine/runs/dfine_l_c1_2048x512/best.pt`
- `batch_views=8`、`workers=8`、conf 0.001、K300、边界/细片过滤；
- Soft-NMS Gaussian σ=0.5；
- 专家 SHARED 3 seeds，score≥0.3，conf≥0.9。

命令：

```bash
# 1) test 候选缓存（与官方 test 同口径）
PYTHONPATH=.:scripts:scripts/detector_upgrade \
/root/autodl-tmp/workspace/baseline/.venv/bin/python \
scripts/build_dfine_candidate_cache.py --split test \
  --batch-views 8 --workers 8 \
  --out /root/autodl-fs/runs_local/dfine_local_iq/test_candidates.npz

# 2) NMS 基线校验（应复现官方 0.740703/0.682079）
... apply_wifi_expert_suppress.py --candidates .../test_candidates.npz \
  --split test --merge nms --skip-suppress --out .../test_nms.json

# 3) Soft-NMS test 基线
... --split test --merge soft --skip-suppress --out .../test_soft.json

# 4) 最终：Soft-NMS + 专家抑制
... --split test --merge soft --conf-thr 0.9 \
  --out .../test_soft_expert.json --save-preds .../test_soft_expert_preds.npz
```

验收：

- NMS 基线应接近官方 0.740703/0.682079；
- Soft-NMS test 应相对 NMS 为正（val +0.391/+0.958）；
- 专家抑制 test 的 mAP50 应 ≥ Soft-NMS test，且 Wi-Fi20/40 组
  不出现回退；若 test 与 val 方向不一致，只记录，不据 test 调参。

## 6. 风险与边界

- sync 率 18.8% 略低于 20% 诊断门槛；但 precision 99.4% 和 AP
  +0.53 pp 远超门槛。专家只在高分、较干净的框上工作。
- 专家训练于 fit Wi-Fi GT；val/test 未参与训练。其自有 holdout
  也是 fit，不含 val/test。
- 前端 CPU 成本：val 3,438 个检测约 132 s（16 workers）；test
  约 5–6k 检测，约 4 min。只在高分 Wi-Fi 检测上运行。
- 该后处理不改框、不改类、不重训；只删除高分错类检测。
- 窄带 class 13 的 background FP（5,204 分数质量）是独立问题，
  不在本方法范围内。
