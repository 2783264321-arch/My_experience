# 勘误 · E1/E2/E3 的“冻结层”实际行为

2026-11-21。针对审阅意见第 1 条：**conv1/conv2 参数冻结 ≠ BatchNorm
running statistics 冻结**。本文件记录已提交实验（`9c27667` /
`359d232`）的实际行为，不重训、不改权重、不改 mAP。

## 1. 原表述

E1/E2/E3 报告与脚本此前写作“冻结 conv1/conv2，训练 conv3+FC”。
该表述容易被理解为整个低层卷积块（含 BN 缓冲）都保持原值。

## 2. 实际行为

`FineTuneNet` 只对 `features[0:8]` 中的 **可训练参数** 设置
`requires_grad=False`；训练循环仍调用 `model.train()`。PyTorch 在
`train()` 模式下会更新 **所有** BatchNorm 层的 `running_mean` /
`running_var` / `num_batches_tracked`，包括 BN1（`features.1`）和
BN2（`features.5`）。因此：

| 对象 | 是否变化 | 证据（9 个 run 全部一致） |
| --- | --- | --- |
| conv1/conv2 的 weight/bias | **不变** | max abs diff = 0 |
| BN1/BN2 running_mean/var | **变化** | 最大差异量级 0.006–0.76 |
| BN1/BN2 num_batches_tracked | **变化** | +800（100 step）或 +3200（400 step），即 8 instances/step |
| conv3 / FC 参数 | 变化 | 正常训练 |
| BN3 running stats | 变化 | 正常训练 |

逐 run 的 SHA256、max abs diff、`changed` 标记见
`finetune_freeze_audit.json`；生成脚本
`scripts/baseline3ci_freeze_audit.py`。

## 3. 正确表述

> 冻结 conv1/conv2 的**卷积参数**，但 BatchNorm1/2 的 running
> statistics 继续随训练更新；conv3 与 FC 正常训练。

这是一种“参数冻结 + BN 统计适配”的混合设置。BN 统计更新本身是
一种无监督域统计适配，可能是 E1/E2/E3 在 final-dev 上产生小幅
变化（包括 GT 域回退）的来源之一；本轮没有做“BN 也完全冻结”的
对照，因此不能把微调效应唯一归因于 conv3/FC 的监督更新。

## 4. 对结论的影响

- **mAP 与分类数字不变**：本勘误只修正配置描述，不改变已保存的
  权重、预测或正式 mAP。
- **解释需加限定**：E1/E2/E3 的微小差异不能被解释为“仅训练高层
  特征”的结果；BN1/BN2 统计也发生了适配。主结论（E2 相对 E0
  仅 +0.235 pp、CI 包含 0）不受影响，但机制归因应更谨慎。
- 原始 Phase B′ 的 W-Shared 权重未被覆盖；E1/E2/E3 权重仍在
  `runs_local/baseline3ca/weights/`（本地，不入库）。

## 5. 权重 SHA256（原始与微调）

原始 W-Shared：

| 文件 | SHA256 |
| --- | --- |
| `SHARED_seed1729.pt` | `e9193a0fef7afb80cdc6cacf40d04fed9a76d002c4460db1efd366f6c9c69c53` |
| `SHARED_seed1730.pt` | `d3e54c41a2d28ad39f28e9adeab2215f8e38a0dc95155f23f87c861c17807e38` |
| `SHARED_seed1731.pt` | `74b322f677ff3e52d2f33996fe362f1ab76f110c2ffe2f6f770a41439777aca8` |

微调权重（9 个）及逐张量差异见 `finetune_freeze_audit.json` 的
`runs` 字段；不在此重复长表。
