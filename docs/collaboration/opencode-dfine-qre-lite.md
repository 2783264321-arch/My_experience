# D-FINE-L QRE-lite：query 级 C1 RoI 残差 5 轮

2026-11-26。分支 `work/dfine-qre-lite`。从第 25 轮 EMA 初始化，冻结
D-FINE base，只训练 query 级局部 C1 RoI 残差头（QRE-lite），5 轮，
完整固定 `val`。未使用 test。

## 0. 结论先行

- **QRE-lite 接口是安全的**：零初始化时输出与 e25 完全一致；8 样本 val
  精确回到 `0.842218837 / 0.836896`。
- **但 C1 RoI 没有引入新信息**：5 轮后最好的是最终模型
  `0.761823 / 0.713390`，相对 e25 **+0.0095 / −0.0538 pp**；mAP50
  几乎不变，严格 mAP 略降。
- Wi‑Fi20/40、SRRC 组全部基本不变，FM +0.08 pp，分类准确率 −0.05 pp；
  没有可保留增益。
- **判断：QRE-lite 收尾，不保留。** 它证明了 query 级残差接法可行，
  但局部 C1 特征检测器本来就能看到；真正需要的是 QRE-full 的原始
  IQ/物理证据。
- 下一步先做更便宜的 **SAHI 后处理短板实验**（Soft-NMS/WBF），再决定
  是否投入 QRE-full。

## 1. 架构

```text
C1 粗图/细片 → D-FINE-L（冻结）→ query
                                  ├─ box head（不变）
                                  ├─ base class logits（identity）
                                  └─ QRE head：C1 RoI → residual logits
                                         logits = base + residual
```

- QRE head：RoIAlign 7×7 → Conv/GroupNorm/SiLU → FC → 14 类残差；
- 最后一层零初始化；
- base backbone/encoder/decoder 全部冻结，`no_grad`；
- decoder 在训练时保留 aux 输出，但 BN/dropout 置 eval；
- 只训练 QRE head，461,198 参数；
- 框、候选身份、NMS/前 100 不变。

## 2. 完整 val

| 模型 | mAP50 | mAP50:95 | Wi‑Fi20 | Wi‑Fi40 | SRRC | FM | 分类准确率 | 最终候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e25 | **0.761728** | **0.713928** | 0.5513 | 0.6558 | 0.7809 | 0.6456 | 0.8540 | 154,165 |
| QRE ep0 | 0.761642 | 0.713513 | 0.5510 | 0.6555 | 0.7807 | 0.6457 | 0.8546 | 154,203 |
| QRE ep1 | 0.761577 | 0.713434 | 0.5509 | 0.6554 | 0.7806 | 0.6457 | 0.8541 | 154,222 |
| QRE ep2 | 0.761451 | 0.712825 | 0.5512 | 0.6546 | 0.7807 | 0.6455 | 0.8533 | 154,212 |
| QRE ep3 | 0.761728 | 0.713093 | 0.5509 | 0.6555 | 0.7809 | 0.6461 | 0.8540 | 154,202 |
| QRE final | 0.761823 | 0.713390 | 0.5511 | 0.6558 | 0.7809 | 0.6464 | 0.8536 | 154,215 |

- mAP50 最好 +0.0095 pp，严格 mAP 最好仍低于 e25 0.05 pp；
- 组级变化都在 0.01–0.02 pp 量级，没有可解释的增益；
- 最终候选数略增 50 个，定位召回在 ep2/final 回到 0.9844。

## 3. 读数

1. **零残差严格等价**：初始 QRE 输出与 e25 eval 输出一致，说明接法
   没有破坏原模型。
2. **C1 RoI 不是新证据**：D-FINE 的 query 已经通过 encoder/decoder
   看到同一 C1 特征；在局部再取一次 RoI，只是换了一个感受野，没有
   引入原始 IQ/相位/调制统计。
3. **残差头容量足够但信息不足**：461k 参数、5 轮，loss 正常，但 AP
   不动；这不是训练失败，而是输入没有新增判别信息。
4. **mAP50 微升、mAP95 略降**：分类分数有小幅重排，但高 IoU 定位/排序
   没有改善；属于噪声量级。

## 4. 判断

**QRE-lite 不保留。** 它的价值是验证接口：

- query 级残差可以安全接入 D-FINE；
- 零初始化、冻结 base、aux 输出、checkpoint 加载都正常；
- 但只加 C1 RoI 不会产生系统收益。

真正的 QRE-full 必须引入 **原始 IQ/局部物理证据**，否则只是把检测器
已有的特征再算一遍。

## 5. 下一步

先做 **SAHI 后处理短板实验**，因为它比 QRE-full 便宜，而且直接检验
高 IoU 合并损失：

- 当前类内 NMS 0.5 在 IoU 0.9/0.95 下损失约 5.9/8.6 pp 覆盖；
- 试 Soft-NMS、WBF、per-class topN，比较完整 val AP；
- 如果后处理能拿回 0.5–1 pp，优先保留；
- 如果后处理无收益，再投入 QRE-full（局部带限 IQ + 逐符号 FFT +
  SRRC 包络/HOS）。

## 6. 产物

- QRE 实现：`scripts/detector_upgrade/qre.py`
- 训练入口：`scripts/train_dfine.py --qre-lite --init-checkpoint ...`
- 验证入口：`scripts/validate_dfine.py`（自动识别 `qre.*` checkpoint）
- 结果：`reports/detector_upgrade/dfine_qre_lite_5ep.json`
- 权重/日志：`runs/dfine_qre_lite_5ep/`（未提交）
- 本轮未使用 test，未改推理合并规则。
