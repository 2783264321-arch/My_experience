# 检测器升级评估：D-FINE-L vs YOLO26l（适配 + RTX 4090 短跑）

2026-11-22。范围：**接入适配与短跑测量**，不开正式训练；官方 test 继续封存。
基线参照：Baseline-1 粗图＋细片，YOLO11s-C1-DirectReg + 细片 0–5 过滤，
完整验证集 **0.649725 mAP@0.5**（1,543 源 / 7,162 GT / 24,975 视图）。

## 0. 结论先行

**唯一一场正式训练建议给 YOLO26l。** 取舍依据（详见第 8 节）：

1. 原生 Ultralytics 链路（loss / 双头 / NMS / 续训）改动最小，已验证正样本
   分配、反向、保存重载和续训进度恢复；
2. 同样 8 视图物理 batch 下比 D-FINE-L 快 **2.05×**（0.189 vs 0.381 s/步），
   50 轮训练 4.0 h vs 8.2 h，给完整验证和误差分析留出余量；
3. 能稳定跑 **12 视图**（22.0 GiB），D-FINE-L 只能稳定 8 视图（17.1 GiB，
   12 视图 OOM）——YOLO26 更接近 Baseline-1 的物理 batch；
4. 200 步 sanity 中 YOLO26l 已到 mAP 0.297 / recall 0.89（D-FINE-L
   0.069 / 0.35；两者 warmup/LR 不同，不作模型优劣结论），说明链路健康。

D-FINE-L 的适配也已跑通、保留为**高上限备选**：若目标是最大化细框定位
（分布回归）或后续有第二场预算，按第 8 节的 1 epoch pilot 条件启动。

---

## 1. 版本、权重与环境

| 项 | 版本/路径 |
| --- | --- |
| D-FINE 源码 | 上游 `956d1709314c2c6a4df6f34de232054578a7449f`，`/root/autodl-tmp/third_party/D-FINE-956d170` |
| D-FINE 权重 | 官方 `dfine_l_obj2coco_e25.pth`（HF `Peterande/D-FINE`，经 `hf-mirror.com`），`/root/autodl-tmp/weights/`，121 MB |
| YOLO26 权重 | 官方 `yolo26l.pt`（HF `Ultralytics/YOLO26`），51 MB |
| 检测框架 | Ultralytics **8.4.155** |
| PyTorch | 2.8.0+cu128 |
| GPU | RTX 4090 24 GB（单卡，串行测量） |
| 数据 | 冻结 split：fit 5,957 / val 1,543 / test 2,500（封存） |

网络限制与取源：GitHub 直连不可用；源码经 `gh-proxy.com` 取归档，权重经
`hf-mirror.com`，Python 依赖走阿里云 PyPI 镜像。D-FINE 的顶层包名 `src`
与项目 `src` 冲突，已在第三方目录建别名软链 `dfine_src -> src`，适配器
以 `dfine_src.*` 导入，不修改上游源码。

## 2. 共同的数据与评估口径

- 视图：沿用 Baseline-1 的 0.7 ms STFT、`F×512`、C1 log-power 粗图与
  频率细片；模型输入实测为 `[N, 1, 2048, 512]`（2048 行频率、512 列时间）。
- 训练采样：每源 1 粗 + 1 随机细片；fit 5,957 源/轮 = **11,914 视图/轮**，
  745 个优化步（8 源 × 2 视图 = 16 视图/步）。
- 验证：1,543 源跑全部粗细视图 = **24,975 视图**。
- 合并：单片预测 → `xyxy + score + class` → 现有 `_map_view_dets` 裁框与
  全局回映 → 内部边界 16 格过滤 → 全局 class-wise NMS（IoU 0.5）→
  每样本 top-100。D-FINE 无片内 NMS（set 预测，postprocessor top-300/视图）；
  YOLO26 用 one-to-many 头 + 片内 NMS（conf 0.001 / IoU 0.5 / top-300/视图）。
- 本轮对照不启用细片 0–5 过滤（正式训练后再决定是否加）。
- 指标：strict class-aware mAP@0.5、class-agnostic AP@0.5、定位召回、
  匹配内分类准确率；正式对照再补 `mAP@0.5:0.95` 与 WiFi/SRRC 分组。

## 3. D-FINE-L 接入

### 3.1 构建与权重迁移

- 用上游 `YAMLConfig` 构建：`num_classes=14`、`remap_mscoco_category=False`、
  `eval_spatial_size=[2048,512]`、`HGNetv2.pretrained=False`（避免联网下载
  ImageNet 权重）。参数量 **31.11 M**。
- **首层**：HGNetv2-B4 的 `stem.stem1.conv` 从 `Conv2d(3,32,3,3)` 换成
  `Conv2d(1,32,3,3)`；官方 RGB 三组卷积核**逐元素求和**后复制。
- **按形状加载**：copied **1,156**，skipped **17**。跳过的是
  `decoder.anchors` / `valid_mask`（缓冲，按 2048×512 重建）、
  `enc_score_head` / `dec_score_head.*`（80→14 类）、
  `denoising_class_embed`（81→15）。分类输出与去噪类别嵌入重新初始化。
- **冻结策略**：官方配置 `freeze_stem_only=True` 会冻结 stem；替换后的
  单通道首层卷积默认可训练，本适配**保留其可训练**（理由：输入通道语义
  已变，需适配 log-power 分布；其余 stem 仍沿用预训练特征）。这一点在
  正式训练前可作为一个单变量 ablation。

### 3.2 矩形位置编码（已修）

上游 `HybridEncoder.build_2d_sincos_position_embedding(w,h)` 用
`meshgrid(arange(w), arange(h), indexing='ij')` 展开（w 外层），而特征
`flatten(2)` 是 `(h,w)` 行优先；方形输入下恰好是 x/y 互换，长矩形下会
错位且形状检查仍能通过。修法：

- `w == h`：走原实现，**逐位不变**（保持官方方形权重的约定）；
- `w != h`：按 `(h,w)` 行优先重建，使横移对应时间列、竖移对应频率行。

验证：`fixed(8,8,64)` 与上游 `orig(8,8,64)` 完全相等；矩形 `(4,3)` 在
索引 `i_h*4+i_w` 处 x 通道为 `sin/cos(i_w)`、y 通道为 `sin/cos(i_h)`。
`eval_spatial_size=[2048,512]` 在构建时重建缓存，不会被官方 640×640
的方形缓存覆盖（该缓存不是 checkpoint buffer，且训练态每次重建）。

### 3.3 损失、解码与保存

- 目标：按 `batch_idx` 拆成每视图 `{'labels': int, 'boxes': 归一化 cxcywh}`
  （D-FINE 原生约定）；`loss = sum(native loss_dict.values())`，含
  VFL、L1、GIoU、FGL、DDF 与 aux 项。
- 空视图：`num_denoising` 在 `max_gt_num==0` 时返回空 dn，loss 有限
  （实测空目标 loss 0.472）。
- 解码：`postprocessor(outputs, [[512,2048]]*B)` → 每视图 top-300，
  像素 `xyxy`；再接入现有回映/边界/全局 NMS。
- 保存/重载：`torch.save({'model': state_dict})` → 新进程重建 → `strict=True`
  加载成功 → 推理输出框均在 `[0,512]×[0,2048]` 内。

## 4. YOLO26l 接入

### 4.1 构建与权重迁移

- 新配置 `configs/yolo26l_c1.yaml`：官方 `yolo26.yaml` + `scale: l`、
  `nc: 14`、`end2end: True`、`reg_max: 1`。参数量 **26.2 M**。
- 首层 `model.0.conv` 3→1 通道，RGB 核求和；copied **1,080**，
  skipped **12**（恰好是 one2many/one2one 两组 6 个分类头，80→14 类）。
- `model.args = {box:7.5, cls:0.5, dfl:1.5, epochs:50}`；`criterion` 在模型
  搬到 GPU **之后**创建（其 stride/anchors 缓冲跟随设备）。

### 4.2 关键修正：目标格式

Ultralytics `v8DetectionLoss` 的 `batch["bboxes"]` 约定是**归一化 cxcywh**：
`preprocess` 会做 `xywh2xyxy(bboxes * [W,H,W,H])`。最初按“像素 xyxy”传入，
GT 被二次放大，正样本分配为空，box/DFL 损失恒 0、模型只学背景
（500 步后 conf=0.001 零检出）。改为直接传 `collate_views` 的归一化
cxcywh 后，正样本恢复：200 步 loss 83.0→33.5，24 样本 mAP 0.297。

### 4.3 双头、损失更新与续训

- 训练走 `E2ELoss`（one2many + one2one 双头，one2one 梯度不回传 backbone）。
  每个 epoch 调用 `criterion.update()`，按官方衰减更新 `o2m/o2o`；
  checkpoint 必须保存 `criterion.updates/o2m/o2o` 并在续训时恢复。
  实测：`updates 0→1`（o2m 0.8→0.7857），保存/重载后 `1→2`（0.7714）。
- 推理：`head.end2end=False` 选 one-to-many；`model(x)` 返回
  `(y, preds_dict)`，NMS 取 `y`（xywh 像素 + **已 sigmoid 的类别概率**，
  不要再 sigmoid），conf 0.001 / IoU 0.5 / top-300/视图。

## 5. 显存与步时（4090，BF16，真实粗细片）

### 5.1 模型单步（前向＋原生损失＋反向＋优化器）

| 模型 | 物理视图 | 源数 | 步时 | 峰值分配 | 峰值保留 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO11s（基线参照） | 16 | 8 | 0.126 s | — | 10.1 GiB | 可跑 |
| **YOLO26l** | 8 | 4 | **0.189 s** | 13.17 GiB | 14.85 GiB | 稳定 |
| **YOLO26l** | 12 | 6 | **0.277 s** | 19.96 GiB | 22.04 GiB | 稳定 |
| YOLO26l | 16 | 8 | — | — | 23.0 GiB | **OOM** |
| **D-FINE-L** | 4 | 2 | 0.267 s | 7.82 GiB | 8.95 GiB | 稳定 |
| **D-FINE-L** | 8 | 4 | **0.381 s** | 15.52 GiB | 17.14 GiB | 稳定 |
| D-FINE-L | 12 | 6 | 0.527 s* | 22.79 GiB | 23.03 GiB | **OOM**（部分步后） |
| D-FINE-L | 16 | 8 | — | — | 22.9 GiB | **OOM** |

*12 视图只完成了部分计时步后 OOM；按不可稳定运行处理。

**结论**：Baseline-1 的 16 视图物理 batch 在 24 GB 卡上两边都放不下。公平
方案是**两边都用 8 视图 × 梯度累积 2 = 等效 16 视图**（745 个优化步/轮
不变）。代价是 BatchNorm 统计基于 8 视图而非 16 视图；这是正式对照必须
记录的混入因素，必要时可冻结 BN 或增加一个 BN 单变量对照。

### 5.2 端到端训练步（真实 DataLoader：IQ 读取 + STFT + 组批 + 模型）

| 模型 | 配置 | 步时（优化步） | 每轮 | 50 轮训练 | 峰值保留 |
| --- | --- | ---: | ---: | ---: | ---: |
| **YOLO26l** | 8 视图 × 累积 2 | **0.386 s** | 287.7 s | **4.00 h** | 14.85 GiB |
| **D-FINE-L** | 8 视图 × 累积 2 | **0.790 s** | 588.8 s | **8.18 h** | 16.90 GiB |

数据等待接近 0（8 workers 预取），上表就是计算主导的墙钟。D-FINE 的
CPU 匹配/损失约 **0.17 s/优化步（21%）**；YOLO26 的损失 <1 ms。

## 6. 验证与端到端

24 个分层 val 样本（5/20/30/40/50/80 MHz 各 4–5 个，共 317 视图）：

| 模型 | 构建（单进程） | 模型 | 回映+合并 | 每视图模型 | 全量 val 外推* | 10 次 val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26l | 9.24 s | 3.38 s | 0.43 s | 10.65 ms | ~6.5 min | ~65 min |
| D-FINE-L | 6.96 s | 4.81 s | 0.32 s | 15.18 ms | ~7.9 min | ~79 min |

*按 8 workers 并行构建 STFT 折算到 24,975 视图；模型与合并按实测线性外推。
验证峰值显存：YOLO26 2.82 GiB、D-FINE 3.05 GiB。

### 6.1 短跑 sanity（200 步，仅确认链路）

| 模型 | loss 首/末 50 步 | 24 样本 mAP@0.5 | AP_agn | 召回 | 分类准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO26l（修正目标后） | 83.0 → 33.5 | **0.297** | 0.604 | 0.890 | 0.596 |
| D-FINE-L | 82.9 → 73.2 | 0.069 | 0.133 | 0.350 | 0.314 |

两者 LR/warmup 不同（本短跑统一用 AdamW 1e-4），**不能据此判定模型优劣**；
它只证明：正样本分配、原生损失、反向、保存/重载、解码与合并全部工作。
D-FINE 官方配方 warmup 更长，200 步尚在预热期。

## 7. 50 轮正式预算估算

共同口径：50 轮 × 745 优化步 = 37,250 步；每 5 轮一次全量验证（10 次）；
等效 batch 16 视图（8 视图 × 累积 2）。

| 项 | YOLO26l | D-FINE-L |
| --- | ---: | ---: |
| 训练 | 4.0 h | 8.2 h |
| 10 次全量验证 | ~1.1 h | ~1.3 h |
| 合计（含 ~10% 余量） | **~5.5 h** | **~10.5 h** |

两方案都能在一天内完成；YOLO26 的余量足够加入一次失败重跑或第二 seed。

## 8. 选型建议与决策规则

**建议：正式训练给 YOLO26l。**

理由（风险调整后收益）：
1. 链路最短、已验证项最多（原生 loss/双头/NMS/续训），单场预算的失败
   风险最低；
2. 2.05× 步时优势与 12 视图稳定上限，给全量验证、错误分析和潜在重跑
   留出时间；
3. 200 步 sanity 已显示快速学习（mAP 0.30 / recall 0.89），而 D-FINE 在
   相同步数仍处预热；
4. 分类瓶颈（WiFi/SRRC）和细框定位都能用同一套合并/评估口径直接比较，
   不需要额外的自定义组件。

**D-FINE-L 的适用条件**：如果更看重细框定位上限、愿意承担自定义链路，
先做 **1 epoch（745 步）pilot**：用官方 LR/ warmup（backbone 1.25e-5、
主 2.5e-4、wd 1.25e-4），跑完在 24–64 个 val 样本上看 mAP/召回；若
ep5 等效 val mAP ≥ 0.50 且 box loss 正常，再投入正式预算。

**正式运行的停机规则**（无论选谁）：第 5 轮全量 val 若
`mAP@0.5 < 0.50`、`localization recall < 0.85`，或 conf=0.001 下出现
整批零检出，停止并切换到另一方案；同时监控 box loss 是否非零
（YOLO26 的目标格式 bug 就是靠这个发现的）。

## 9. 复现命令

```bash
# 依赖（一次性）
/root/autodl-tmp/workspace/baseline/.venv/bin/pip install \
  faster-coco-eval calflops transformers loguru pyyaml
ln -sfn src /root/autodl-tmp/third_party/D-FINE-956d170/dfine_src

cd /root/autodl-tmp/worktrees/detector-upgrade
# 单步/显存
PYTHONPATH=. /root/autodl-tmp/workspace/baseline/.venv/bin/python \
  scripts/detector_upgrade/bench_steps.py --kind yolo26 --views 8 12
PYTHONPATH=. .../python scripts/detector_upgrade/bench_steps.py --kind dfine --views 4 8
# 端到端训练步
PYTHONPATH=. .../python scripts/detector_upgrade/bench_train_e2e.py --kind yolo26
# 验证子集
SPACENET_FFT_THREADS=2 PYTHONPATH=. .../python \
  scripts/detector_upgrade/bench_val.py --kind dfine --samples 24
# 短跑 + 保存/重载 + 小评估
PYTHONPATH=. .../python scripts/detector_upgrade/short_run.py --kind yolo26 --steps 200
PYTHONPATH=. .../python scripts/detector_upgrade/eval_subset.py --kind yolo26
```

## 10. 工件与提交

- 代码：`scripts/detector_upgrade/{adapters,bench_steps,bench_train_e2e,bench_val,short_run,eval_subset}.py`
- 配置：`configs/yolo26l_c1.yaml`
- 测量：`reports/detector_upgrade/*.json`（8/12 视图步时、端到端、验证子集、
  短跑数值）；`short_*.pt` 是 smoke 权重，仅服务器本地，不入库。
- 第三方源码与权重在 `/root/autodl-tmp/{third_party,weights}`，不入库。

## 11. 环境附录（后续会话可直接复用）

- 当前为有卡模式：RTX 4090 24 GB；无卡模式下只能做 CPU 读工件/写报告。
- 外网：GitHub 直连不可用；`gh-proxy.com` 可取 GitHub 归档/raw，
  `hf-mirror.com` 可取 HF 权重与 API，PyPI 走 `mirrors.aliyun.com`。
- 项目 venv：`/root/autodl-tmp/workspace/baseline/.venv`（工作树共用）。
- 数据：`/root/autodl-fs/SpaceNet_Dataset/advanced`；manifest 在
  `/root/autodl-tmp/workspace/spacenet/analysis/outputs/split_manifest.csv`。
- 共享工作树可能被并行会话切换分支；提交前先 `git status -sb`，必要时用
  独立 worktree（本报告即如此）避免污染对方未提交文件。
