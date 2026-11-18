# 单样本推理全链路：从原始 IQ 到最终检测框

本文档描述 SpaceNet baseline（YOLO11s-C1-DirectReg，2048×512）的**推理**路径。
测试集仍未执行（保持封存）；下面的流程对 val/test 完全一致，只是 test 不提供 GT。
配套的可复核单样本 trace 见 `reports/inference_trace_train_872.md` 与
`reports/inference_trace_train_872.json`（val 样本 `train/872`，B=80 MHz，T=150 ms）。

- 项目根：`/root/autodl-tmp/workspace/baseline`
- 模型：YOLO11s 主体，C=1，14 类，`reg_max=1`，`end2end=False`（DirectReg）
- 输入：`[1,1,2048,512]` 单通道 log-power
- 推理精度：FP32；训练用 BF16 AMP

---

## 0. 输入：一个样本由什么组成

一个样本对应两个只读文件：

- `xxx.bin`：原始 IQ，`float16` 小端，交错排列 `I,Q,I,Q,...`，每个复采样 4 字节。
- `xxx.json`：元数据，包括 `B`（观测带宽，MHz）、`T`（观测时长，ms）、`N`（复采样数）、`f_lo/f_hi`，以及原始 `signals`（仅 val 有 GT 时使用；test 推理不需要）。

关系：

```text
Fs = B × 10^6              # 采样率按 B 解释
N  = B × 10^6 × T/1000     # 复采样数
```

推理时 `N` 必须和 manifest/文件大小对得上，否则直接报错，不猜格式。

---

## 1. IQ 读取

代码：`src/signals/data.py::load_iq`

```python
raw = np.fromfile(path, dtype="<f2")   # 交错 I/Q
iq  = raw[0::2] + 1j*raw[1::2]         # complex64
```

原理：

- 复基带信号 `x[n] = I[n] + jQ[n]`。
- 我们只使用幅度/功率，不用相位；但 STFT 必须在复数域做，否则正负频率会混叠。
- `N` 与文件长度校验，保证没有截断/错位。

---

## 2. STFT：把一维 IQ 变成二维时频图

代码：`src/signals/stft_fast.py::stft_power_fast`；参考实现 `data.py::stft_power`

### 2.1 参数

```text
窗长      L = round(Fs × 0.7 ms) = 700 × B      # B 以 MHz 计
窗函数    periodic Hann: w[n] = 0.5 - 0.5 cos(2πn/L)
跳步      h = ceil(N / 511)
右补零    N_ext = 511 × h
中心补零  L/2（STFT center=True, pad_mode=constant）
FFT 点数  M = L（不额外补零）
输出      恰好 512 个时间列，F = L 个频率格
```

关键点：

- **0.7 ms 窗长固定**，所以频率分辨率固定为 `1/0.7ms ≈ 1.43 kHz`，与 B 无关。
- **512 列固定**，所以跳步 `h = ceil(N/511)` 随 T 变：T 越长，`h` 越大，时间分辨率越粗。
  - B=80/T=150ms：h=23,484，列间隔 0.294 ms；
  - B=5/T=20ms：h=196，列间隔 0.039 ms。
- `N_ext=511h` 保证 `center=True` 后正好 512 帧；右边最多补 `h-1` 个零，不到一列。
- 复数 IQ 做**双边** FFT（`onesided=False`），再做 `fftshift`，频率轴从 `[-B/2, B/2)` 排成从低到高。

### 2.2 功率

```text
X[k,n] = Σ_m x[n h + m] w[m] e^{-j2πkm/L}
P[k,n] = |X[k,n]|² / (Fs × Σ_m w[m]²)
```

这是功率谱密度（PSD）量纲的线性功率。除以 `Fs·Σw²` 是为了让不同 B、不同窗长的功率可比。

### 2.3 为什么用 scipy 而不是 torch.stft

`torch.stft` 语义正确，但在长窗（L=56,000）上 CPU 较慢。快速实现用：

```python
frames = sliding_window_view(x, L)[::h]     # [512, L] 视图
fr     = frames * win                       # 复数乘实数
z      = scipy.fft.fft(fr, axis=-1, workers=2)
p      = |z|² / (Fs·Σw²)
p      = fftshift(p, axis=-1)               # [512, F] 时间优先
```

与 `torch.stft` 数值等价（相对误差 ~4e-7）。保留 `[512,F]` 时间优先，是为了后续按频率切片时不用对整张 `[F,512]` 做转置。

---

## 3. 对数功率与归一化

代码：`stft_fast.py::normalize_slice`

```text
D = 10 × log10(max(P, tiny))
z = clip((D - lo) / (hi - lo), 0, 1)
```

- `lo = -103.23661804199219`，`hi = -45.258453369140625`。
- 这两个值只在 **fit** 的 128 个分层样本上按 0.5%/99.5% 分位数拟合，然后冻结；val/test 不参与。
- 不做逐图拉伸，也不除以 255；所有视图共用同一归一化。
- 对数压缩动态范围；`clip` 到 `[0,1]` 后作为单通道图像输入 CNN。

---

## 4. 视图构造：fine 切片 + coarse 全景

代码：`data.py::build_sample_views`、`frequency_starts`、`aggregate_frequency_fast`

原生功率图 `P` 的形状是 `[F, 512]`，其中 `F=L=700B`。直接把它缩到 2048 会损失窄带信号，所以我们同时构造两类视图：

### 4.1 Fine 切片

```text
高度 2048 个原生频率格
宽度 512 列
步长 1639 格（重叠 409 格 = 20%）
起点 0, 1639, 3278, ...；最后一片回贴到 F-2048
```

- 每个 fine 视图覆盖 **2048 × 1.43 kHz = 2.93 MHz**，与 B 无关。
- B 越大，片数越多：B=5→2 片，20→9，30→13，40→17，50→22，80→34。
- 20% 重叠（409 格）保证边界附近的信号在相邻片里是完整的；后面 16 格的边界候选规则远小于这个重叠。
- 每片单独做 log + 归一化，得到 `[1,2048,512]`。

### 4.2 Coarse 全景

把 `F` 个原生频率格按区间**面积加权平均**压到 2048 行：

```text
bounds = linspace(0, F, 2049).round()
coarse[i] = mean(P[bounds[i]:bounds[i+1]])
```

- 不是双线性 resize，而是分段平均，保留每个频段的功率。
- 频率分辨率变成 `B/2048`：B=5 时 2.44 kHz/行，B=80 时 39.06 kHz/行。
- 然后 log + 同一归一化，得到 `[1,2048,512]`。

### 4.3 为什么 fine + coarse

- **窄带信号**（class 9/12/13，6 kHz–250 kHz）只有在 fine 视图里才有足够频率像素；coarse 会把它们压到 1 行以内。
- **宽带/满带信号**（10/20/40 MHz）会跨很多 fine 片，每片只能看到片段；只有 coarse 全景包含完整框。
- 这相当于 SAHI 的“切片 + 全局上下文”思想：fine 是沿频率的切片，coarse 是全景；但我们的输入是科学张量，不用 SAHI 的 RGB/OpenCV predictor，切片、坐标回映和合并都是自己实现的。

---

## 5. 模型前向：YOLO11s-C1-DirectReg

代码：`src/signals/model.py`、`configs/yolo11s_c1_directreg.yaml`

### 5.1 输入/输出

```text
输入   [1, 1, 2048, 512]         # 单通道 log-power
输出   [1, 18, 21504]            # eval 模式
       4 个框距离 + 14 个类别 logits
```

`21504 = P3 256×64 + P4 128×32 + P5 64×16`，stride 分别为 8/16/32，anchor-free。

### 5.2 C1

IQ 是复数，但第一版只用功率（幅度²），所以 C=1。预训练 YOLO11s 是 RGB 三通道，加载时把首层 3 个输入核**按通道求和**成 1 个核；其余主体按名称/形状迁移。推理时只加载训练好的 C1 权重。

### 5.3 DirectReg：为什么不用默认 DFL

默认 YOLO 头 `reg_max=16`，最大 stride 32，单边最大距离约 `15×32=480` 输入像素。我们的框可以接近 **2048 高**（满带信号），默认头表示不了。

DirectReg 设 `reg_max=1`：

```text
每个 anchor 直接输出 4 个距离 (l,t,r,b)
x1 = anchor_x - l,  y1 = anchor_y - t
x2 = anchor_x + r,  y2 = anchor_y + b
```

距离没有 16 档上限；训练用 CIoU + 分类 BCE + 归一化 L1（第三项不是 DFL）。这是本 baseline 叫 DirectReg 而不是原版 YOLO11s 的原因。

### 5.4 推理精度

推理用 FP32，不开 autocast；训练才用 BF16（FP16 在该 21504-anchor 头上会溢出，已在阶段 2 拒绝）。

---

## 6. 每个视图的 NMS

代码：`src/signals/infer.py::_nms_view_batch`

```python
non_max_suppression(
    preds,
    conf_thres=0.001,
    iou_thres=0.5,
    nc=14,
    max_det=300,          # 每视图
    multi_label=False,
    max_time_img=20.0,
    max_nms=30000,
)
```

原理：

- 模型输出每个 anchor 的 14 类分数；`multi_label=False` 取最大类，得到一个框一个类。
- 先按 `conf=0.001` 过滤。AP 是阈值无关的，保留低分候选才能画完整 PR 曲线；部署时可以调高。
- 类内 NMS 去掉同一个类里高度重叠的重复框。
- 每视图最多 300 个，最终每样本还要再合并到 100 个。

---

## 7. 坐标回映：视图像素 → 全局归一化

代码：`infer.py::_map_view_dets`

这一步是整条链路最容易出错的地方。NMS 输出的框是**视图像素坐标** `[0,512]×[0,2048]`，但模型可能预测到视图外，所以先做**裁剪**（这是我们在 stage 5 修过的 bug）：

```python
x1,x2 = clip(x1,x2, 0, 512)
y1,y2 = clip(y1,y2, 0, 2048)
```

然后：

- 时间（fine/coarse 一样）：
  ```text
  x_global = x_view / 512
  ```
- Fine 频率：
  ```text
  y_global = (y_view + start) / F
  ```
  `start` 是这片在原生频率轴上的起点，`F` 是总原生频率格数。因为 fine 的 2048 行就是 2048 个原生格，不需要再乘 `2048/F`。
- Coarse 频率：
  ```text
  y_global = y_view / 2048
  ```

最终坐标是 `[0,1]×[0,1]` 的全局归一化 `xyxy`。物理量：

```text
t_ms = ((x*512 - 0.5) × h / Fs) × 1000
f_MHz = f_lo + y × B
```

`0.5` 是像素中心约定：`t=0` 对应第 0 列中心 0.5；训练标签用同一约定，所以可逆。

---

## 8. 内部边界候选过滤

代码：`infer.py::_boundary_mask`

对每个 fine 视图，如果框距离**内部频率切片边界** ≤16 个原生格：

```text
near_left  = (start > 0) and (y1_global*F <= start + 16)
near_right = (start+2048 < F) and (y2_global*F >= start+2048 - 16)
```

就把这个框标成 boundary candidate，**不作为完整事件输出**。

- 原因：一个被切片截断的目标不是完整事件；相邻片有 409 格重叠，应该在另一片里完整出现。
- 这条规则**不用 GT**，只用于推理；训练标签不删。
- **全局频带边缘**（start=0 或 start+2048=F）不算内部边界，边缘框保留；那是真实物理边界，不是切片伪影。
- Coarse 视图不适用这条规则。

在修正后的 val 上，这条规则扣留了 202,142 个 fine 框；这看起来很多，但相邻片和 coarse 仍然提供完整检测。

---

## 9. 跨视图合并：class-wise 全局 NMS

代码：`infer.py::global_classwise_nms`

把一个样本的 **全部 fine 存活框 + coarse 框**放在一起，按类分别做 NMS：

```text
for each class c:
    keep_c = torchvision.ops.nms(boxes[c], scores[c], iou=0.5)
按 score 排序，最终最多保留 100 个
```

为什么按类而不是 class-agnostic：

- 任务允许同频同时的不同类信号共存；class-agnostic NMS 会把不同类的真框互相压掉。
- 代价：同一个目标被分到多个类时，这些框会同时保留。val 上统计到 3,800 组“同位置多类”框；这是 class-wise NMS 的已知限制。
- 最终每样本最多 100 个框；评估协议也是 100。

---

## 10. 最终输出

代码：`infer.py::predict_sample`

返回：

```python
Prediction(
    boxes,    # [N,4] 全局归一化 xyxy
    scores,   # [N]
    labels,   # [N] 0..13
)
stats = {
    "n_views", "n_raw_dets", "n_after_boundary",
    "n_boundary_dropped", "n_final"
}
```

如果要做交付，把它转成物理坐标：

```json
{
  "sample_id": "test/xxx",
  "detections": [
    {"t1_ms": ..., "t2_ms": ..., "f1_mhz": ..., "f2_mhz": ...,
     "score": ..., "class": ...}
  ]
}
```

最后把物理框裁回原始 `[0,T]×[f_lo,f_hi]`。类别目前是数字 0–13；官方类别名未提供。

---

## 11. 一个具体数字例子：B=80 MHz，T=150 ms

```text
N   = 12,000,000
L   = 56,000
h   = ceil(12,000,000/511) = 23,484
N_ext = 511×23,484 = 12,000,324   # 右补 324 个零
F   = 56,000                       # 频率格数
Δf  = 1/0.7ms = 1.4286 kHz
Δt  = h/Fs = 23,484/80e6 = 0.2936 ms
512 列覆盖 ~150 ms
fine 片数 = 34（起点 0,1639,...,53952）
每片 2048×1.4286kHz = 2.9286 MHz
步长 1639×1.4286kHz = 2.3414 MHz
重叠 409×1.4286kHz = 0.5843 MHz（20%）
coarse 每行 = 56,000/2048 = 27.34 原生格 = 39.06 kHz
模型 anchor = 21,504
最终 ≤100 框
```

对比 B=5 MHz，T=20 ms：

```text
N=100,000; L=3,500; h=196; F=3,500
Δt=0.0392 ms；fine 片数=2（起点 0,1452）
coarse 每行 = 1.709 格 = 2.44 kHz
```

这组数字直接解释了前面的性能：
- 窄带信号在 fine 里频率格多，但 6 kHz 在 80 MHz 下也只有约 4 个原生格；
- 10/20/40 MHz 信号跨 3–28 个 fine 片，只有 coarse 有完整框；
- 1 ms 信号在 150 ms 下只有约 3.4 列（按 h 计），时间像素极少。

---

## 12. 用到的技术与原理小结

| 环节 | 技术/原理 | 为什么 |
| --- | --- | --- |
| IQ 读取 | 复数基带、float16 交错、字节序 | 原始数据格式固定，N 必须校验 |
| STFT | 短时傅里叶、Hann 窗、补零、双边谱、fftshift | 时频局部化；复数 IQ 正负频率独立 |
| 0.7 ms 窗 | 频率分辨率 1.43 kHz | 兼顾窄带 6 kHz 与瞬态时间窗 |
| 512 列 | 固定 CNN 宽度；h=ceil(N/511) | 不同 T 统一输入尺寸；时间分辨率随 T 变 |
| 功率/对数 | PSD 归一化、log 压缩、冻结归一化 | 动态范围大；避免逐图拉伸和 val 泄漏 |
| 视图 | 多分辨率、20% 重叠、fine+coarse | 窄带看细节，宽带看全景；SAHI 式切片 |
| 模型 | YOLO11s、FPN P3/P4/P5、anchor-free | 一阶段检测；多尺度 |
| C1 | RGB 核求和、功率单通道 | 保留预训练主体；第一版不用相位 |
| DirectReg | reg_max=1，直接距离 + L1/CIoU | 满高框超出 DFL16 的 480 px 上限 |
| NMS | 每视图类内 NMS + 全局类内 NMS | 去重；保留同频不同类 |
| 边界规则 | 16 格内部边界候选扣留 | 截断片段不是完整事件；重叠 409 格保证相邻片完整 |
| 坐标 | 全局归一化、像素中心、物理反变换 | 多视图可合并；标签/预测可逆 |
| 评估 | 类内 greedy one-to-one、101 点 AP | 严格 class-aware；不静默去重 |

---

## 13. 代码地图

| 步骤 | 文件/函数 |
| --- | --- |
| IQ 读取 | `src/signals/data.py::load_iq` |
| STFT（参考） | `data.py::stft_power` |
| STFT（快速） | `src/signals/stft_fast.py::stft_power_fast` |
| 归一化 | `stft_fast.py::normalize_slice` |
| 视图/标签 | `data.py::build_sample_views`、`frequency_starts`、`_label_boxes` |
| Coarse 聚合 | `stft_fast.py::aggregate_frequency_fast` |
| 单视图 NMS | `src/signals/infer.py::_nms_view_batch` |
| 坐标回映/裁剪 | `infer.py::_map_view_dets` |
| 边界规则 | `infer.py::_boundary_mask` |
| 全局合并 | `infer.py::global_classwise_nms` |
| 单样本入口 | `infer.py::predict_sample` |
| 全 val | `src/signals/validate.py::run_validation` |
| 评估 | `src/signals/metrics.py::evaluate` |
| 模型 | `src/signals/model.py::build_model` |
| 配置 | `configs/yolo11s_c1_directreg.yaml` |
| 单样本 trace | `scripts/trace_sample.py` |

---

## 14. 实战中最容易踩的坑

1. **N 不一致**：文件长度/JSON/manifest 必须一致，否则不训练、不推理。
2. **视图外框**：模型会预测到 2048 高之外；必须先裁到 `[0,512]×[0,2048]` 再回映。我们 stage 5 就是因为漏了这一步，训练期 val 被低估到 0.54；补上后是 0.64。
3. **边界规则与重叠**：16 格边界规则必须远小于 409 格重叠，否则会误删完整目标；全局物理边缘不能按内部边界处理。
4. **class-wise NMS**：同位置多类框会保留，这是任务要求（同频不同类）带来的，不是 bug；报告里要如实说明。
5. **归一化冻结**：不能用 val/test 重新拟合 lo/hi；不能逐图 stretch。
6. **test 封存**：test 只跑推理、不调参、不选 checkpoint。

---

## 15. 可复核的单样本 trace

`scripts/trace_sample.py` 对 `val` 样本 `train/872`（B=80 MHz，T=150 ms，8 个 GT，
35 个视图）逐步打印并保存：

- N、L、h、N_ext、F、Δf、Δt；
- 每个视图的类型、起点、形状、归一化统计、GT 数；
- 每个视图 NMS 的原始/边界扣留/存活数量；
- 全局合并前后的数量、分数分布；
- 最终物理框（ms/MHz）与 GT 对照；
- 该样本的 per-class AP/定位召回/混淆。

输出：

- `reports/inference_trace_train_872.json`（完整数值）
- `reports/inference_trace_train_872.md`（可读表格）
- 运行日志 `/tmp/trace_train_872.log`

复现命令：

```bash
cd /root/autodl-tmp/workspace/baseline
PYTHONPATH=. .venv/bin/python -u scripts/trace_sample.py \
  --sample train/872 --split val \
  --checkpoint runs/yolo11s_c1_directreg_2048x512/best_corrected.pt
```
