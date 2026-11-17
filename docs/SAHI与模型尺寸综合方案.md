# STFT、频率切片与检测网络张量适配报告

**2026年11月18日执行更新：**统一2048×512主线已进入计划阶段，当前执行口径以[Baseline执行计划](Baseline执行计划.md)为准。落实训练时补充发现默认DFL框跨度限制，因此执行基线保留YOLO11s主体、采用C1与直接回归头（reg_max=1）；此前默认头的前向测试仅证明输入可运行。缓存已按新512列重算：全量complex64约992.88GiB，native log16约248.22GiB，默认按需计算而非全量落盘。

本报告回答一个具体问题：**如何将复数IQ的原生STFT，通过仅沿频率轴切片，送入成熟检测器，并明确它实际收到的通道数和矩形尺寸。**沿用0.7 ms单窗作为工程起点；区分无重叠与50%重叠，不启动正式训练，不修改或去重原始标签。

## 当前更新：以2048×512统一输入反推前端

用户进一步倾向统一空间尺寸。本轮因此将**2048×512作为当前主要候选**，此前固定物理hop＋分桶保留为比较方案；不据此宣称512已是精度/速度最优，也不开始正式训练。下面先给新方案，再保留原研究与证据。

### A. 分桶或补边是否有主流先例

有。需要适配数据入口，不代表需要发明一种新训练方法：

- **DETR，ECCV 2020**：会议正式版第7页§3.2脚注3明确说明，将图片补到batch内最大H/W；官方实现还创建padding mask。[论文原文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf#page=7)、[官方NestedTensor实现](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/util/misc.py#L307-L329)。
- **Faster R-CNN作者实现**：ASPECT_GROUPING按相近长宽比组批，注释指出可减少补边计算。这是作者代码中的工程证据，不能冒充论文正文。[作者配置](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/fast_rcnn/config.py#L93-L97)、[实际分组](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/roi_data_layer/layer.py#L23-L43)。
- **YOLO9000，CVPR 2017**：作者使用每10个batch改变输入分辨率的多尺度训练。这支持跨batch变尺寸，不是本项目矩形/RF策略的直接验证。[作者论文第4页](https://data.pjreddie.com/files/papers/YOLO9000.pdf#page=4)。

这些先例证明组批方式成熟，不证明我们的0.7 ms、2048×512或信号检测性能已经有论文保证。DETR的mask机制也不会自动出现在YOLO里。既然当前优先统一尺寸，可以采用统一batch，不必继续以分桶为主线。

### B. 为什么之前是448，现在用512是否合适

448来自上一方案最长T=150 ms、hop约0.35 ms时的429列：ceil(429/32)×32=448。它是减少空白的结果，不是YOLO的特殊原生宽度。512同样满足常规P5网络的32倍数对齐；无需强求2的幂，但可以采用。

把448改为512，单片空间面积增加14.29%。更大的变化来自前端：所有短时样本也要计算512列，不能仅按14.29%估算相对此前分桶方案的成本。

### C. 用常规整数hop精确得到512列

窗长L=Fs×0.7 ms、FFT点数M=L保持不变；本数据L均为偶整数。使用PyTorch的center=True、pad_mode='constant'：

\[
K=1+\lfloor N_{ext}/h\rfloor.
\]

最小实施方案为：

\[
h=\lceil N/511\rceil,\qquad N_{ext}=511h.
\]

在原IQ末尾补N_ext−N个零，再运行上述STFT，必得K=512。额外补入0–510个采样点；当前10,000个样本实际最大为458点，最大额外时长为31.2 µs（5 MHz×20 ms补156点）。常规center的左右窗口边界补零仍存在，因此应称“避免大面积谱图padding”，不能称“完全无补零”。

只调整整数hop而完全不补IQ，未必能精确512。例如N=100000时，h=195会得513列，h=196只得511列；少量IQ末尾补零即可解决，不必实现自定义非均匀STFT。标签的真实T、原始有效N均不改变，另外记录h、N_ext和帧时刻；时间坐标按实际h/Fs还原，不机械套用t/T×512。

| 原样本 | L=M | h | IQ额外补点 | 原生STFT |
| --- | ---: | ---: | ---: | --- |
| 5 MHz、20 ms | 3500 | 196 | 156 | 3500×512 |
| 20 MHz、40 ms | 14000 | 1566 | 226 | 14000×512 |
| 80 MHz、150 ms | 56000 | 23484 | 324 | 56000×512 |

以上三组已经实际运行合成IQ验证，最后原始样本也被最后帧观察到；全量10,000样本通过公式检查。没有读取原始数据集IQ。[STFT帧数规则](https://docs.pytorch.org/docs/2.14/generated/torch.stft.html)

时间步长随T变化，约为0.039–0.294 ms；0.7 ms窗的重叠率约从94.4%降到58.1%。这是512个不同窗位置的FFT观测，可能改善稀疏谱图表征，但相邻列高度相关；不是512个独立时间分辨单元，也不改变0.7 ms窗决定的谱模糊。

### D. 频率轴不需要为凑整而做FFT补零

当前所有F=L均≥3500，大于切片高度2048。因此每片都能取满2048行真实谱数据：中间正常重叠滑动，最后一片向前贴齐F−2048。F无需被2048整除，也无需被切片步长整除。

20%重叠采用int(2048×0.2)=409行，滑动步长1639；末片允许有更大的重叠。时间轴已有512列，不再切片或补谱图边：

| B MHz | STFT | 频率片数 | 每片 | 末片起始行 |
| ---: | --- | ---: | --- | ---: |
| 5 | 3500×512 | 2 | 2048×512 | 1452 |
| 20 | 14000×512 | 9 | 2048×512 | 11952 |
| 30 | 21000×512 | 13 | 2048×512 | 18952 |
| 40 | 28000×512 | 17 | 2048×512 | 25952 |
| 50 | 35000×512 | 22 | 2048×512 | 32952 |
| 80 | 56000×512 | 34 | 2048×512 | 53952 |

例如F=3500时，两片为[0,2048)与[1452,3500)，都取满真实行数。这里频率图像补边为0；重叠复用上下文，并不生成新的独立频率信息。

每窗补零、将FFT长度M增到大于L，是另一种操作：它会得到更密的频谱采样，不是频谱里新增一片全零行，可帮助峰值采样/幅度估计；但不会提高固定0.7 ms窗的真实频率分辨能力。它还会改变每2048格覆盖的物理频宽和片数，不能为了凑整就默认采用。第一版保持M=L即可。[MathWorks零填充与频率分辨率说明](https://www.mathworks.com/help/signal/ug/amplitude-estimation-and-zero-padding.html)

### E. 统一张量在YOLO11s内部怎样流转

本机PyTorch2.14 CPU、Ultralytics8.4.155，随机权重、14类模型，C=1和C=2均实际前向通过：

| 阶段 | 实际张量形状，n=1示例 |
| --- | --- |
| 输入 | [1,C,2048,512] |
| 第一个卷积 | [1,32,1024,256] |
| 检测P3，stride8 | [1,128,256,64] |
| 检测P4，stride16 | [1,256,128,32] |
| 检测P5，stride32 | [1,512,64,16] |
| 框与类别解码 | [1,18,21504] |

网络还包含多尺度融合，完整24层记录见[核查JSON](张量链路核查.json)。21504=256×64+128×32+64×16，是候选点数；18=4个框参数+14类分数，不是最终检测出21504个信号。输出仍需筛选、NMS及全图坐标还原。没有先把输入缩成640方形。

最终统一batch为[n,1,2048,512]或[n,2,2048,512]；一次实验固定C，batch大小n须由4090显存实测决定。当前验证的是核心前向，不是完整高层训练适配。

### F. 当前统一方案及代价

```text
IQ[N]
 → 根据N计算h并做极少量IQ末尾补零
 → 0.7 ms窗、M=L、center=True的STFT[F,512]
 → 单通道log-power，或双通道Re/Im
 → 仅沿频率滑窗，每片2048×512
 → 同形片组成[n,C,2048,512]
 → 同一个YOLO11s
 → 原始频率/时间坐标与类别
```

全频粗功率视图仍保留，聚合频率到2048行、时间512列；单通道完整任务可与细片共用同一模型。仅统一尺寸不能解决宽带目标跨片问题，不直接删标签或假定能自动恢复完整框。

统一512列后，fine分支每个样本平均总输入面积为train 15.8721 MP、test 19.7531 MP，不计额外全频粗图；相较此前固定0.35 ms hop并按32补齐的矩形分桶，约为2.54倍/2.21倍。这是输入面积代理，不是GPU时延比例。换来的是同形batch、减少图像空白和短时样本更密的时间取样；真实收益仍需验证集和4090成本测量。

## 1. 前一版方案：保留为对照


前一版采用以下变长空间接口，保留作对照；当前主要候选以上面的统一512列方案为准：

```text
复数IQ[N]
  → 0.7 ms单窗、双边STFT，fftshift
  → complex64[F,K]
  → float32[C,F,K]，C可明确选择1或2
  → 仅沿频率轴取Sf×K，时间轴保持完整
  → 只补边，不resize有效内容
  → float32[n,C,Sf,32·ceil(K/32)]
  → YOLO11s核心网络（首层C通道、类别数14）
  → 去除padding影响、恢复频率偏移与物理坐标
```

- **空间起点建议Sf=2048**，1024为小切片参照，4096作为后续减少调用次数的候选。这是依据当前标签带宽与调用量的工程选择，尚非实测最优。
- **输入表示分两项做对照**：A0为单通道log-power，A1为双通道Re/Im。先用A0建立易解释的检测基线，再比较A1表示的效果；C的选择不应改变空间协议。两种表示同时改变数值变换、通道数与相位信息，精度差异不能全部归因于相位。双通道并不需要把整个网络改成复数神经网络。
- **hop单独选择**：无重叠规则用于核对用户表和低成本尺寸测试；首个完整观测输入建议比较并优先验证50%重叠、center=True、零填充。它仍是单窗，但每片时间宽度约翻倍；若坚持无重叠，须另行明确末窗填充。
- **模型先用YOLO11s**；RT-DETRv2-S作另一种检测范式的对照，RTMDet-s保留为后续方案候选。2通道和矩形输入本身不构成换成ViT的理由。
- **仅局部细片不能作为完整任务的全部输入**：2048格约2.93 MHz，仍有27.29%的train实例自身宽于切片。正式全类别baseline至少还需保留全频粗上下文；本报告给出最简起点，不展开跨块恢复算法。

本轮已经用本机现有环境完成合成IQ与随机权重模型的尺寸验证：**16组STFT形状检查、12次双通道YOLO11s前向均通过**。包括`[1,2,2048,32]`、`[1,2,2048,448]`和`[2,2,2048,32]`。这证明所测核心张量链可运行，尚不证明默认训练器已适配、真实数据有效或4090速度达标。

## 2. 证据范围与版本

| 证据 | 本轮实际完成 | 不能由此推出 |
| --- | --- | --- |
| 本地样本/实例主表 | 10,000样本、54,773实例，重新计算几何与切片统计 | 没有读取原始服务器IQ |
| 用户提供的尺寸图 | 按valid帧公式逐项核对 | 图中36个理论组合不等于数据实际具有36个组合 |
| 官方源码与文档 | 核查矩形、通道、patch、预处理与坐标路径 | 服务器安装版本自动具有相同能力 |
| 合成IQ + 随机权重YOLO11s | CPU实际STFT及前向，优化器步数0 | 预训练迁移效果、检测精度、训练显存与GPU延迟 |

本机实测为PyTorch 2.14.0+cpu、Ultralytics 8.4.155；源码审查主要固定Ultralytics 8.4.215的commit `4d254d8`、SAHI commit `10061af`、官方RT-DETR commit `df2a96c8`、MMDetection 3.3.0。没有安装或升级环境，没有下载预训练权重。服务器版本应另行记录。

GPU资源仍依据用户转述OpenCode实查：RTX4090、24564 MiB，约24 GB；本轮没有登录服务器。

## 3. 先核对用户的0.7 ms尺寸表

设Fs=B、采样点数N、窗点数L=Fsτ、τ=0.7 ms、FFT点数M=L、hop点数h。用L和h避免与图像H/W混淆。complex IQ采用双边谱，所以频率行数F=M，而非M/2+1。

无居中填充、只取完整窗口时：

\[
K_{valid}=1+\lfloor(N-L)/h\rfloor.
\]

当h=L时，K=floor(N/L)，正是用户图中的列数。频率行数如下：

| B=Fs MHz | 5 | 20 | 30 | 40 | 50 | 80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F=L | 3500 | 14000 | 21000 | 28000 | 35000 | 56000 |

| T ms | 无重叠valid，h=L | 50%重叠valid，h=L/2 | 无重叠center，h=L | 50%重叠center，h=L/2 |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 28 | 56 | 29 | 58 |
| 40 | 57 | 113 | 58 | 115 |
| 60 | 85 | 170 | 86 | 172 |
| 80 | 114 | 227 | 115 | 229 |
| 100 | 142 | 284 | 143 | 286 |
| 150 | 214 | 427 | 215 | 429 |

center列明确指本报告PyTorch设置`center=True,pad_mode='constant'`，对应K=1+floor(N/h)。本轮合成IQ实测与这些公式一致。[PyTorch STFT说明][P1]

**图中数字正确，但有两个边界。**

1. valid无重叠时，上述六种T尾部各有0.4/0.1/0.5/0.2/0.6/0.2 ms不足一窗而未进入STFT；不能把最后一列默认为精确延伸到原始T。无重叠Hann还会对每窗边缘赋予很小权重；若后续需要时域重建，无重叠Hann也不能直接当作满足重建覆盖条件的方案。
2. 用户表是理论6×6组合；实际数据是31组，5 MHz仅有20 ms，其余5个5 MHz单元是数学上的尺寸，并非已经观测到的数据。

**建议**：将图中的valid规则作为复核基准；完整观测输入优先验证`center=True、h=L/2、零填充`。仅设置center=True在h=L时仍不保证末尾被观察到：20/60/100 ms例子仍遗漏约0.05/0.15/0.25 ms，需要额外末窗padding。本轮末样本脉冲测试确认了这种差别，不能把center本身当作完整边界保证。

当前数据中，上表center与valid的差异不会改变按32补齐后的网络时间宽度：已经对10,000个样本、两种hop核对，变化数为0。若另加末窗，仍须按实际K重新取整。边界补零用于保存边界观察机会，不会提高真实时频分辨率。

关于“时间格点数大家一样”：**同T、不同Fs**时，应让L与h都按固定物理时间缩放，即L/Fs=τ、h/Fs=δ，这样K一致。不能跨采样率复用固定整数h。**不同T**则保持相同δ、允许K不同，再分桶或补边；若强行让所有T都等列数，必须改变δ或重采样，这是另一套表示，不能称为完全保留同一时间格点。

## 4. 三种尺寸的关系与最终张量

采用图像高度为频率、宽度为时间的统一约定：

\[
\text{原生谱}\ [F,K]\ \rightarrow\ \text{片}\ [\min(F,S_f),K]
\ \rightarrow\ \text{batch}\ [n,C,\bar S_f,\bar K],
\]

其中\(\bar S_f=32\lceil\min(F,S_f)/32\rceil\)、\(\bar K=32\lceil K/32\rceil\)。只在右/下补边时，有效内容不平移；如果对称补边，必须同时记录偏移。不同bucket可以有不同K，同一稠密batch的最终H/W必须一致，**不要求H=W**。

以Sf=2048为起点，时间轴不切片：

| T ms | 无重叠实际tensor，C=2 | 50%重叠实际tensor，C=2 |
| ---: | --- | --- |
| 20 | `[n,2,2048,32]` | `[n,2,2048,64]` |
| 40 | `[n,2,2048,64]` | `[n,2,2048,128]` |
| 60 | `[n,2,2048,96]` | `[n,2,2048,192]` |
| 80 | `[n,2,2048,128]` | `[n,2,2048,256]` |
| 100 | `[n,2,2048,160]` | `[n,2,2048,288]` |
| 150 | `[n,2,2048,224]` | `[n,2,2048,448]` |

单通道实验只将C=2改成1。Sf也可改为1024/4096；这些都是网络配置候选，不是架构唯一“原生尺寸”。相同T跨B的时间列数一致；本方案各片频率跨度也一致，因为Δf=Fs/M=1/τ≈1.4286 kHz。

例如80 MHz×150 ms：图中无重叠原生矩阵为56000×214；2048频率切片、20%频率重叠会得到34片；每片2048×214补为2048×224，最终直接进入`[n,2,2048,224]`。该例采用无重叠center边界时，原生为56000×215，网络仍是2048×224；采用建议验证的50%重叠center方案则为56000×429，网络为2048×448。没有1024→640之类的隐藏缩小，也没有强制方形。

## 5. SAHI在这里承担什么，哪些接口需要改

SAHI是切片、检测、坐标复原与预测合并的外围流程，不规定检测器必须使用方形输入。[SAHI原论文][S1]

若频率为图像高度，实施上应显式指定`slice_height=Sf,slice_width=K,overlap_height_ratio=0.2,overlap_width_ratio=0`；关闭自动猜测切片尺寸。时间轴不切片，所有频率片保留相同K。频率滑动步长为D=Sf-floor(0.2Sf)，末片贴紧远边界；**它与STFT的时间hop h是两个独立参数**。

SAHI几何机制可以复用；浮点复数数据应配置自己的tensor adapter：

| 环节 | 现成图片流程的具体风险 | 本项目要求 |
| --- | --- | --- |
| complex转通道 | `[F,K,2]`、`[2,F,K]`容易混淆 | 模型前统一为连续的`[n,C,F,K]` |
| SAHI输入识别 | 所审版本CHW自动识别仅识别首维1/3/4，不识别2 | 明确传HWC浮点数组`[F,K,2]`，或直接复用坐标窗口裁tensor |
| SAHI Ultralytics adapter | 无条件`img[:,:,::-1]`会交换Re/Im，然后进入图片预测流程 | 自定义adapter，通道语义不交换 |
| 值域 | 默认训练/验证图片流程除255；tensor加载还可能按最大值猜测是否除255 | 所有路径显式归一化一次，禁止值域猜测 |
| 图片增强 | Mosaic/MixUp等实现含uint8画布或转换，HSV有BGR假设 | 首轮关闭RGB/uint8及改变空间尺度的增强 |
| 模型预处理 | 指定imgsz、rect仍可能resize或额外补边 | 检查backbone前实际shape及有效内容累计缩放 |

不能笼统说当前SAHI会把所有ndarray都变uint8：所审版本的HWC ndarray直传分支可以保留dtype。真正的问题是**具体输入分支、通道识别、adapter和后续预测器的组合**。[SAHI数组读取][S2]、[SAHI模型适配器][S3]、[YOLO训练预处理][U2]、[tensor输入的值域处理][U3]

第一版可只复用SAHI的窗口与坐标结构，由自有tensor预测入口调用模型核心。对于这里只有一维频率滑窗的情况，直接实现透明的小型裁片器也是可接受的工程选择；不需要为了使用SAHI而把科学数据伪装成RGB PNG。

## 6. 通道数：3不是检测任务的固有要求

STFT是complex64[F,K]；将实虚部分成float32[2,F,K]后，常规实数卷积就能处理。2通道不等于复杂度翻倍的“复数网络”，也不要求encoder/decoder全部改为2维。首层输出通道保持原值，后续模块接口可以原样复用。

| 表示 | tensor通道 | 初始用途 | 主要限制 |
| --- | ---: | --- | --- |
| log-power | 1 | A0：最简、易解释的检测baseline | 不保留相位 |
| Re/Im | 2 | A1：保留复数STFT信息，首选复数输入改造 | 对全局相位旋转敏感，需要实验确认收益 |
| 幅度+原始相位 | 2 | 可研究，暂不作为第一选择 | 相位在−π/π跳变，低幅度处不稳定 |
| log幅度+sin相位+cos相位 | 3 | 后续相位编码参照 | 虽然恰好3通道，也不是RGB语义 |

这是数学结构与工程取舍，尚无本项目的表示精度对比。全局相位变化使Z→exp(jθ)Z并旋转Re/Im，而功率不变。Re/Im若要保持复平面几何，宜共享幅值尺度，不应随意用不同系数独立压缩两通道。

建议保留原始复STFT，模型输入另行构造。跨采样率可采用明确的两边谱密度尺度，例如在FFT未额外归一化时：

\[
\tilde Z=\frac{Z}{\sqrt{F_s\sum_l w_l^2}},\qquad P=|\tilde Z|^2.
\]

随后A0使用相对固定参考的`log(P+ε)`，A1使用`[Re(tilde Z),Im(tilde Z)]`再除以共同尺度。训练部分估计的尺度/分位数必须冻结用于val/test；固定参考、ε、裁剪与填充值均要记录。复数双边谱不机械套用实信号单边谱的倍乘规则。上述归一化未证明是本数据最优，也不能据此声称已有绝对功率标定。[SciPy谱归一化说明][P2]

### 最少需要改哪些模型位置

YOLO11s：构造时明确`DetectionModel(...,ch=C,nc=14)`，或在所审版本的数据配置中使用channels；修改首层输入通道，不改其输出通道。再统一Dataset、collate、训练/验证预处理、预测器和导出。**不能仅改channels字段就认为整个复数浮点训练链已经正确。**

RGB预训练迁移也需记录：所审8.4.215代码对3→1首层采用RGB核求和，对3→2直接复制前两个核，不是具有Re/Im物理语义的初始化。对相同标量z复制成[z,z,z]，RGB核求和可保持首层线性响应；这不意味着RGB语义转移到无线信号。若使用不同版本，应核查真实加载日志。[模型构造与首层加载][U1]

2→3的1×1 adapter可保留原RGB首层，但仍需改数据链，不会自动解决缩放、值域或RGB增强。对于本项目，直接C通道stem更透明；adapter留作预训练策略对照。

RT-DETRv2-S同理只需改R18输入stem、归一化与权重加载，encoder/decoder隐藏维度无需因C改变。官方PResNet的pretrained=True会严格加载RGB首层；修改为2通道后，应关闭这条严格加载并明确迁移匹配权重，类别数另设14。[R18 stem与预训练加载][R2]

## 7. CNN步长、Transformer patch与STFT格点不能混为一层

| 概念 | 作用位置 | 不能据此推断什么 |
| --- | --- | --- |
| STFT bin | 原生时频采样 | 4个bin不是4个网络特征token |
| CNN stride | 特征提取与检测层 | stride8不意味着小于8像素必然消失 |
| ViT patch | 输入投影/分块 | patch16不代表每块仅保留一个标量 |
| 位置编码 | 给token提供位置关系 | 插值编码不能恢复已经丢失的信号信息 |
| object query / RoI池化 | 检测表示 | 固定query数、7×7 RoI不意味着全图固定方形 |

RT-DETRv2-S采用ResNet-18 CNN产生stride8/16/32特征，默认HybridEncoder只在最深层做全局注意力，再融合多尺度；它不是先把原图按16×16 patch切开的ViT。[官方配置与hybrid encoder][R1]

以本次默认候选Sf=2048为例：

| 输入 | stride8 | stride16 | stride32 | 原图patch16的空间token网格 |
| --- | --- | --- | --- | --- |
| 2048×32 | 256×4 | 128×2 | 64×1 | 128×2 |
| 2048×224 | 256×28 | 128×14 | 64×7 | 128×14 |
| 2048×448 | 256×56 | 128×28 | 64×14 | 128×28 |

这些是结构推算。20 ms无重叠桶在最深层时间轴只剩1个位置，值得单独评估短时边界；浅层仍参与检测，因此不能直接判为不可用。4.2个频率bin的窄带目标在stride8上不足一个步长，也不能用原生几何覆盖代替真实检测结果。框回归输出是连续坐标，但能否可靠学习仍需验证集分层指标。

真正改用ViT时，还要核查patch整除、位置编码、空间token重排、CLS/register剔除、多尺度neck和检测头对接。torchvision传统ViT显式要求H=W=image_size；timm可通过dynamic_img_size开启矩形位置编码插值，另需dynamic_img_pad才自动补到patch倍数。[torchvision ViT][V1]、[timm动态尺寸][V2]、[patch补边][V3]

例如patch14下1024×256不能直接整除，补边可变1036×266；新增register token不增加时频网格。若检测neck另需stride32对齐，应检查完整组件约束，而不是对所有ViT盲目统一按32补边。**第一版先沿用成熟CNN/混合检测器，比同时换ViT骨干更容易解释改造效果。**

## 8. 切片宽度如何选：不仅看能否forward

τ=0.7 ms时，物理频率分辨率Δf≈1.4286 kHz；Sf格对应Sf/τ Hz。

| Sf | 片物理频宽上限 | train平均片数 | test平均片数 | train无重叠输入MP | train 50%重叠输入MP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1.4629 MHz | 30.2199 | 37.4616 | 3.1787 | 6.2178 |
| 2048 | 2.9257 MHz | 15.1368 | 18.8380 | 3.1916 | 6.2428 |
| 4096 | 5.8514 MHz | 7.8368 | 9.7208 | 3.2915 | 6.4384 |

仅沿频率切片，20%重叠、末片贴边、无额外全图推理，网络只补到32倍数。MP是所有片输入空间面积之和，不包括通道、权重、激活、优化器或后处理，不是速度/显存实测。表按valid K重算；本数据center补边未改变最终bucket宽度，因此网络面积代理相同。由于时间不切片，STFT时间重叠不改变片数，只增加单片时间宽度。

为什么建议2048起步：相较1024，调用量约减半，累计输入面积近似相同，并能让2 MHz目标有机会在单片完整出现。4096进一步减少调用，但峰值单片变大，注意力和batch效率也会变化；累计面积相近不代表同样速度。

### 细片的真实上下文边界

| Sf | train自身宽于片的实例 | test自身宽于片的实例 |
| ---: | ---: | ---: |
| 1024 | 20,310 / 58.4376% | 11,741 / 58.6522% |
| 2048 | 9,483 / 27.2853% | 5,428 / 27.1156% |
| 4096 | 9,483 / 27.2853% | 5,428 / 27.1156% |

这是**不能完整包含的下界**，并非检测漏检率；带宽较小的实例仍可能跨边界。2048→4096未降低此比例，因为剩余宽带群跨越更大的带宽档位。统计基于原始instances.csv，不删除任何实例。

因此，第一场全任务baseline建议使用**同一STFT的局部细视图＋一张全频粗功率视图**。粗视图明确做功率聚合/降采样，用于宽带完整上下文；细视图不resize，保窄带细节。这不是双窗。A0单通道下两种视图可共用检测器及通道语义，是最容易解释的起点。粗视图损失窄带细节是有意的，不能把它的分辨率算到细视图头上。

为使粗视图的尺寸也具体可审，初始可将全频功率P按频率区间聚合到2048行，时间K保持不变，再取log并补时间轴。于是A0两条输入均为`[n,1,2048,Kpad]`：细视图每片保留原生约2.93 MHz，粗视图一张覆盖整个B。粗视图记录频率缩放2048/F，标签按全B映射；它不满足细视图的“频率缩放为1”，应作为独立视图报告。先在功率上聚合，再取log，不把log值均值或复数均值当成功率均值。每个原始样本额外增加一张粗视图，前述成本表尚未计入这一张。

A1复数细视图可先沿相同空间协议验证；若扩展为完整双通道方案，需要独立定义粗视图表示。**不能直接平均复数系数冒充功率聚合**，因为不同相位会相消。可以保留单独的功率上下文入口，或研究明确的相位聚合；这是后续改良，不作为本轮已完成事项。

局部训练片段与全局完整框的标签语义、跨视图合并仍需在正式训练前固定。当前不直接套SAHI默认面积过滤、不删同框标签、不以“已经有NMS”宣称可恢复完整宽带事件。此处只保留不可省略的输入上下文要求，不展开恢复算法。

## 9. 多模型能力清单与选型依据

表中“支持”指所列PyTorch实现的核心尺寸路径；不表示默认命令已经保留原始像素，也不表示已在服务器运行。示例1024×256与1024×640两轴均可被32整除，适合作为最初验收形状。

| 模型与实现 | 原生矩形能力 | 默认或接口中的尺寸变化 | 保留矩形的实施要点 |
| --- | --- | --- | --- |
| YOLOv5s，官方v7.0普通P5版 | 支持`[n,3,H,W]`，二维网格随形状变化 | 图片AutoShape会letterbox | 明确tensor或图片预处理；P5按32对齐，P6另核查64 |
| YOLOv8s，Ultralytics | 支持上述两个矩形示例 | 图片推理会按imgsz缩放；训练imgsz为标量 | 推理可指定(H,W)；精确矩形训练需控制dataloader与组批，rect=True本身不保证无缩放 |
| YOLO11s，Ultralytics | 同上，核心没有固定640方形 | 同上 | 保留为主baseline；验收train/val/predict三条路径 |
| YOLOX-s，官方实现 | 支持矩形；input_size/test_size本身为(H,W) | 默认640方形及多尺度增强 | 两轴按32对齐，统一train/test尺寸并控制多尺度和增强 |
| RTMDet-s，MMDetection 3.3.0 | 核心支持矩形，点网格按实际特征形状生成 | 标准配置含640方形crop/pad及随机resize | 改两阶段训练与测试pipeline，使用按32补边，检查batch_augments |
| RT-DETRv2-S，官方R18 | 支持矩形，encoder/decoder使用实际H/W | 默认Resize和eval缓存为640方形 | 修改变换；多桶评估可用eval_spatial_size=null；关闭collate尺度扰动 |
| RT-DETR-L，Ultralytics | 核心支持矩形；当前显式tuple推理路径也可矩形 | 标量imgsz默认stretch至方形；标准训练imgsz为标量 | 不将docstring的square描述误当核心限制；矩形训练需单独适配数据链 |
| FCOS-R50-FPN，Torchvision | 支持矩形NCHW | 默认短边800、长边最多1333，再按batch补边 | 调整/替换模型内部transform.resize，保留归一化、padding与坐标元数据 |
| RetinaNet-R50-FPN，Torchvision | 同上 | 同上 | 同上；anchor按实际图与特征图生成 |
| Faster R-CNN-R50-FPN，Torchvision | 同上 | 同上 | RoI固定池化尺寸不限制全图H/W |
| 原始DETR-R50，官方正弦位置编码版 | 支持矩形NCHW及padding mask | 默认数据增强仍有resize | 保留mask、控制pipeline；不将结论直接套用到每轴只有50个embedding的可选learned位置编码 |
| SSD300-VGG16，Torchvision成品 | 标准成品流程固定300×300 | 对外可接收不同尺寸，但transform用fixed_size统一 | 作为固定输入流程的反例，不列为本项目直接采用的矩形候选；修改架构/先验另论 |

分组证据：

- YOLOv5：[动态二维网格](https://github.com/ultralytics/yolov5/blob/v7.0/models/yolo.py#L58-L88)。YOLOv8/11：[矩形batch形状](https://github.com/ultralytics/ultralytics/blob/4d254d8141e49dd85009e1ec7657ab3c4ee8c150/ultralytics/data/base.py#L484-L506)、[训练标量imgsz检查](https://github.com/ultralytics/ultralytics/blob/4d254d8141e49dd85009e1ec7657ab3c4ee8c150/ultralytics/engine/trainer.py#L362-L365)。训练接口限制不等于核心空间形状限制。
- YOLOX：[尺寸配置说明](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/docs/manipulate_training_image_size.md)、[二维检测网格](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/yolox/models/yolo_head.py#L215-L245)。
- RTMDet：[标准尺寸变换配置](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py)、[实际特征尺寸与网格](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/rtmdet_head.py#L240)、[批量补边与张量堆叠](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/data_preprocessors/data_preprocessor.py#L37-L49)。
- Torchvision三种FPN检测器：[共同resize规则](https://github.com/pytorch/vision/blob/4260301135e34180d865426765487113f84e7707/torchvision/models/detection/transform.py#L25-L72)、[batch_images](https://github.com/pytorch/vision/blob/4260301135e34180d865426765487113f84e7707/torchvision/models/detection/transform.py#L237-L255)。固定1024×256实验可设min_size=256/max_size=1024令缩放比为1；这不是对任意尺寸都不缩放的通用设置。任意尺寸保留需统一控制train/eval的resize；私有_skip_resize仅作用于training分支，不能据此声称全流程关闭。
- DETR：[模型输入](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L44-L65)、[批量补边和mask](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/util/misc.py#L307-L329)、[位置编码](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/position_encoding.py)。
- SSD300：[成品固定尺寸与transform](https://docs.pytorch.org/vision/main/_modules/torchvision/models/detection/ssd.html)。对外接口接收可变尺寸，不自动表示backbone入口可变。


## 10. 模型改造应止于哪些接口

| 层次 | 第一版处理 | 暂不同时引入的改动 |
| --- | --- | --- |
| 表示与Dataset | complex/float数组，A0/A1，固定轴和数值协议 | 伪彩色PNG、RGB色彩增强 |
| 切片与collate | 频率滑窗、按T分桶、只补边 | 不明缩放、改变T的伸缩 |
| stem | C=1/2，输出通道沿用原模型 | 为输入适配而重做整套backbone |
| backbone/neck/head | 复用架构、类别数14，检查真实特征尺寸 | 立刻增P2层、改stride或换ViT |
| 训练/验证/预测 | 同一数值和几何协议，记录权重迁移 | 只改训练Dataset而推理仍走RGB默认值 |
| 导出 | 记录每个矩形或动态profile范围、C与dtype | 复用固定640×640×3引擎 |

YOLO的imgsz/rect是工具接口，并非原生尺寸证明；训练、训练期间验证、独立验证与预测都要检查。所审版本训练期间验证仍可能按rect与stride额外补边，仅设置训练rect=False不足以保证指定tensor。采用自有Dataset、collate和tensor预处理适配，复用成熟模型、loss及训练机制；本轮尚未实现完整训练器。[训练组批实现][U4]

官方RT-DETRv2多bucket可设置eval_spatial_size=null，动态生成位置编码与anchors；若使用固定缓存，应与当前[H,W]一致并同步重建。R18的collate scales保持null。模型输入是[H,W]，官方后处理orig_target_sizes为[W,H]，必须检查方向。所核版本包含矩形位置编码顺序修复，旧版能forward不保证方向正确。[动态位置编码][R1]、[decoder动态anchors][R3]、[位置编码修复][R4]

从固定640缓存的预训练检查点迁移至矩形或动态尺寸配置时，应排除或重建旧anchors/valid_mask缓存，按匹配权重迁移；不能直接按完整resume严格加载。同名缓存尺寸不符时，仅设strict=False也不足以跳过冲突。[官方tuning与resume加载逻辑][R5]

## 11. 坐标协议与最小验收

图像x轴为时间、y轴为频率，bbox采用[x1,y1,x2,y2]。频率片起点为j0时，局部y恢复为y_global=y_local−pad_top+j0；时间只去padding，不加切片偏移。SAHI shift_amount应对应[x_shift,y_shift]=[0,j0]。

物理时间映射必须基于实际帧时刻。若第0帧中心为t0、间隔δ=h/Fs，采用像素中心坐标m+1/2，则x=(t−t0)/δ+1/2；inverse用相同约定。center=True时t0=0，valid时首帧中心约为τ/2。频率同样用实际fftshift频率中心和Δf定义pixel-center/edge对应。**不要忽略首帧时刻、半格偏移和尾部规则，直接把所有标签统一写成t/T乘K。**输出最终裁回原物理观测范围，补边不延长真实T或B。

往返验收应包括t=0/T和整频宽框。中心网格与物理边界可能相差不足一格，例如50%重叠center下T=100 ms的终点映射为x≈286.214而K=286；补边画布仍有空间容纳。不要在恢复物理坐标前先按未补边的K/F裁框，否则会引入额外边界误差。模型后处理也须使用同一坐标约定，不能仅验证shape。

第一轮无训练验收应包括：

1. 同T不同Fs的F/K与预期一致；N使用真实样本点数；仅改变Fs时不改变物理hop。
2. 已知tone或小矩形信号映射到正确的频率方向、时间方向；Re/Im顺序可精确重构complex数据。
3. 记录原始片形状、源频率偏移、两轴累计resize比例、padding、最终NCHW、dtype、值域；原生细片要求两轴累计缩放均为1。
4. 在backbone第一层前挂检查，覆盖C=1/2、最短/最长T桶、多个batch；不以最后tensor尺寸相同掩盖中间缩小再放大。
5. 检查frame/tile/physical坐标往返，检查channel顺序、除255和默认normalize未重复执行。
6. 接通正式训练前再验证loss与标签匹配、完整样本级预测、跨片/粗细视图合并；全局评估不能只报裁片AP。
7. 在服务器4090上测batch=1起的峰值显存、STFT、预处理、前向与合并耗时，再尝试更大batch；仅在验证集选择配置。

本轮完成了第1/2项中的合成STFT及通道重构，以及第4项的核心网络前向；未完成真实数据loss、全局合并和GPU成本验收。所有预训练RGB适配、真实数据phase收益也尚未实测。

## 12. 本轮实际运行记录与后续顺序

[张量链路核查.json](张量链路核查.json)记录了16组STFT和12次YOLO11s前向：

- 合成IQ的Fs/T组合：5 MHz×20 ms、20 MHz×80 ms、80 MHz×20 ms、5 MHz×150 ms；每组比较两种hop、center开关。
- complex64双边谱、周期Hann、PSD尺度归一化，构造Re/Im float32，频率裁片后只补时间轴；有效区域complex重构逐元素完全相同。
- YOLO11s随机权重，nc=14、ch=2；首层权重形状实际为[32,2,3,3]。hook观察到的入口与设定NCHW完全一致。
- 输出18=4+14个通道，候选点数与stride8/16/32的矩形特征面积之和一致，全部有限；测试包括batch=2。
- 另补做2次单通道前向，覆盖2048×32/224；12组末样本脉冲检查验证无重叠center可能漏尾、50%重叠center在所测时长均观察到末样本；3组tone检查确认跨Fs的fftshift频率方向。
- 没有优化器、没有训练步、没有预训练权重下载；不是性能对比，也未测试完整Ultralytics高层图片/SAHI pipeline。

前一版的推进顺序如下；当前尺寸试验优先验证上面的统一2048×512方案：

1. 固定轴、0.7 ms窗、hop和边界约定；默认Sf=2048，按T分桶，无时间切片、无细片resize。
2. 服务器复现上述C=1/2核心前向，并实现统一float Dataset/adapter与三条数值预处理路径。
3. 用A0功率细视图＋全频粗视图完成最小完整样本级协议，再决定第一场正式训练；保留原始实例，不直接去重或ignore。
4. 先在相同细视图、相同切片和训练条件下比较A0 log-power与A1 Re/Im；完整任务比较须保持粗功率上下文及合并协议一致，不能把表示差异全部归因于相位。随后比较RT-DETRv2-S、1024/4096与重叠率，每次只增加一个主要因素。
5. 只有窄带分层结果确实暴露stride、neck或输入表示瓶颈时，再考虑P2、小patch/非对称patch、相位分支或双窗。

0.7 ms单窗此前的97.7989% train总体、96.2536%最差类是q=4原生几何诊断；不能当成细片完整上下文覆盖、模型召回率或mAP。双窗99.97%仍仅作后续几何质量参照。

## 附：保留的既有产物与复核口径

- [项目理解](项目理解.md)：数据与决策入口。
- [原有尺寸与成本明细](SAHI尺寸与成本评估.csv)、[比较图](SAHI覆盖与成本.png)、[可编辑图](SAHI覆盖与成本.svg)、[复算脚本](sahi_size_tradeoff.py)：旧表采用50%重叠、帧中心[0,T)等明确假设，保留作历史几何比较；不能冒充本轮valid帧或2通道模型实测。
- [物理窗配置](STFT物理窗长配置.csv)：各B/T的窗点数与hop参照。
- 本轮主表SHA256：samples.csv为`f882d913a3efccc259dfdfa59f2abc6a9fb080aae1ac0a1f787e7fbb12cbec8b`；instances.csv为`fbd072fa0578fb6ac39377ee0a3e4443c87d12fa78d6da6e98f60f279e5ed0f3`。实际值以本轮核查JSON为准。

[P1]: https://docs.pytorch.org/docs/2.14/generated/torch.stft.html
[P2]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
[S1]: https://arxiv.org/abs/2202.06934
[S2]: https://github.com/obss/sahi/blob/10061afd5fa076e00f8eda55e2ee290f9d72f3c1/sahi/utils/cv.py
[S3]: https://github.com/obss/sahi/blob/10061afd5fa076e00f8eda55e2ee290f9d72f3c1/sahi/models/ultralytics.py#L149-L154
[U1]: https://github.com/ultralytics/ultralytics/blob/4d254d8141e49dd85009e1ec7657ab3c4ee8c150/ultralytics/nn/tasks.py
[U2]: https://github.com/ultralytics/ultralytics/blob/4d254d8141e49dd85009e1ec7657ab3c4ee8c150/ultralytics/models/yolo/detect/train.py#L114-L134
[U3]: https://github.com/ultralytics/ultralytics/blob/4d254d8141e49dd85009e1ec7657ab3c4ee8c150/ultralytics/data/loaders.py#L669-L688
[U4]: https://github.com/ultralytics/ultralytics/blob/4d254d8141e49dd85009e1ec7657ab3c4ee8c150/ultralytics/models/yolo/detect/train.py#L62-L74
[R1]: https://github.com/lyuwenyu/RT-DETR/blob/df2a96c8c3a098567a1149bf0248da6efe6d495d/rtdetrv2_pytorch/src/zoo/rtdetr/hybrid_encoder.py
[R2]: https://github.com/lyuwenyu/RT-DETR/blob/df2a96c8c3a098567a1149bf0248da6efe6d495d/rtdetrv2_pytorch/src/nn/backbone/presnet.py#L174-L218
[R3]: https://github.com/lyuwenyu/RT-DETR/blob/df2a96c8c3a098567a1149bf0248da6efe6d495d/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py#L480-L501
[R4]: https://github.com/lyuwenyu/RT-DETR/commit/e97db5a685a874c01c7977cb3a42ab0684ea134b
[R5]: https://github.com/lyuwenyu/RT-DETR/blob/df2a96c8c3a098567a1149bf0248da6efe6d495d/rtdetrv2_pytorch/src/solver/_solver.py
[V1]: https://github.com/pytorch/vision/blob/4260301135e34180d865426765487113f84e7707/torchvision/models/vision_transformer.py#L269-L285
[V2]: https://github.com/huggingface/pytorch-image-models/blob/d90ea08620e684ade06b61bed382bb4f42767b1d/timm/models/vision_transformer.py
[V3]: https://github.com/huggingface/pytorch-image-models/blob/d90ea08620e684ade06b61bed382bb4f42767b1d/timm/layers/patch_embed.py#L120-L139
