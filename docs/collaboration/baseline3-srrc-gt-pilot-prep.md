# SRRC GT 小跑准备

这是 [SRRC 阶段 0](opencode-srrc-local-iq-phase0.md) 的后续开发，依据 [局部 I/Q 专家方案](baseline3-local-iq-expert-plan.md)。目的是在 D-FINE 占用 4090 时完成可独立验收的代码，空卡后直接启动可比较的小跑。

## 现在能完成的工作

- 修正裁剪时间缓冲。框坐标以 ms 计，每侧加 0.2 ms，转换为原始 IQ 点数时再从 ms 转到秒。取窗按原始 IQ 实际读取的首样点回算，窗尾限制在 GT 时段内。
- `prepare --metadata-only` 只读 fit 的标注 JSON，固定 `group_id` 分组的内部训练/开发集和每个实例的交叠层。fit 内部开发集取固定种子下 5 折的第 0 折。同一 IQ 或重复组只进一侧。各实例用 `sample_id` 和源标签的零基下标 `signal_idx` 对齐。SpaceNet 的正式 val 和 test 留在原划分中。
- `prepare` 可以只切一个小组子集，也可以生成完整 fit 缓存。单进程读取 IQ，走相同的带限、20 MS/s 重采样和取窗前端，保存每实例 8 个 `2×1,024` float32 窗。保留窗内 I/Q 的原幅度，CNN 在进网络前做相同的 RMS 归一化。1,824 个实例的主张量约 114 MiB，16 窗约 228 MiB。先用 8 窗做首轮对照。

## D-FINE 运行时的 CPU 与存储安排

- 这次监控显示容器配额 16 vCPU、内存 120 GB。按多核使用率口径，CPU 常态约占 4–6 核，尖峰接近 15–16 核。内存约 60 GiB。可以考虑小规模单进程预处理，主要观察 D-FINE 的步时和共享存储读取。
- 先生成完整 fit 的标注清单，再以 **40 组、一个低优先级进程、单线程**实切 8 窗缓存。D-FINE 正常训练时，用相邻稳定训练段的 `train_log.csv` 比较试切前后的 `step_s` 中位数、尾部耗时和 GPU 等数据的空档。验证、保存权重和换轮的区间另看。如果出现持续的可辨别变慢或明显缺数据，停止试切进程，保留已交付的清单。步时稳定后才考虑同一低优先级单进程切完整 fit。
- 本地数据盘当前约剩 6 GB，优先供正式训练写权重。SRRC 清单、合并后的 `.npz` 缓存和结果可放已挂载的 `/root/autodl-fs`，开工前以实例的实际挂载和剩余空间为准。它是网络文件存储，准备缓存时读原始 IQ 仍可能占用共享带宽。CNN 启动时一次把缓存载入内存。8 窗完整缓存约 0.12 GB，按超额部分单价折算约 ¥0.0012/日的新增计费量，实际按当日峰值和最低费结算。 [AutoDL 文件存储说明](https://www.autodl.com/docs/fs/)
- 优先保护现有训练。试切前后的对照用于判断可观测影响。若要求严格隔离，实切安排在不争用这次训练资源的环境，或在 D-FINE 结束后执行。SRRC 的 GPU 小跑仍排在 GPU 释放之后。

## 命令

从仓库根目录执行。用 AutoDL 上已有的项目 Python 环境。输出放在已挂载的 AutoDL 文件存储。

~~~bash
srrc_out=/root/autodl-fs/runs_local/srrc-gt-pilot
df -h /root/autodl-fs /root/autodl-tmp

# 标注和分组核验，不读 IQ .bin
CUDA_VISIBLE_DEVICES="" python -m scripts.baseline3srrc_gt_pilot prepare \
  --metadata-only --out "$srrc_out/fit_manifest.json"

# 先试 40 组，观察 D-FINE 步时
nice -n 19 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.baseline3srrc_gt_pilot prepare --max-groups 40 --out "$srrc_out/fit_8win_trial.npz"

# 试切步时稳定后，同一低优先级单进程准备完整 fit
nice -n 19 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.baseline3srrc_gt_pilot prepare --out "$srrc_out/fit_8win.npz"

# 缓存生成后，先在 CPU 上跑同窗 B0
nice -n 19 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.baseline3srrc_b0_same_windows \
  --cache "$srrc_out/fit_8win.npz" --out-dir "$srrc_out/b0_same_windows"

# D-FINE 释放 GPU 后，同一缓存对比两个 CNN
python -m scripts.baseline3srrc_gt_pilot train \
  --cache "$srrc_out/fit_8win.npz" --out-dir "$srrc_out/seed1729" \
  --device cuda --epochs 5 --batch 8
~~~

两组共用源文件划分、窗和模型主干。8 指每批的**实例数**。每个实例的所有窗口先分别出 logits，再平均成一个实例分数，用实例标签算损失。输出每轮完整开发集和交叠分层指标、开发集逐实例 logits 及运行配置和实际峰值显存。训练命令也可用 `--device cpu` 在本机跑小样本检查。

## 本轮判断与执行顺序

在同为 10 MHz 的 SRRC 10/11 类里，先回答一个受控问题。同样的时间位置、同样多的带限基带 IQ，现有统计方法、只看幅度的 CNN、保留 I/Q 的 CNN 各能分到什么程度？理想符号中心的 QPSK 只有一个幅度半径，方形 16QAM 有三个。这个性质给了符号恢复明确的动机，但当前缓存是未经符号同步的波形，任意采样点会叠加相邻符号的贡献。GT 框给出位置，同一时频区域内的干扰仍会随目标进入缓存。

### 1. 独立复核裁剪与带边

用合成单音、已知参数的 10 MHz SRRC 波形和少量真实 fit 样本，复核现有缓存对应的前端。覆盖不同观测采样率、正负频移、框靠近原观测边界的情况。核对 IQ 格式、时间点数和每侧 0.2 ms 缓冲、观测中心的频率基准、搬频方向、20 MS/s 重采样和最终取窗是否在标注时段内。

量出通带内部、标注边缘及带外的实测增益，建议至少看基带两侧的 4.4、5.0、5.1 MHz 和更外侧。长裁剪下，当前掩膜约在 ±4.59 MHz 开始过渡，到 ±5.61 MHz 截止。本机合成单音在 +5.0 MHz 实测衰减约 3.7 dB，现有掩膜的设计响应对称。这是滤波响应，仍需用已知波形判断滚降区、框首尾或中心偏差是否损伤调制证据。把带外邻道抑制和同频干扰残留分开解释。请独立判断影响。若发现会使现有缓存失效的前端错误，先写回证据和受影响范围，保留原缓存作核查。

[裁剪与取窗](https://github.com/2783264321-arch/My_experience/blob/dbc72c1/scripts/baseline3srrc_phase0_verify.py#L86-L138) · [下变频与带限](https://github.com/2783264321-arch/My_experience/blob/dbc72c1/scripts/baseline3_dsp.py#L46-L77)

### 2. 三种方法用同一份缓存

前端核验支持复用后，按上面的命令用现有 `fit_8win.npz` 跑 CPU 同窗 B0、幅度 CNN、I/Q CNN。沿用固定的 1,458 / 366 内部 train/dev 分组。两个 CNN 用同一骨干和训练预算，先各跑 5 轮、种子 1729。按 `sample_id + signal_idx` 校对三种方法的实例与分层标记。两组 CNN 都做每窗 RMS 归一化，幅度和 I/Q 的对比直接检验复数信息是否有帮助。同窗 B0 还用了绝对功率、频谱和相位差等统计量，其结果作为传统特征的参照。

将总体及无、弱、强重叠各层的样本数、平衡准确率、10/11 类召回写到 `docs/collaboration/` 的结果记录中。附同实例的改对/改错、运行配置、耗时和 CNN 峰值显存，保留逐实例结果以供复核。重叠按其他标注框与目标框的最大交叠面积占目标面积分层，无交叠为 0，弱交叠在 0 与 25% 之间，强交叠为至少 25%。核对目标本身已排除。这个比例描述框几何，干扰强度需另测。

### 3. 根据分层结果决定下一步

在 20 MS/s 下，8 个 1,024 点窗合计只看 0.4096 ms，约是一条 20 ms 信号的 2%。窗数决定观察位置的覆盖，窗长决定局部的连续观测量。若无重叠组能分开，错误主要集中在重叠组，且不同窗口有明显不同的证据，下一轮优先保留原 8 个位置再加 8 个。若需要比较位置覆盖和连续长度，则用相同点数预算的 16×1,024 与 8×2,048 对照。若符号恢复更值得优先验证，先用同窗对照检验匹配滤波和定时能否稳定改善幅度证据，同步失败采用未同步基线并计入全体成绩。

本轮的实验范围是 fit 内的 GT 分类和分层评估。查看实测后再决定下一轮投入。检测器接入留待后续验证。

## 本机已核验

早期 DSP 回归和毫秒缓冲回归检查使用合成 IQ 通过。另用 24 个合成文件验证了无 IQ 的清单生成、组内分割、8/16 窗缓存，以及幅度和 I/Q 各一轮的 CPU 前向、反向和逐实例 logit 导出。同窗 B0 的 53 维特征用 16 个合成窗口和旧实现逐项核对，三臂导出的开发集实例与交叠标记已通过配对检查。真实分类表现待 AutoDL 实测。

**AutoDL 实测（2026-11-22，OpenCode）**：完整 fit 清单 1,824 实例 / 1,571 组；40 组试切 33.34 s / 3.09 MB，D-FINE 步时 +1.54%；全量缓存 1,112.84 s / 114.4 MiB，D-FINE 同时段步时 +1.03%，未触发停止。缓存与监控记录见 [opencode-srrc-gt-pilot-cache.md](opencode-srrc-gt-pilot-cache.md)。

**同窗三臂实测（2026-11-23，OpenCode）**：前端核查通过、缓存可复用；5 epoch、3 seeds 的 balanced accuracy 为 B0 **0.6305**、幅度 CNN **0.5967±0.0662**、I/Q CNN **0.5295±0.0181**；strong 重叠层幅度 CNN 明显低于 B0。20 epoch 单 seed 诊断仍未给出稳定增益，当前应先稳定训练/分类头，再决定加窗或符号恢复。 [完整结果](opencode-srrc-three-arm-pilot.md)
