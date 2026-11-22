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

# D-FINE 释放 GPU 后，同一缓存对比两个 CNN
python -m scripts.baseline3srrc_gt_pilot train \
  --cache "$srrc_out/fit_8win.npz" --out-dir "$srrc_out/seed1729" \
  --device cuda --epochs 5 --batch 8
~~~

两组共用源文件划分、窗和模型主干。8 指每批的**实例数**。每个实例的所有窗口先分别出 logits，再平均成一个实例分数，用实例标签算损失。输出每轮完整开发集和交叠分层指标、开发集逐实例 logits 及运行配置和实际峰值显存。训练命令也可用 `--device cpu` 在本机跑小样本检查。

复用 [原版 B0 结果与方法](https://github.com/2783264321-arch/My_experience/blob/2c5e815/reports/baseline3_srrc_local_iq/srrc_local_iq_results.md) 时，用同一份清单和开发集，以 `sample_id + signal_idx` 配对。库里的旧 61.54% 对应原有留出。CNN 小跑先验收成本和学习情况，决定正训前补上同数据的 B0 读数。预测框对照仍用真正的检测器候选。

## 本机已核验

早期 DSP 回归和毫秒缓冲回归检查使用合成 IQ 通过。另用 24 个合成文件验证了无 IQ 的清单生成、组内分割、8/16 窗缓存，以及幅度和 I/Q 各一轮的 CPU 前向、反向和逐实例 logit 导出。真实 fit 裁剪、服务器吞吐和正式分类表现待 AutoDL 实测。

**AutoDL 实测（2026-11-22，OpenCode）**：完整 fit 清单 1,824 实例 / 1,571 组；40 组试切 33.34 s / 3.09 MB，D-FINE 步时 +1.54%；全量缓存 1,112.84 s / 114.4 MiB，D-FINE 同时段步时 +1.03%，未触发停止。缓存与监控记录见 [opencode-srrc-gt-pilot-cache.md](opencode-srrc-gt-pilot-cache.md)。
