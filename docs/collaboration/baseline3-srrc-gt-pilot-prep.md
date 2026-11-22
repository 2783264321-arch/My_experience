# SRRC GT 小跑准备

这是 [SRRC 阶段 0](opencode-srrc-local-iq-phase0.md) 的后续开发，依据 [局部 I/Q 专家方案](baseline3-local-iq-expert-plan.md)。目的是在 D-FINE 占用 4090 时完成可独立验收的代码，空卡后直接启动可比较的小跑。

## 现在能完成的工作

- 修正裁剪时间缓冲。框坐标以 ms 计，每侧加 0.2 ms，转换为原始 IQ 点数时再从 ms 转到秒。取窗按原始 IQ 实际读取的首样点回算，窗尾限制在 GT 时段内。
- `prepare --metadata-only` 只读 fit 的标注 JSON，固定 `group_id` 分组的内部训练/开发集和每个实例的交叠层。fit 内部开发集取固定种子下 5 折的第 0 折。同一 IQ 或重复组只进一侧。各实例用 `sample_id` 和源标签的零基下标 `signal_idx` 对齐。SpaceNet 的正式 val 和 test 留在原划分中。
- `prepare` 可以只切一个小组子集，也可以生成完整 fit 缓存。单进程读取 IQ，走相同的带限、20 MS/s 重采样和取窗前端，保存每实例 8 个 `2×1,024` float32 窗。保留窗内 I/Q 的原幅度，CNN 在进网络前做相同的 RMS 归一化。1,824 个实例的主张量约 114 MiB，16 窗约 228 MiB。先用 8 窗做首轮对照。
- D-FINE 同时使用 CPU 数据加载和磁盘。要求该任务零干扰时，在本机或独立环境做编程、合成验证和结果复核。真实 IQ 在 AutoDL 时，起跑前要留出切片时间。若服务器资源有明确余量并要提前试，先做标注清单或有上限的单进程缓存，结合 D-FINE 步时和读取压力判断是否继续。

## 命令

从仓库根目录执行。用 AutoDL 上已有的项目 Python 环境。输出路径放在数据盘。

~~~bash
srrc_out=/root/autodl-fs/runs_local/srrc-gt-pilot

# 标注和分组核验，不读 IQ .bin
CUDA_VISIBLE_DEVICES="" python -m scripts.baseline3srrc_gt_pilot prepare \
  --metadata-only --out "$srrc_out/fit_manifest.json"

# 资源允许时切完整 fit。只试小缓存时可加 --max-groups 40，并另设输出文件
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.baseline3srrc_gt_pilot prepare --out "$srrc_out/fit_8win.npz"

# D-FINE 释放 GPU 后，同一缓存对比两个 CNN
python -m scripts.baseline3srrc_gt_pilot train \
  --cache "$srrc_out/fit_8win.npz" --out-dir "$srrc_out/seed1729" \
  --device cuda --epochs 5 --batch 8
~~~

两组共用源文件划分、窗和模型主干。8 指每批的**实例数**。每个实例的所有窗口先分别出 logits，再平均成一个实例分数，用实例标签算损失。输出每轮完整开发集和交叠分层指标、开发集逐实例 logits 及运行配置和实际峰值显存。训练命令也可用 `--device cpu` 在本机跑小样本检查。

复用 [原版 B0 结果与方法](https://github.com/2783264321-arch/My_experience/blob/2c5e815/reports/baseline3_srrc_local_iq/srrc_local_iq_results.md) 时，用同一份清单和开发集，以 `sample_id + signal_idx` 配对。库里的旧 61.54% 对应原有留出。CNN 小跑先验收成本和学习情况，决定正训前补上同数据的 B0 读数。预测框对照仍用真正的检测器候选。

## 本机已核验

早期 DSP 回归和毫秒缓冲回归检查 使用合成 IQ 通过。另用 24 个合成文件验证了无 IQ 的清单生成、组内分割、8/16 窗缓存，以及幅度和 I/Q 各一轮的 CPU 前向、反向和逐实例 logit 导出。真实 fit 裁剪、服务器吞吐和正式分类表现待 AutoDL 实测。
