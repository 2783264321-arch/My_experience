# AutoDL + OpenCode 运行手册

2026-11-26。记录这台 AutoDL 实例的硬件、网络、磁盘和 OpenCode 使用方式，
避免每次重新踩坑。事实优先，命令可直接复用。

## 1. 实例事实

| 项 | 值 |
| --- | --- |
| 宿主机 CPU | 128 逻辑核（2× Xeon Gold 6430） |
| **实例 CPU 配额** | **16 vCPU**（`cpu.max=1600000 100000`） |
| 宿主机内存 | ~1.0 TiB |
| **实例内存配额** | **120 GiB**（`memory.max=128849018880`） |
| GPU | RTX 4090 24 GB，驱动 570.124.04 |
| 快速本地盘 | `/root/autodl-tmp`，50 GB，曾用 94%（仅剩 ~3 GB） |
| 共享数据盘 | `/root/autodl-fs` → `/autodl-fs/data`，200 GB，剩余 ~81 GB |
| 系统盘 | overlay 30 GB，剩余 ~17 GB |
| 网络 | AutoDL 学术加速，按 shell 生效，不稳定 |

**关键：`nproc` / `lscpu` / `free -h` 看到的是宿主机，不是租用配额。**
先读 cgroup：

```bash
nproc
cat /sys/fs/cgroup/cpu.max        # 1600000 100000 -> 16 cores
cat /sys/fs/cgroup/memory.max     # 128849018880 -> 120 GiB
cat /sys/fs/cgroup/cpu.stat       # nr_throttled
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
df -h /root/autodl-tmp /root/autodl-fs /
```

## 2. 网络

AutoDL 的学术加速：

```bash
source /etc/network_turbo
# http_proxy=https_proxy=http://172.29.51.4:12798
# no_proxy 包含 localhost/modelscope/aliyuncs/tencentyun/wisemodel
```

注意：

- **只对 github.com / huggingface.co 有意义**，且不承诺稳定。
- **不跨 shell 持久**；每个需要外网的 shell 都要重新 source。
- 对 pip/conda 往往**更慢**；装包前可先 `unset http_proxy https_proxy`。
- 直连 `git clone https://github.com/...`、`webfetch` 仍可能超时；
  不要在一个 clone 上阻塞几分钟。
- 我们的仓库 SSH remote（`github-myexperience`）push/pull 可用，即使
  HTTPS clone 超时。
- 外部仓库优先用 **GitHub MCP**：
  `tools.github.get_file_contents({owner, repo, path})`，返回
  `content[].resource.text`；目录列表用 `fields:["name","type","path"]`。
- 需要整仓时再用短超时浅克隆：

```bash
timeout 60 git clone --depth 1 <url> /tmp/opencode/<name>
```

## 3. 磁盘策略

- 代码、worktree、临时脚本：`/root/autodl-tmp`。
- 大数据：`.npz`、`.bin` 缓存、checkpoint、报告：`/root/autodl-fs/...`。
- 每次写 >1 GB 前先 `df -h`；`autodl-tmp` 只剩几 GB 时很容易把
  worktree/缓存写满。
- 现有 SpaceNet 数据：`/root/autodl-fs/SpaceNet_Dataset/advanced`；
  实验缓存：`/root/autodl-fs/runs_local/...`。

## 4. 并行与 GPU

- IQ/STFT/crop 提取用 `ProcessPoolExecutor(max_workers=16)`（12–16）；
  实例是 16 vCPU 配额，32–64 会超订并被 throttling。
- 不要同时跑多个 CPU 重任务（4 个 merge 并行已经超订 16 核）。
- 监控：`ps -eo pid,pcpu,pmem,etime,cmd --sort=-pcpu | head`，
  `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`，
  `cat /sys/fs/cgroup/cpu.stat` 看 `nr_throttled`。
- 单核 100% 不等于机器忙；看到只有一个 Python 进程 100% 就是串行瓶颈。
- 反例：Soft-NMS 固定候选合并优化到 ~82 s；直接 val ~800 s。
  后处理优先复用固定候选缓存，不要重跑检测器前向。

## 5. OpenCode 特性

- `execute` Code Mode：用 JS 组合多个工具调用；用 `search` 发现工具，
  不猜工具名。
- 后台 `shell`：长任务放后台，完成会通知；不要反复 sleep/读日志。
- `subagent`：昂贵实验前做只读审查；可并行独立探索。
- worktree/branch：实验隔离；`opencode.worktree_create/list/remove`。
- skill：把可复用流程写进 `.opencode/skills/...`。
- automation：定时/周期检查，不占用当前会话。
- 里程碑汇报：数据/模型就绪、任务启动、训练/推理完成、分析完成、
  commit/push 完成。

## 6. 复用检查清单

1. `nproc` / `free -h` / `nvidia-smi` / `df -h` 先看一眼。
2. 需要外网：先试 GitHub MCP；再 `source /etc/network_turbo` +
   短超时；不要阻塞。
3. 大文件写 `autodl-fs`。
4. 提取类任务用 16 进程（12–16）；GPU 任务保证 batch 足够大。
5. 后处理复用固定候选/缓存，不重跑前向。
6. 实验结束把新教训写回本手册、skill 或 ledger。
