# 标签示意图与交互浏览器使用流程

工具：`scripts/plot_label_schematic.py`（2026-11-20 加入）。
把每个样本的标注信号实例画成归一化 0-1 时频示意（2:1 频率:时间，编号框，
半透明填充显示重叠）。`--mode browser` 生成**单文件自包含交互 HTML**，
只需 Python 标准库；`--mode static` 生成 PNG，需要 matplotlib。

## 前提

- 仓库：`/root/autodl-tmp/workspace/baseline`
- 数据（脚本顶部 `MANIFEST` / `DATA_ROOT` 两个常量，路径变了改这里）：
  - split 清单 `/root/autodl-tmp/workspace/spacenet/analysis/outputs/split_manifest.csv`
  - 标签 JSON `/root/autodl-fs/SpaceNet_Dataset/advanced/{train,test}/{stem}.json`

## 常用命令

```bash
cd /root/autodl-tmp/workspace/baseline

# 交互浏览器（fit+val，默认 7,500 样本 / 34,755 实例，~5 s，1.56 MB）
python scripts/plot_label_schematic.py --mode browser \
    --out reports/label_schematic_browser.html

# 含 test 的全量版（纯标签可视化，不涉及模型，~2.1 MB）
python scripts/plot_label_schematic.py --mode browser \
    --out reports/label_schematic_browser_all.html --include-test

# 只装某个 split
python scripts/plot_label_schematic.py --mode browser --out /tmp/opencode/val.html --splits val

# 静态 PNG：分层样张（每个观测带宽最密的 + 全局最密/最疏，~0.1 s/张）
python scripts/plot_label_schematic.py --mode static --stratified 10 \
    --out reports/label_schematics

# 静态 PNG：指定样本（可加 --dpi 200 提高分辨率）
python scripts/plot_label_schematic.py --mode static \
    --samples train/2,train/25 --out reports/label_schematics
```

## 打开交互版

- **离线**：HTML 单文件自包含、零依赖，scp / JupyterLab 下载到本地后双击即可；
- **服务器上起静态服务**（VS Code Remote / SSH 端口转发后访问
  `http://localhost:8791/label_schematic_browser.html`）：

```bash
cd reports && python -m http.server 8791        # 前台
# 或 nohup python -m http.server 8791 >/tmp/http8791.log 2>&1 &   # 后台
# 停止：pkill -f "http.server 8791"
```

## 浏览器内操作

- 输入 sample id 直达（支持只输数字，如 `2`）；`←/→` 翻页；random 随机；
- 按 split（fit/val/test）与观测带宽（5/20/30/40/50/80 MHz）筛选；
- **悬浮图内框**：信息卡（类别、带宽、时长、rel_bw、归一化范围、绝对频率/时间）；
- **悬浮左侧清单行**：同样的信息卡，且把对应框提到最上层并加粗——
  专门用于查看被其它框覆盖的实例；
- 头部：样本元信息 + 逐行重叠对列表（`#a ∩ #b · x% of #a · y% of #b`）。

## 成本预期（RTX 服务器实测）

| 操作 | 耗时 | 产物 |
| --- | ---: | --- |
| browser 构建（fit+val） | ~5 s | 1.56 MB 单文件 |
| browser 构建（含 test） | ~7 s | ~2.1 MB |
| static 单张 | ~0.1 s | 10-24 KB |

## 自定义

- 类别名 `NAMES`、家族配色 `FAM_COLOR` / `FAM_NAME`、纵横比 `ASPECT`
  都在脚本顶部的常量区；
- 交互版模板在 `HTML` 字符串里（纯原生 JS + SVG，无框架）。

注意：test split 的封存口径是"不用于模型选择"；本工具只读标注做可视化，
`--include-test` 不违反该口径。
