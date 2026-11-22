# D-FINE 矩形位置编码复核（2026-11-22）

- 上游 `956d170` 的 `HybridEncoder.build_2d_sincos_position_embedding(w,h)` 用
  `meshgrid(arange(w), arange(h), indexing='ij')` 展开；特征
  `flatten(2)` 是 `(h,w)` 行优先。
- 实测方形原约定：**前一半通道随垂直（h/频率）变化，后一半随水平（w/时间）变化**。
- 旧补丁在矩形分支把两组顺序写成“w 在前”，与方形约定互换；已改为
  **h 组在前、w 组在后**，并保持行优先。
- 证据：方形 `fixed == orig` 逐位相等；矩形纵移只改前一半、横移只改后一半；
  `[2048,512]` 下 `eval_spatial_size` 缓存 `[1,1024,256]` 与动态网格 `allclose`，
  token 数 = 64×16 = 1024（不是 640×640 的 400）。
