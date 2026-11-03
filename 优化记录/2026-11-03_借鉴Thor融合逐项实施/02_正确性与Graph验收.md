# 正确性与 Graph 验收

- CTest：25/25 通过，GPU 测试没有跳过。完整输出见 ctest-detail.log.gz。
- 新增 GeGLU F16 输出与独立 cast 比较 max_abs=0，包含 H=4096/4097/16384、行数 3/50、交换 gate/up 和三次 Graph 执行。
- 新增 residual+AdaRMS 与两个旧 GPU 算子比较 max_abs=0，覆盖 H=1/33/257/1024/1025、行数 1/50、分支输出 F32/F16；另与 CPU 公式比较。
- KV direct 测试覆盖 prefix=17/278/518、mask 开/关及 Graph 重放。
- 两模型四个变输入测试（普通长度/桶16）全部通过，变化图片/文本/状态/噪声，回到旧输入仍输出逐值相同。
- 两模型视角 1→2→1→2、encode_images 后 predict、预计算 embedding 入口和变图像回退均通过。
- 两模型 × F32/F16 完整入口四组，与前轮已验收动作输出逐元素相同，max_abs=0；保持原分词及相机顺序。
- 相对历史 CPU/MUSA 参考的误差未增加；不放宽原门限。完整 CPU 误差见 summary-full-entry.json。

## Graph 拷贝核验

|模型|配置|Graph 数|同步数|Graph 内 D2D 节点|Graph 内 D2D 字节|H2D 次数/字节|D2H 次数/字节|
|---|---|---:|---:|---:|---:|---|---|
|pruned|baseline|3|1|74|4202496|4 / 1259776|1 / 6400|
|full|baseline|3|1|325|12488704|3 / 1259776|1 / 6400|
|pruned|final|3|1|2|2359296|4 / 1259776|1 / 6400|
|full|final|3|1|1|4194304|3 / 1259776|1 / 6400|

每模型每配置核验连续 3 帧，稳态无 malloc/free/GraphInstantiate；外部 D2D=0。Graph 内剩余 D2D 是视觉布局相关。

msys 的 Graph 节点导出可能复用时间戳与 duration，因此这里只比较节点数和字节，绝不相加成拷贝耗时。Profiler 窗口包含分析器扰动，不用于最终性能表。

正确性边界：上述是数值回归与 API/Graph 生命周期验证；没有进行真实机器人任务成功率测试，也不声称任意未来权重、硬件或 SDK 下自动成立。
