# 2026-11-03：借鉴 Thor 的三项融合优化

在 ae77ed3 的全部已验收优化之上，完成三个独立开关，先验证正确性，再选择组合配置。没有引入 FP4/FP8，没有重新转换或修改模型权重。

|改动|开关|实施边界|
|---|---|---|
|KV 转换直接写尾部缓存|VLA_PI05_KV_DIRECT=1|保持 F32→F16 舍入，删除中间临时张量与多余 D2D|
|GeGLU 直接写 F16|VLA_MUSA_GEGLU_F16=1|复用 Down GEMM 原本的 F16 输入精度，消除独立转换；不是 GateUp GEMM epilogue|
|gated residual + FFN AdaRMS|VLA_PI05_RES_ADARMS=1|MUSA F32 激活 expert 块内融合；保持原运算顺序和 WG256 归约；F16 激活入口保持原路径|

## 最终端到端性能

F16 权重、F32 主激活、两视角、固定预处理输入、Graph；每轮预热 3 次、测量 15 次，A/B/B/A 共每配置每模型 30 次。A 是前轮完整推荐配置，B 在其上开启三个新选项。

|模型|基线均值 ms|本轮均值 ms|下降|P95 基线→本轮 ms|
|---|---:|---:|---:|---:|
|pruned|518.881|512.134|1.30%|519.490 → 512.889|
|full|905.296|886.364|2.09%|906.532 → 887.639|

三部分耗时及完整轮次统计见 [最终交错复测](05_最终交错复测.md)。单项 5 次诊断见 [单项方案与计时](01_单项方案与计时.md)，不将无关阶段的波动归因于单项优化。

## 正确性与拷贝

25/25 CTest 通过，12 项完整回归任务通过。两模型普通/分桶变输入、视角切换、图像 embedding 复用、完整入口 F32/F16 均通过。与已验收基线逐元素相同，max_abs=0，重复推理漂移=0。历史 CPU/MUSA 参考误差未增加。详见 [正确性与 Graph 验收](02_正确性与Graph验收.md)。

Graph 内 D2D：pruned 74→2，full 325→1；每次仍为 3 个 Graph、1 次最终同步，稳态无重新捕获或动态分配。仅报告 Graph 节点/字节，未错误相加 profiler 复用的节点 duration。

## 两种模型的运行配置

```bash
cd /home/skysi/code/vla.cpp
source 优化记录/2026-11-03_借鉴Thor融合逐项实施/推荐环境.sh pruned
# 完整模型将上行 pruned 换成 full
.venv-converter/bin/python scripts/bench_pi05.py --model "$VLA_MODEL" --weight-dtype f16 --act-dtype f32 --flash-attn --warmup 3 --runs 15 --out-dir /tmp/pi05-thor-verify
```

实际图像/文本/状态推理继续使用 scripts/run_pi05.py，并保留相同环境。四个已验证完整调用保存于 full-entry/*/*-command.json；selection.json 保存两模型完整推荐开关。开关默认不强制全局启用，便于回退和对照。

## 记录与适用范围

[方案与文档逐项核对](方案与验证说明.md) 记录已实现、已存在、硬件不适用及后续候选，未把本轮三个改动宣称为文档全部优化。跨层融合、action 边缘融合、swapped GEMM、L2 侧流预取等仍需独立实验。

源码基线 ae77ed3；本轮 patch、库和后端 SHA256 在 源码与构建校验.json，原始样本在各 stage/report.json，完整环境在 command.json。新开关未加入旧 bench 元数据白名单，以 command.json 为完整环境证据。日志、图节点及 patch 的 .gz 归档随 Git；.npy 和 msys 原始报告保留服务器，路径/大小/SHA256 见 外部原始文件清单.json。

仅在 M1000/MP22、当前 MUSA SDK/模型上进行数值与性能回归，没有验证真实机器人任务成功率。没有同步到 Mac。

续轮已执行这里列出的剩余候选并按模型完成取舍，最新配置见 [FlashRT剩余候选验收](../2026-11-03_FlashRT剩余优化逐项验收/README.md)。
