# M1000 搬运、视觉计算与三段 Graph 迭代

起点提交 `dfc0da1`。本轮完成 9 组优化实验，并在最终配置下交错复测；代码与本文同一提交。只在服务器归档，不同步到 Mac。原始 checkpoint、GGUF、15/27 视觉层、136/256 token、3/10 去噪步、chunk=50 均保留，没有 FP4 量化。

## 最终确认的性能

M1000，performance 模式，4 CPU 线程，F16 权重 / F32 逻辑激活，FA 和 Graph 开启。两路 224×224 合成图、6 个固定文本 token、固定 state/noise。加载与预热不计入延迟，无 profiler 并发。

最终按 baseline-r1 → final-r1 → final-r2 → baseline-r2 的顺序，每组都测两个模型，每个进程预热 3 次、测 15 次。下表合并各自两轮的 30 个正式样本取中位数；没有把不同阶段的最低值拼成最终结果。

| 模型 | 起点 ms | 最终 ms | 节省 ms | 延迟降低 |
|---|---:|---:|---:|---:|
| pi05_cube_merge_cksize50_step3_0929 | 748.035 | 719.617 | 28.419 | 3.80% |
| cup_cube_joint_061_a800_50k_20261022 | 1532.608 | 1508.142 | 24.466 | 1.60% |

两轮最终整次延迟分别为裁剪 719.525 / 719.703 ms，完整 1508.929 / 1507.265 ms。基线两轮为裁剪 748.493 / 747.771 ms，完整 1530.390 / 1534.185 ms。最终视觉中位数为 261.531 / 420.459 ms，原基线为 289.927 / 443.312 ms。收益主要来自视觉，前缀与动作计算没有明显加速。

没有将本轮结果称为硬件理论最优；这是本轮已测方案中兼顾正确性、三段捕获和稳定延迟的配置。

## 三部分耗时

最终配置另以诊断分段模式预热 3 次、测量 7 次。该模式在每段完成后同步，正常执行则只在前缀与动作之间取消同步，因此不能把诊断延迟当作无扰动 GPU kernel 时间。

| 部分 | 裁剪（3 步）ms | 完整（10 步）ms |
|---|---:|---:|
| 视觉与图像 token 准备 | 261.523 | 420.875 |
| 多模态前缀 / KV | 299.735 | 486.877 |
| 动作去噪全部步骤 | 158.444 | 604.480 |

诊断整次 wall time 为 720.475 / 1512.741 ms，比正常最终 30 样本中位数约高 0.12% / 0.31%。各项独立取中位数，不要求相加严格等于整次延迟。视觉包含预处理、token 准备和搬运，前缀包括图像和文本，动作包括原始全部步骤。

## 逐轮方案和结果

以下每组预热 3 次、测 7 次。它们用于筛选，最终结论以上面的交错复测为准。各阶段保存 source.patch、command.json、runtime.json、report.json、run.log 与动作数组。

| 阶段 | 方案 | 裁剪整次 / 视觉 ms | 完整整次 / 视觉 ms |
|---|---|---:|---:|
| baseline | 本轮起点复测 | 748.577 / 290.667 | 1533.279 / 442.648 |
| s1-device-prune | GPU 稳定选 token / gather | 740.983 / 282.910 | 1535.369 / 443.385 |
| s2-shared-attention | 复用最后视觉层评分；KV 复用关闭 | 732.349 / 275.117 | 1533.992 / 443.253 |
| s3-kv-reuse | 前缀 KV 首次拼接后仅覆盖动作尾部 | 733.471 / 274.808 | 1531.570 / 443.298 |
| s4-batch-vision | 关闭 KV 复用；完整模型双相机批处理 | 733.965 / 274.814 | 1514.215 / 420.506 |
| s5-prune-flash | 裁剪前 14 层 FA，最后层保持评分 | 717.813 / 258.605 | 1537.024 / 442.744 |
| s6-f32-tile | 全视觉 FP32 分块；guard 迁移故障，未捕获 | 763.459 / 305.074 | 1519.287 / 421.811 |
| s7-mixed-vision-graph | 前 14 层 FA + 最后层 F32 分块；视觉捕获 | 723.514 / 262.654 | 1513.662 / 423.051 |
| s8-three-graphs | 三段 Graph；前缀/动作异步串联 | 719.883 / 261.732 | 1508.659 / 420.633 |
| s9-tile-k32 | 较大 K 分块，无收益，撤回 | 721.619 / 264.164 | 1511.464 / 420.997 |

- s1/s2 以及完整模型批处理的输出对本轮起点逐元素一致。评分计算本身保持 mean(head) 再 mean(query)。
- s3 裁剪略慢、完整仅有约 2 ms 的变化，不能认定稳定收益；默认关闭 KV 复用，仅留下显式实验开关和单元回归。没有把它写成最终加速来源。
- s5 开始，裁剪前 14 个视觉层采用 FA，K/V 为 F16；最后一层保留完整 F32 attention 权重用于选 token。裁剪输出因此不再逐元素相同，误差见后文。
- s6 的全视觉 F32 自写 GEMM 比原 muBLAS 路径慢，不采用。该次还遇到补丁升级重复 guard，实际没有捕获视觉图；保留原始失败状态，没有将它当作 Graph 性能。
- s7 仅最后评分层采用可捕获的 F32 GEMM，其余层使用 FA；两个模型的视觉都能捕获。
- s8 将前缀、动作拆为独立 Graph，同一后端 stream 先后异步提交，再统一等待。视觉阶段仍有同步和少量搬运，并非整条管线零同步或零拷贝。
- s9 增大 K 分块没有改善，最终恢复 K=16。s5 的裁剪单组数值略低于 s8，但不是交错对照；最终选择全部三段可捕获的配置，并以 30 样本复测确认。

## 实现与内存搬运

1. **GPU token 选择**：对最多 256 个视觉 token 进行稳定排名，分数相同优先较小索引，再按原空间顺序 gather。消除完整视觉 embedding 的 GPU→CPU 读回、CPU 拼装和重新上传；新路径仍有 GPU 内部输出搬运。超过该支持范围或显式选择 host 路径时保留原实现。
2. **复用最后层评分**：从真实视觉 attention 分支导出权重，取消额外的 norm/Q/K/softmax。没有用另一个近似指标替代裁剪规则。
3. **双相机批处理**：完整模型的两路视觉计算一起提交，保持相机次序及各自的 token 布局。
4. **可捕获 F32 attention**：小型、连续、同批次的 F32 QK/AV 采用 16×16 分块内核。保留 F32 运算，避免 muBLAS 内部 capture 时同步；其他形状继续原实现。CPU 参考单元测试覆盖 72 维头、256 token 和尾部维度。
5. **三段执行**：固定分配的完整 GGML 图继续管理张量生命周期，切片为前缀和动作两个执行范围；图依赖决定顺序，不把有依赖的阶段误作并行。
6. **KV 实验**：每层首次 concat 后固定缓冲区，后续只更新动作尾部；测试覆盖三步、不同前缀长度、mask 和捕获重放。未证明稳定速度收益，默认关闭。

## Graph 正确性修复与证据

s2 首次 CTest 出现 BF16 小图异常，重跑通过。调查发现 Graph 属性比较记录了输入指针、尺寸、stride，却没有记录输入 dtype；同一地址被 F16/BF16 复用时，可能错误复用旧 executable。

新增固定地址 F16→BF16→F16 定向测试。旧 dfc0da1 动态库在 BF16 段峰值归一化误差 7.734；补上输入 dtype 比较后误差 0.000700（CPU 参考），捕获、重放和实际变更重新捕获均通过。证据为 dtype-old.log、dtype-fixed.log。最终 16 项 CTest 连续执行 3 次，全部通过，没有 GPU 跳过。

补丁同时验证了干净 pinned llama 源码、从 dfc0da1 旧补丁升级、重复应用幂等性，并确认生成文件与实际构建依赖逐字一致。patch-validation.log 保留结果。中间重复 guard 的失败也记录在 s6，最终已修复。

官方 msys 对两个模型分别执行 4 次 predict（2 warmup + 2 measured）：
- 每个模型都是 3 次 Begin/EndCapture、3 次 GraphInstantiate、9 次 GraphLaunch。
- 最后一次实例化之后，均有 7 次 GraphLaunch，0 次直接 musaLaunchKernel。
- 前后仍存在输入/输出搬运和同步 API，未宣称它们全部消失。

详见 summary-graph-traces.json、trace-pruned/api_musa_api_trace.json.gz、trace-full/api_musa_api_trace.json.gz。msys 仅验证调用路径，不用于正式延迟结论。

## 精度和入口验证

- 最终合成基准：裁剪对本轮起点峰值归一化误差 0.35894%，对原 MUSA F32 参考 0.43543%；完整对本轮起点逐元素一致，对原 MUSA F32 参考 0.19086%。均低于沿用的 0.5% 阈值。
- 变换输入：两模型各执行 12 次，覆盖同长度换文本、改变图像/state/noise、6→9→6 token 重建与回切。每种输入重复漂移为 0；裁剪三个输入对基线误差均低于 0.36%，完整为 0。
- 完整 run_pi05.py 入口：使用保存的 480×640 图片、真实 tokenizer、实际相机顺序和 50×7 输出，两个模型分别验证 F32/F16 激活。43 / 42 个文本 token；对 CPU 参考峰值归一化误差分别为裁剪 0.15334% / 0.14896%，完整 0.12191% / 0.14455%。每组 2 次预热后输出漂移为 0。
- 选 token 单测覆盖并列分数、小尺寸、全部保留和 Graph 重放；最终 CTest 共 16 项 × 3 遍。

这些是数值与接口回归，不是原训练框架全策略对照或机器人成功率验收。

## 使用与开关

本机经过验证的库是 `/home/skysi/code/vla.cpp/build-musa-official/libvla.so`。转换产物不需要重新生成。新建环境仍按 scripts/build_musa.sh 构建；它默认生成 build-musa，因此 VLA_LIBRARY 应指向实际构建目录。

```bash
cd /home/skysi/code/vla.cpp
export VLA_LIBRARY="$PWD/build-musa-official/libvla.so"
export VLA_DEVICE=0 VLA_N_THREADS=4
unset GGML_CUDA_DISABLE_GRAPHS GGML_MUSA_DISABLE_GRAPHS
.venv-converter/bin/python scripts/bench_pi05.py   --model /home/skysi/VLA/models/pi05_cube_merge_cksize50_step3_0929/gguf/model.gguf   --out-dir /tmp/pi05-new-benchmark --weight-dtype f16 --act-dtype f32   --flash-attn --warmup 3 --runs 15
```

替换模型路径即可验证完整模型。加 --split-phases 输出诊断三部分时间。所有环境开关在进程启动前设置。

| 开关 | 用途 |
|---|---|
| VLA_PI05_HOST_PRUNE=1 | 保留 CPU 选 token 与搬运路径，便于回归 |
| VLA_PI05_NO_PRUNE_FLASH=1 | 裁剪视觉所有层恢复普通 attention |
| VLA_PI05_NO_VISION_BATCH=1 | 完整视觉恢复逐相机计算 |
| VLA_PI05_NO_SEPARATE_GRAPHS=1 | 恢复前缀与动作合并主图 |
| VLA_MUSA_F32_ATTN_TILE=0 | 禁用新增 F32 分块；不兼容的视觉图会回退普通执行 |
| VLA_PI05_KV_REUSE=1 | 显式启用未证明稳定速度收益的 KV 实验 |
| VLA_PI05_NO_KV_REUSE=1 | 强制关闭上述 KV 实验 |

历史 s4–s9 的正向实验开关和最终默认设置不同，应以各阶段源码差异、命令及本文说明配套复现，不拿最终环境开关假装还原已改变的旧版本。

## 归档与后续空间

environment-final.json 记录基线/最终库哈希、环境和依赖版本。原始动作数组及 msys-rep 留在服务器，external-artifacts.json 记录路径、大小和 SHA-256；manifest.json 校验本目录的原始记录文件（不含自身）。

当前仍有视觉边界同步、输入上传、GPU 内部布局搬运及 KV 拼接。后续更大的收益需要针对前缀/动作 GEMM 与内存访问继续优化，单纯再拆 Graph 不能替代这些计算。本轮没有承诺未测量的收益。

提交前复查补充：双相机输出布局调整限定到 MUSA，分段开关在进程内固定，避免影响其他后端或运行中切换造成图边界不一致。这不改变已测 MUSA 配置的计算图；另以 ctest-scope-guard.log 和 final-scope-check/ 保存最后构建回归。environment-measured.json 保留交错复测时的库哈希，environment-final.json 对应交付构建。

较大的原始 API 跟踪和完整后端编译日志以 gzip 无损压缩保存；compressed-artifacts.json 同时记录解压后的原始字节数和 SHA-256，内容未删减。可用 gzip -dc 查看。
