# INT8 恢复、分块实验与加载时 F16 解量化

本轮从暂停的全局 INT32 输出 WMMA 原型继续。结论：共享内存分块和双组预取仍慢于 F16 BLAS；
新增显式 `quant_route: "load_f16"`，可消除 Q8 路径的逐 GEMM 解量化开销，但尚未快于浮点基线。
原目标 Native P50 降低≥20%、P95 降低≥15%、完整软件链路 P50 降低≥15% **未完成**。

## 范围与起点

- 生产 `vla.cpp/main` 为 `67f47a07c48c29ac276e0e74122733b94b127a84`；量化工作区为
  `/home/skysi/code/vla.cpp-quant`，分支 `codex/pi05-low-precision`，起点 `3d621a330a2e0ae53001f96223a16f13c8418129`。
- 离线接入使用 `robot_runtime-quant` 的 `396f10a`；只借用生产 runtime 的 Python 环境和冻结观测。
  生产 runtime 的推理计时及其他未提交修改保留。生产配置和 libvla 的起始 SHA256 在 `start-state.json`。
- 机器人/摄像头没有打开，没有 CAN 发送、使能或动作命令。回放持有生产 runtime 的 HardwareLease；
  专用配置 `execute_robot=false`、`dry_run=true`、设备均为 null。
- M1000，驱动 5.1.1-M1000，SDK 5.1.0，mp_22，CPU performance。
  编译器、依赖版本和 GPU 状态见 `environment.json`；隔离依赖补丁和构建参数见
  `llama-build.patch`、`CMakeCache.txt`。没有改生产依赖或库。
- 模型、语料、原 checkpoint BF16 参考沿用
  [上一轮](../2026-11-04_低精度实现与验收/README.md)，未重新量化或复制大模型。
  BF16 精度参考与运行时浮点性能基线分开。模型及输入 SHA256、噪声/任务/token 哈希、
  二进制 SHA256、源码差异指纹在各模型报告内。

## INT8 微内核：保留负结果

每种版本 11 组形状/分块，3 次预热、20 次计时，串行运行。
定时包含动态激活量化、INT8 GEMM、INT32→F32 缩放；权重预打包在计时外。
对照包含 F32 激活→F16 转换和 F16 BLAS。小形状全量、其余最多4096点与相同 W8A8 数学的 CPU 参考比较，
各组失败数为0。这只证明量化内核算术，**不证明原模型动作精度**。
下面列出大形状的最佳分块 P50（ms），对应浮点对照范围来自三种版本的获选配对样本。

| 形状 K×M×N | 共享内存，无 padding | 共享内存，padding=16 | 全局读取，预取2组 | 对应 F16 BLAS |
| --- | ---: | ---: | ---: | ---: |
| 2048×32768×320 | 32.375 | 29.898 | 16.117 | 6.652～6.916 |
| 16384×2048×320 | 32.860 | 32.191 | 18.105 | 5.797～6.020 |
| 1024×8192×50 | 1.546 | 1.329 | 0.750 | 0.557～0.578 |
| 1152×3456×256 | 3.877 | 4.342 | 1.970 | 0.370～0.423 |

- 无 padding 源码保存在 `shared-unpadded.cpp`，输出在本目录；带 padding 版是
  `tests/bench_musa_row_quant_shared.cpp`，输出在 `padded/`。
- 全局预取版是 `tests/bench_musa_row_quant_prefetch.cpp`，输出在 `prefetch2/`。
  每组原始20次时间在 `gpu_samples_ms`，命令、返回码在各 `shared-screen.json`。
- `shared-resource-object.txt`、`prefetch2-build.txt` 的资源报告无 spill/local memory。
  不能把共享内存版失败仅归因于寄存器溢出；未完成指令级瓶颈归因。
- `tests/probe_musa_int8_blas.cpp` 另测 NN/NT/TN/TT、INT32/FP32输出、
  GemmEx 两种算法及 Lt，共24组，实际执行全部 status=8 (NOT_SUPPORTED)。
  Lt heuristic 返回成功并给出候选不等于实际运算支持。见 `int8-blas-layouts.txt`；
  `checks=0` 不计作精度通过，也不能据此断言 GPU 不支持 INT8。
- 失败的链接命令（缺少 musart）、动态库路径以及不支持设备汇编文本输出的日志保留在
  `shared-build-link-failed.txt`、`shared-small-env-failed.txt`、`shared-resource.txt`。
  后续显式链接 musart、设置 LD_LIBRARY_PATH、用对象编译资源报告继续验证。
- 三种原型均未接入正式模型，不能用单 tile 数学正确替代整模型性能/精度验收。

## 实现：加载时一次性解量化

默认 `auto/blas` 保持 packed 量化权重。新增可选 `load_f16`，要求 runtime.weight_dtype=f16，
加载时通过现有 GGUF 转换器生成 F16 驻留 GEMM 权重；显式指定 packed 类型仍保持 packed。
普通/融合加载、直接与暂存上传路径均由测试覆盖 Q8_0、Q4_0、Q5_0。

公共 API 显式管理 `VLA_LOAD_QUANT_F16`，基线清除继承值，进程选项仍不能混用。
resolved_parameters 同时记录磁盘类型、量化 GEMM 驻留规则和 quant_route。
新配置只在本实验目录，仍为 experimental。

这是**存储量化、F16 计算**：牺牲量化 GEMM 权重的驻留显存节省，换取省掉每次 GEMM 转换。
不属于 INT8 计算加速，也不承诺加载更快。Q8 本轮加载均比浮点基线慢。
日志中的驻留权重：pruned 的 packed Q8 为3.12 GiB，load_f16与基线均4.71 GiB；
full 分别为3.47 GiB和5.05 GiB。此处是驻留权重大小，不是进程峰值显存。

## 双模型配对开发集

每模型同30个开发观测，按脚本对每个实际语言长度桶预热3次；pruned保持3步、
full保持10步；每观测50个7维动作。所有候选保留相同输入、任务、token、采样噪声。
每条路径一个独立进程。pruned顺序 baseline→BLAS→load_f16；full顺序 baseline→load_f16→BLAS。
6个进程均正常退出。各1500个逐时刻向量最低余弦均≥0.999，无失败向量。

| 模型 / 路径 | Native P50 / P95 ms | Policy predict P50 ms | 加载 s | 最低逐时刻余弦 |
| --- | ---: | ---: | ---: | ---: |
| pruned / baseline | 479.086 / 489.307 | 483.587 | 20.576 | 0.999965999 |
| pruned / q8-prefix-mlp-blas | 665.755 / 678.495 | 672.240 | 28.398 | 0.999963318 |
| pruned / q8-prefix-mlp-load-f16 | 501.727 / 513.504 | 505.909 | 29.511 | 0.999963318 |
| full / baseline | 809.427 / 839.866 | 814.316 | 21.938 | 0.999977659 |
| full / q8-prefix-mlp-load-f16 | 823.769 / 837.398 | 828.530 | 38.185 | 0.999988962 |
| full / q8-prefix-mlp-blas | 959.767 / 1025.925 | 966.052 | 28.000 | 0.999988962 |

相对本轮 Q8 BLAS，新加载路径的 Native P50 降低约24.6%（pruned）和14.2%（full）；
但相对本轮浮点基线分别**慢4.73%和1.77%**，未达到既定性能目标。
串行单轮样本存在时钟/温度波动，不能将小幅差异解释为稳定收益。

`action-equivalence.json` 对完整30×50×32数组核对：
两模型本轮基线与旧基线逐元素相同；load_f16 与同轮 Q8 BLAS 也逐元素相同。
逐时刻精度仍独立对原 checkpoint BF16 参考计算。
`*-comparison.json` 包含全部门槛判定；`*.actions.npz` 保留原始动作。
旧轮历史两次退出内存损坏仍未定因，本轮正常退出不代表问题修复。
只做开发筛选，未使用396个留出观测，未做真机闭环或部署验收。

## 复现入口与原始文件

编译原型的公共参数：
`/usr/local/musa/bin/mcc -x musa -O3 -fsigned-char --offload-arch=mp_22 -I/usr/local/musa/include <source> -L/usr/local/musa/lib -Wl,-rpath,/usr/local/musa/lib -lmusart -lmublas -lmudnn -o <binary>`。
原始源码/输出映射见上述各版本说明；运行需 `LD_LIBRARY_PATH=/usr/local/musa/lib`。
资源报告额外使用 `-res-usage`，共享内存版使用 `-c` 输出对象文件。

`run_shared_screen.py` 记录并执行11组命令；可通过参数指定输出子目录和二进制名。
`run_load_screen.py` 执行6组配对整模型实验并对照 BF16，配置见 `pruned/full-load-policy.json`。
`run_validation.py` 顺序构建、CTest、Python 单测和 `run_runtime_replay.py`。
复测应在新的归档副本执行，避免覆盖本轮已提交数据。

## 本轮回归与保护核验

- 完整构建通过；CTest 39/39 通过，未跳过 GPU 测试；Python 18/18 通过。
  详见 `regression-build.txt`、`regression-ctest.txt`、`regression-python.txt`。
- 两模型各3次预热+3次 runtime CLI 回放正常退出，明确解析到 load_f16、F16驻留和 experimental；
  `source_contract_verified=false` 保持原值。只作为适配器 smoke，不用3样本替代配对性能验收。
- 生产 main 仍为干净的67f47a0；4份生产配置及生产 libvla.so 与起点 SHA256 一致；
  robot_runtime-quant 无源码改动。见 `preservation-check.json`。未向机械臂下发动作。
- 整模型测量后仅调整源文件注释/文档并重新完整构建；测量时二进制与源码指纹以各模型原始报告为准。

| runtime 回放 | predict P50 ms（3样本） | 动作形状 | 实际路径 / 状态 |
| --- | ---: | --- | --- |
| pruned | 466.487 | 50×7 | load_f16 / experimental |
| full | 758.986 | 50×7 | load_f16 / experimental |

## 后续仍需做的工作

1. 对 INT8 原型实际热点做指令/带宽分析，区分矩阵 load、MMA、量化与输出开销；
   不继续仅凭理论吞吐或修改 tile 参数宣布加速。
2. 候选必须先在真实 prefix/down/expert 形状上计入动态量化和缩放后胜过 F16，
   再进入整模型；当前 W8A8 原型尚未验证原 checkpoint 动作精度。
3. 整模型通过精度与性能双门槛后，再扩大开发集、运行锁定留出集与完整 runtime 链路。
   真机配置和生产 baseline 继续保留；本记录不构成部署通过。
