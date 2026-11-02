# 2026-11-02 官方文档再核查

本轮性质：重新阅读官方资料、对照源码及实机能力，不修改推理算法，不声称新增性能收益。
核查代码基线：`6543803`；服务器：`skysi`，仓库：`/home/skysi/code/vla.cpp`。
记录仅存服务器，并随 Git 提交，不同步到 Mac。

## 1. 结论

此前已解决 Graph 捕获、GPU 裁剪、部分布局和融合问题，但计算内核本身仍有未充分验证的空间。
最值得优先补的是：真实形状 GEMM 算法搜索、M1000 适用的 Attention 内核、视觉投影与输出阶段融合。
另外，集成内存输入路径和残余同步还有空间。是否值得保留，必须分别用两个模型验证。

这不是首次提出所有方向：10 月 30 日调研已列出 muBLASLt、内存和分析工具等建议；
本轮补上“实际尚未实施什么”“硬件能否支持”“如何避免照搬不适用的官方示例”的核查。
不能因为开启了 F16、FlashAttention 或 Graph 就认定已经充分使用矩阵计算单元。

## 2. 已完成与仍有缺口

| 方向 | 当前状态 | 本轮核查出的剩余工作 |
| --- | --- | --- |
| 三段 Graph | 视觉、前缀、动作去噪均已捕获并重放；修复 dtype 缓存失效 | 视觉结束、跨图输入复制、主机上传仍有同步；没有做到只在最后同步 |
| 低精度 GEMM | F16 权重、部分低精度输入和原生 F16 路径已有 | 普通 GemmEx/批量 GemmEx 默认算法，不是按真实形状搜索 muBLASLt 算法 |
| FlashAttention | 已避免大部分完整注意力矩阵；MUSA 分派走通用 TILE | 未接入 MUSA 专用 MMA Attention 路径；未测本机 muDNN SDPA |
| 视觉融合 | 完整模型已批量处理两摄像头；裁剪模型最后一层复用打分 | MUSA 视觉 Q/K/V 仍分开投影；bias、GELU、LayerNorm 等还有融合候选 |
| 内存复制 | GPU topk/get_rows 去掉裁剪中间量 CPU 往返 | 主机临时 vector、CHW 再复制、同步 tensor_set、D2D 衔接仍存在 |
| KV 重用 | 已试验、开关可选 | 上轮没有稳定端到端收益，默认关闭；不能重复记为新收益 |
| 分析工具 | msys 已验证 Graph API/重放 | 仍缺可信的逐热点硬件利用率、带宽、访存停顿、资源压力分析 |

## 3. 实机能力核查

原始探针：`probe_capabilities.cpp`；输出：`capabilities.log`；命令和版本：`environment.log`。

| 项目 | 实机查询值 | 对优化的意义 |
| --- | --- | --- |
| 架构 / 多处理器数 | cc 2.2（MP22）/ 8 个 MP | MP22 是架构代号，不是 22 个多处理器 |
| warp 大小 | 128 | 不能直接套用 S5000 示例的 32 线程假设 |
| 每 block 共享内存 / opt-in | 73728 / 73728 字节（72 KiB） | 文档 MP31 的 192 KiB 分块示例不适用 |
| integrated / canMapHostMemory | 1 / 1 | 可试固定映射输入缓冲；支持映射不代表一定比复制更快 |
| asyncEngineCount | 1 | 多流不能被理解成无限传输重叠能力 |
| l2CacheSize / persistingL2CacheMaxSize | 0 / 0 | 当前驱动未报告可用持久化容量，不按 L2 set-aside 作为可直接落地方案；不据此断言没有物理 L2 |
| accessPolicyMaxWindowSize | 1073741824 | 单独有窗口上限不等于持久化 L2 功能可用 |
| 系统包 | musa 5.1.1-M1000；musa-sdk 5.1.1 | 与 M1000 官方适配指南的包版本要求一致；`/usr/local/musa` 链接仍名为 musa-5.1.0 |
| 分析工具 | mcu 1.2.2 | 使用本机支持的参数、报告；不照抄指南中 NVIDIA ncu 命令 |
| muBLASLt | 启发式算法查询和算法 ID 接口可动态加载 | 可开展真实形状实验；尚未证明特定形状速度或 Graph 兼容性 |
| muDNN | FlashForward、SDPA descriptor 接口可动态加载 | 可先做 FP16/BF16、head_dim、mask、GQA、布局和 Graph 能力微测 |

加载探针曾出现两个环境问题：Lt 单独 dlopen 时缺 muTensorDescriptorEncode；muDNN 找不到其依赖库。
本轮通过全局加载实际驱动 libmusa.so.1，并在子进程设置 LD_LIBRARY_PATH=/usr/local/musa/lib 后成功。
该现象不是缺少驱动符号，也不能据此宣称库不可用。不修改系统环境或升级驱动。
符号可用仅证明接口存在，不证明目标内核能在 MP22 上执行或获得加速。

## 4. 按优先级安排的未完成实验

### P0：先量清楚当前热点

- 从两个模型各三个部分提取 kernel 累计耗时、调用次数和 GEMM 的 M/N/K、dtype、转置、stride、批量数。
- 用 msys 看关键路径；用本机 mcu 对代表性热点采集可用的执行/访存/资源指标，区分计算、带宽和启动开销。
- 精确时长使用设备事件和不带 profiler 的端到端结果交叉核对。不能把 profiler 的运行时间当正常推理时间。
- 普通路径测端到端；分段同步模式只用于解释阶段占比。没有实际计数器证据前，不宣布“已经跑满带宽”或“Tensor Core 利用率多少”。

### P1：真实形状 muBLASLt 搜索

官方 Lt 文档提供算法启发式、工作区、布局和 epilogue 配置。当前实现尚未系统使用。

- 在初始化时查询多个候选，按实际事件计时选择；测试 0/8/32/64 MiB 等有限工作区，不盲目使用第一条启发式结果。
- 重点覆盖动作 50 个 token 的小矩阵、两种长度前缀和批量视觉矩阵；实际语言长度要单独纳入。
- 分别比较 FP16 输入 + FP32 累加/输出，以及现有原生 F16 路径；不要为了速度默认降低所有累加精度。
- 算法缓存键覆盖形状、转置、leading dimension、batch stride、dtype、epilogue、workspace、库版本和 GPU 架构。
- 工作区和描述符在捕获前创建，复用地址；单独验证 capture/replay 与输入变化。
- 之前小尺寸 FP32 Lt/BLAS 捕获失败，不足以排除所有低精度 Lt 算法。
- 候选实验：仅对合适的 GEMM 行数做 50→64 padding，再正确切片。若涉及 Attention，必须屏蔽补齐 token，保持 chunk50、去噪步数和模型语义。额外计算可能抵消收益。
- 权重打包仅初始化一次，不计入稳态重复计算；同时报告初始化和额外内存成本。

### P1：适用于 MP22 的 Attention

当前 ggml 的 volta/turing/ampere MMA 与 cp_async 判断限定 NVIDIA，其他 MMA 分支限定相应 AMD 架构。
MUSA 通用 TILE 路径并不等于已有 MUSA 专用矩阵单元 Attention 内核。

- 先验证本机 muDNN SDPA 对真实 head_dim（视觉 72、Gemma 256）、短序列、非因果/加性 mask、共享 KV、布局的支持。
- 若支持，比较其数值、Graph 捕获和总耗时；若不支持，保留现路径，评估 MP22 专用 kernel。
- 定制 kernel 按 72 KiB 共享内存、warp128、寄存器压力和真实短 Q/K 长度调整 tile、向量访存、数据复用和流水线。
- 动作去噪 Q=50、前缀几百 token，不照搬文档 N=4096 的长序列块大小。
- GQA 重点测共享 KV 的复用和 head 调度，不能未经检查断言现实现完全没复用。
- 裁剪模型最后一层必须产生打分，不能直接用只输出 O 的黑盒 FA 替代。可另试带打分归约的专用融合路径，但需要比对 topk 排序、边界和最终动作。
- 上轮 K32 自定义 tile 更慢已撤回，不意味着所有 tile/向量化组合都已经搜索完。

### P2：视觉 QKV 与 GEMM 输出阶段融合

`src/models/pi05.cpp` 中视觉 `cfg.fused` 受 SYCL 条件限制；MUSA 当前仍使用分别的 Q/K/V 权重与投影。
`src/modules/encoder.cpp` 可见普通路径三个 linear，以及 FFN 的 bias/GELU。

- 加载时拼接 Q/K/V 权重和 bias，一次 GEMM 后按视图分开，减少启动次数和相同输入读取。
- 保持两摄像头 batch 与裁剪打分语义；先做 F16 权重、F32 逻辑激活基线兼容版本。
- 测 GEMM+bias、GEMM+bias+GELU epilogue，减少中间读写；准确核对 GELU 近似公式。
- GEGLU 的门控乘法、AdaRMS、残差不同于普通 GELU，不能用单一 Lt GELU epilogue 替代已有语义。
- 再考虑 LayerNorm+scale+bias 和 dtype/layout 转换融合；避免产生多余 F32↔F16 往返。

### P2：集成内存输入路径、持久缓冲和残余同步

- CPU 预处理直接写持久输入缓冲，减少 CHW vector 再复制；缓存固定形状的位置和 mask，形状改变时更新。
- 先试 pinned host buffer + 同流异步上传，再与 mapped host input 对比。映射路径先限输入和小型常量，不把整套权重一律改为 host mapped。
- 将视觉输出 D2D 衔接和输入更新放在有明确依赖的 stream 上；改用异步接口不等于可以无条件删除同步。
- 稳态只在读取最终动作前等待的方案，需要处理 CPU 缓冲生命周期、Graph 固定地址、动态输入和输出可见性。
- 只有存在独立工作才有计算/传输重叠：下一帧输入双缓冲主要影响吞吐/流水线，不自动减少单帧关键路径，也不默认改变控制时序。
- 若消除 D2D，需要固定输出分配/别名契约，避免破坏 ggml allocator 和 Graph 缓存。

## 5. 本轮明确不照搬的建议

| 文档建议 | 此项目的适用限制 |
| --- | --- |
| MATE FlashAttention / 新架构 MMA | 官方 MATE 当前要求 MP31；此机 MP22，不能直接宣称可用 |
| L2 持久化窗口 | 当前驱动报告 persistingL2CacheMaxSize=0；须有实际支持证据再考虑 |
| S5000 warp32 / MP31 大共享内存参数 | 实机 warp128、72 KiB，共享内存和线程布局需重新设计 |
| 直接升级 SDK 5.2 | 5.2 发布说明的验证平台不能替代 M1000 专用兼容要求；本轮不升级 |
| llama.cpp 增大 ubatch | 因果 LLM 的分块不能直接照搬到双向视觉/π0.5 前缀 Attention |
| --no-mmap / drop_caches | 主要涉及加载或操作系统缓存；当前稳态指标不含模型加载，不计作稳态收益，也不运行清缓存命令 |
| 多拆 Graph | 已有三段 Graph；更细拆分会增加提交，不是默认优化方向 |
| 更多 stream | 三个部分有数据依赖，多流本身不会消除依赖；还可能增加事件和同步成本 |

## 6. 已测基线及后续验收

以下复用上一轮记录，不是本轮重测：

| 指标（ms） | 裁剪模型 | 完整模型 |
| --- | ---: | ---: |
| 普通推理端到端，30 样本交错实验最终均值 | 719.617 | 1508.142 |
| 视觉，独立分段诊断 | 261.523 | 420.875 |
| VLM 前缀，独立分段诊断 | 299.735 | 486.877 |
| 动作去噪，独立分段诊断 | 158.444（3 步） | 604.480（10 步） |

分段诊断会增加同步，其和不能冒充普通推理均值。合成性能输入为两张 224 图、6 个文本 token；
实际 tokenizer 的 43/42 token 验证属于另一测试条件，不能混用。

后续每项单独开关、分别记录两个模型阶段/总耗时、内存开销、精度、Graph 重放和变输入测试。
使用同电源/精度/输入/预热条件进行交错 A/B，保留退步结果；收益不稳定的优化不默认启用。
对候选内核先测微基准，再检查整模型及原始 CPU/MUSA 参考、token 长度变化、连续多次重放；
裁剪模型额外比较分数和 topk。没有跑完验证前不合入默认路径。

## 7. 官方资料和源码依据

本轮详细核对以下章节；不是宣称读完全部 SDK API。`sources.json` 保存抓取 URL、日期和提取文本 SHA256；
全文临时阅读文件在 `/tmp/vla-official-review-20261102/`，不把整站内容复制入仓库。

1. [muBLASLt 开发指南](https://docs.mthreads.com/musa-sdk/version-5.1.0/libraries/xlib/muBLASLt_developer_guide/)及[API](https://docs.mthreads.com/musa-sdk/version-5.1.0/libraries/xlib/muBLASLt_api_reference/)
2. [GEMM/GEMV 优化](https://docs.mthreads.com/musa-sdk/version-5.1.0/programming_guide/performance_tuning/gemm_gemv_optimization/)
3. [FlashAttention 优化](https://docs.mthreads.com/musa-sdk/version-5.1.0/programming_guide/performance_tuning/flash_attention_optimization/)
4. [性能瓶颈分析](https://docs.mthreads.com/musa-sdk/version-5.1.0/programming_guide/performance_tuning/performance_bottleneck/)
5. [内存优化](https://docs.mthreads.com/musa-sdk/version-5.1.0/programming_guide/performance_tuning/memory_optimization/)及[计算优化](https://docs.mthreads.com/musa-sdk/version-5.1.0/programming_guide/performance_tuning/compute_optimization/)
6. [高级内存优化](https://docs.mthreads.com/musa-sdk/version-5.1.0/programming_guide/programming_model/advanced_memory/)及[L2 管理](https://docs.mthreads.com/musa-sdk/version-5.1.0/programming_guide/programming_model/l2_cache_management/)
7. [MATE](https://docs.mthreads.com/musa-sdk/version-5.1.0/libraries/openlibs/mate/)
8. [SDK 5.2 发布说明](https://docs.mthreads.com/musa-sdk/version-5.2.0/releasenote/)
9. [M1000 llama.cpp 快速开始](https://docs.mthreads.com/llama.cpp-musa-m1000/Llama.cpp-MUSA-M1000/AIModule/AIModule1.4.1+0.0.9252/quick_start)及[环境要求](https://docs.mthreads.com/llama.cpp-musa-m1000/Llama.cpp-MUSA-M1000/AIModule/AIModule1.4.1+0.0.9252/environment_setup)

代码依据（基线 6543803）：
- `src/models/pi05.cpp`：608 行视觉融合条件；808 行附近 host 图像缓冲；1056–1140 行附近传输和同步。
- `src/modules/encoder.cpp`：37 行附近权重声明；163 行附近分别 Q/K/V 投影。
- `src/musa/vla_musa_fused.cu`：自定义融合、F16 GEMM 和 F32 attention tile。
- `build-musa-deps/llama/ggml/src/ggml-cuda/common.cuh`：360–374 行 MMA/cp_async 架构限定。
- `build-musa-deps/llama/ggml/src/ggml-cuda/fattn.cu`：640–724 行内核选择及通用 TILE 回退。
- `scripts/patch_ggml_musa.py`：依赖补丁；本轮不修改依赖或默认算子。
