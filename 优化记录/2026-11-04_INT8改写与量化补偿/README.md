# 原生 INT8 改写与量化补偿（进行中）

用户目标已建立：两个 π0.5 模型逐观测、逐时刻七维动作与原 checkpoint 实际 BF16 前向的余弦均不低于 0.999；计入动态量化、补偿和预后处理后，整模型及软件链路相对当前浮点运行时有可重复性能提升。目标未完成，未合并、部署或下发机械臂动作。

## 范围与参考

- 工作区：/home/skysi/code/vla.cpp-quant，codex/pi05-low-precision；本轮起点 d97946b。生产 main 保持 67f47a0。
- M1000 MP22，warp 128，MUSA SDK 5.1.0、驱动 5.1.1-M1000，CPU performance；配置详情见 start-state.json及原始日志。
- 精度参考：低精度实现与验收目录的两份 bf16-reference-30.json 及动作数组。是真实 BF16 前向，保留原框架的视觉/归一化 FP32 例外。
- 性能参考：相同输入和运行配置的 F16 权重/F32 激活/F32 累加运行时。精度参考与性能参考不得混用。
- 数据：real-v1，120 calibration、90 development、396 heldout，按 episode 分离。本轮动作筛选使用前 30 个 development 观测；尚未完成全部 development / heldout 验收。
- 每次 GPU 实验获取 robot_runtime HardwareLease(model-replay)。无机器人、相机或 CAN 访问。生产 robot_runtime 未提交改动保留。

## 实现

实现显式 native_int8 precision route，仅将 18 层专家 GateUp 投影改为原生 W8A8、INT32 累加。其余投影仍为浮点；不是全模型纯 INT8。参数由原 F16 驻留权重和校准统计导出；加载时转码成 MP22 原生 fragment 布局，常驻权重缩小约 0.14 GiB，加载时间增加约 20–27 秒。精度 profile 仍为 experimental。

- 动态逐 token 激活量化、逐输出通道权重尺度、SmoothQuant 风格通道缩放。
- AWQ 风格平均绝对激活尺度搜索，用于算子实验；这是适配 W8A8 的实现，不是完整官方 AWQ 权重量化流程。
- 可选 16/32 个离群通道保留 F16，直接在原生输出 epilogue 补偿。增加计算成本，实测不宜作为默认候选。
- manifest 检查原模型 SHA256、校准 split、完整 18 层和逐文件哈希。原生二进制参数检查尺寸、有限尺度、唯一合法通道及文件长度。
- V1：独立量化与 GEMM。V2：将 GateUp、GEGLU 和 F16 打包融合，省去中间输出及独立激活 kernel。V3：正在验证残差/AdaRMS 同时生成 INT8 激活，保留原有 F16 值及残差。
- 原始输入可按白名单抓取用于校准，默认关闭；捕获运行不计作性能测量。

## 实验结果与边界

fragment-layout.json 验证 MP22 的 128 线程 fragment 布局。32/64 线程尝试数值失败，已保留。直接两个 32-bit 寄存器加载替换通用 fragment 装载；所有 direct-screen 算子数值检查通过。独立累加器 1/2/4 路均数值通过，增加累加器未带来足够收益，未采用。

真实输入算子筛选：每模型独立 4 个 calibration / 4 个 development 观测，保留专家各去噪步。完整 120 calibration 的通道 amax 用于尺度；开发输入不参与拟合。原始 .f32 数据在 validation/quant/int8-compensation-inputs，清单及哈希随本目录归档，未把大输入文件加入 Git。

专家 GateUp 的 SmoothQuant alpha=0.75 将最低算子余弦提高至 pruned 0.999969 / full 0.999961；32 通道补偿进一步提高至 0.999989 / 0.999985，但变慢。前缀两层 AWQ 风格 alpha=0.5 有精度收益，算子仍比浮点慢。**算子余弦不等于动作余弦，单算子速度不等于整模型速度。**

### V1 整模型筛选

每 profile 新进程；每语言长度预热 3 次、30 个 development 输入、原有 3/10 步。所有进程正常退出，所有 profile 的 1500 个动作向量均过 0.999。单轮筛选不足以证明稳定加速。

| 模型 | profile | native P50 ms | baseline P50 ms | 最低动作余弦 |
|---|---|---:|---:|---:|
| pruned | plain | 488.078 | 489.684 | 0.999951405 |
| pruned | smooth | 479.077 | 489.684 | 0.999967083 |
| pruned | compensated | 534.895 | 489.684 | 0.999969202 |
| full | plain | 792.236 | 774.207 | 0.999967515 |
| full | smooth | 809.073 | 774.207 | 0.999978568 |
| full | compensated | 1018.876 | 774.207 | 0.999980178 |

原始报告、动作数组、比较报告、命令和返回码见 native-model-screen.json 及各 profile 文件。full 性能不达标；pruned 单轮 2.17% 收益不能作为稳定结论。

V2 生产子图四轮 AB/BA：F16 GateUp+GEGLU P50 0.593–0.599 ms，INT8 融合 0.557–0.565 ms（graph-gemm-v2.txt）；包括动态量化。整模型结果独立保存 native-model-v2-screen.json，不能用子图结果代替。

## 验证、失败与复现

- native-v2-test.txt：20 组原生整数/补偿形状检查（F32/F16 输入，N=1/50/64/65/161，R=0/16，5 次变化输入），额外验证 GEGLU 与尾部零填充。
- native-python-tests.txt：12 项 policy/补偿测试通过（当时版本）；最终回归结果另记。
- 初版分离 F16 补偿 GEMM 在 graph capture 失败，见 native-test-blas-correction-failed.txt；已改为融合 epilogue。
- capture-build-failed.txt、graph-gemm-build-failed.txt 等保留编译失败；后续成功日志不覆盖失败记录。
- 历史两次 native 退出堆损坏尚未确认根因；本轮正常退出不能宣布已修复。
- run_native_models.py / _v2.py / _v3.py 记录各版本命令。源码快照及 patch 保留历史实现，当前代码不能假装重现所有历史版本。
- run_kernel_followup.py、run_accumulators.py、screen_compensation.py 和各原始 JSON 记录算子参数、命令及返回码。
- native-v1-source、native-v2-source 为对应内核快照；parameters 保存导出的小参数与 manifest，模型权重和构建产物不进 Git。

## 后续验收

继续优化并进行双模型交替复测，冻结候选后扩展完整 development 和独立 heldout 的实际 BF16 配对；完成公共 API / robot_runtime 离线回放及回归检查，确认生产配置与库哈希未变化。没有达到全部门槛前不宣称可部署、稳定加速或目标完成。物理单位和闭环效果仍未验证。

方法来源：[SmoothQuant 官方实现](https://github.com/mit-han-lab/smoothquant)、[SmoothQuant 论文](https://arxiv.org/abs/2211.10438)、[AWQ 官方尺度搜索](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py)、[LLM.int8 论文](https://arxiv.org/abs/2208.07339)。只借鉴校准/分解方法，性能结论来自本机原始测量。

## V2 / V3 阶段补充

| 版本 | pruned 基线→INT8 P50 ms | full 基线→INT8 P50 ms | 结论 |
|---|---:|---:|---|
| V2（GEGLU 融合） | 483.917→484.731 | 799.524→812.405 | 精度通过；整模型未加速 |
| V3（再融合 AdaRMS 量化） | 485.197→490.619 | 796.919→815.620 | 精度通过；整模型未加速 |

两版最低动作余弦均为 pruned 0.999967083 / full 0.999978568。V3 新融合节点与独立量化在 R=0/16、N=1/50/64 的残差及矩阵输出逐元素一致。输出完整数组已归档；尚未作最终独立留出验收。

最终本阶段回归：40/40 CTest 通过；23 项相关 Python unittest 通过（native-v3-python-corrected.txt）。一次全目录 unittest discovery 错误地混合了依赖桩测试和需要 pytest 的测试，在当前环境下发生导入失败并污染 numpy/gguf 模块；原始失败保留 native-v3-python.txt。改为仓库既有显式模块入口，加上新的补偿测试，全部通过；没有为此修改生产 Python 环境。

protected-state-checkpoint.json 核对四份生产配置及 libvla.so SHA256 均未变化，生产 HEAD 仍为 67f47a0。V3 源码快照保留于 native-v3-source，参数异常拒绝测试包含 NaN/Inf/过小过大尺度/重复越界通道/截断及尾随数据。

正在采集 V3 与浮点基线的 MCU 热点（Graph OFF，单次预热后的 development 观测，有独立起止 marker）。该采集受 profiler 干预，只定位瓶颈，不用于宣称性能收益。split-k-prototype.cpp / run_split_k.py 是下一轮待执行的 K 维拆分原型，尚无测量结论。

## 分段归约与采集范围调整

split-k-screen.json：50 个独立算子 case（SplitK=1/2/4/8/16，3 个实际专家形状和小形状，20 次计时、3 次预热）均通过整数数学检查。计入动态量化和 INT32 部分和归约后没有值得接入模型的性能收益；专家 Down 最佳约 0.853 ms，同行 F16 BLAS 约 0.515 ms。此原型未接入模型。

full MCU 全范围采集进入多轮 application replay，耗时过长，已主动终止自有 profiler 与推理子进程；mcu-full-stopped.json、native-mcu-commands.json 和部分日志保存该事实，不作为有效热点报告。run_native_mcu_narrow.py 改用 pruned 中相同形状的专家算子、较窄 kernel filter，待执行。int2/int4 向量加载原型正在独立筛选，不影响已测试的 V3 模型实现。

向量加载筛选已完成（vector-screen.json，20 case 均通过数学检查）：int4 这里表示一次加载四个 32-bit 字，不是 INT4 量化。专家 GateUp 最佳 INT8 动态量化+GEMM 约 0.426 ms，同行通用 F16 BLAS 约 0.579 ms；前缀最佳约 4.050 ms，仍慢于 F16 3.473 ms，专家 Down 也没有收益。这个寄存器装载原型尚未接入模型，不能从该算子结果宣称整模型收益。
