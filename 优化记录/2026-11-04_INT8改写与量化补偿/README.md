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

## V3 热点瓶颈已定位

narrow 第一轮误用默认 function 名称过滤，模板参数选择器未命中矩阵乘和归一化，报告仅为部分数据。mcu-narrow-filter-note.txt 保留原因。narrow2 显式指定 demangled，两个进程均正常结束，标记范围内完整记录：

| 内核 | 浮点次数/单次 ms | V3 INT8 次数/单次 ms |
|---|---:|---:|
| 专家 GateUp 矩阵乘 | 54 / 0.410056 | 与 GEGLU 融合 54 / 0.245188 |
| GEGLU | 54 / 0.047522 | 已融合 |
| 普通残差/AdaRMS | 105 / 0.031731 | 51 / 0.033093 |
| 残差/AdaRMS + 动态量化/打包 | 无 | 54 / 0.298449 |

见 v3-mcu-pruned2-*-summary.json 及原始 .log。量化归一化额外耗时抵消矩阵乘节省；这是 Graph OFF、受 MCU 干预的瓶颈诊断，不能替代普通 Graph ON 配对性能验收。

V4 已在源码中改为 K 可被 64 整除时的四字向量装载布局，并把动态激活打包由逐字节存储改为四字节存储；K=96 等保留原布局。权重、独立量化和融合 AdaRMS 同步更新；布局双射及四字节对齐检查保存 native-v4-layout-check.json。正在进行 K=96/128/1024 GPU 数值和模型验证，尚未宣称改善。

## V4 / V5 继续迭代

V4 的旧布局 K=96、新布局 K=128、真实专家宽度 K=1024 数值测试均通过（包含 F32/F16、R=0/16、尾部和变化输入）。V4 整模型 30 development 观测均通过动作门槛；pruned P50 476.651→502.779 ms（退步 5.48%），full 807.281→780.857 ms（单轮改善 3.27%）。这仍不满足双模型稳定提升，不作达标结论。原始报告 native-model-v4-screen.json，构建失败与修正日志均保留。

V5 继续把量化的逐元素严格除法改为循环外计算倒数、逐元素乘法；极小尺度下限为最小正规 F32，避免倒数溢出。可能在少数取整边界改变一个 INT8 code，因此以新的明确整数数学参考测试，不能沿用 V4 的逐元素动作一致性结论。K=96/128/1024 GPU 数值测试（额外包含极小 F32 输入）均通过；精度仍须实际 BF16 配对。

V5 子图四轮 AB/BA：浮点 GateUp+GEGLU P50 0.562–0.569 ms，INT8（含动态量化）0.318–0.325 ms。整模型回放正在独立运行，子图收益不能代替最终验收。没有修改编译器全局 fast-math、生产配置、模型结构或去噪步数。

V5 首轮双模型 30 development 观测均过 BF16 动作门槛：pruned native P50 473.211→465.189 ms（1.70%），最低余弦 0.999968427；full 794.549→770.542 ms（3.02%），最低余弦 0.999978946。软件 predict P50 分别 478.974→469.964 / 799.243→774.687 ms。尚需独立重复、完整开发集及留出集验证；单轮结果不能宣称稳定收益。V5 暂冻结为验收候选，保留 experimental 状态。

V5 全量回归 42/42 CTest 通过（native-v5-ctest.txt），候选库和配置哈希见 candidate-v5-freeze.json。90 观测开发集预先固定为 A0-B0-B1-A1 两对新进程；每对 native/predict P50 均须改善，按 episode 成组重采样的配对平均收益 95% 区间下界须大于零，另保留全部 P95 和动作误差。该标准针对本次“有所提升”目标，不宣称达到历史 20% 目标。

## V5 完整开发集交替复测与下一步覆盖扩展

冻结提交 b0c667b 后，以 A0/B0/B1/A1 新进程顺序对两模型各 90 个开发观测复测。
精简版最小逐动作余弦 0.999948107；两对 native P50 降幅 1.531% / 1.279%，通过本轮开发性能条件。
完整版最小逐动作余弦 0.999959676；两对 native P50 降幅 -0.271% / 6.285%，因此严格的逐对改善条件未通过。
两模型重复动作数组逐元素一致。原始结果见 v5-abba-*.json / *.actions.npz / *.log。
不能用 pooled / mean 的正收益掩盖完整版单对回退；V5 还不是目标验收完成。

当前 18 个专家 GateUp 约占总权重 6%，已向用户明确范围，默认继续扩大主干覆盖。
V6 扩展 17 个实际执行的前缀 GateUp，最后一层仅 K/V 不运行 MLP，仍保留浮点。
采用独立校准数据导出的 SmoothQuant 参数，专家 alpha=0.75、前缀初筛 alpha=0.5；
不是声称复现官方 AWQ。支持前缀独立 outlier 数量，初筛 0。
原生 GEGLU 融合支持大于 64 的 token 数，新增 65/161 跨 tile 数值与 padding 测试。
校准/开发/留出划分不变，尚未查看留出推理结果。V6 尚待构建、测试和整模型验证。

快速激活打包的 Split-K 复测共 32 例，独立整数参考均通过（split-k-fastpack-screen.json）。
实际主干合并 GateUp 为 K=2048/M=32768，原早期 M=16384 仅半尺寸，不能当作实际投影结果。
该实际尺寸 N=320 时不分 K 最好：5.985ms vs 通用 F16 BLAS 6.674ms；
分 K=2/4/8 反而约 13.87/15.43/17.95ms，不采用。
专家 Down K=4096/M=1024/N=50 不分 K 约0.298ms vs BLAS0.497ms，split4最快约0.274ms；
但通用 BLAS 不是最终模型路径，暂不宣称模型获益，也尚未扩展 Down 接入。
V6 构建最初误指不存在的 build-musa-official，未修改生产；错误日志保留在 native-v6-directory-error*。
修正为隔离 build-musa-quant 后构建、3项原生测试及23项Python测试通过。

V6 两模型 30 观测初筛均通过逐动作余弦门槛：
精简 native P50 482.968→415.469ms（-13.976%），predict 487.973→419.754ms，最低余弦0.9999499246；
完整 native P50 771.354→686.529ms（-10.997%），predict 776.581→690.654ms，最低余弦0.9999818196。
加载时间约40.18/43.04s，浮点基线约11.23/10.69s；当前按加载时转换权重，不能把加载性能写成提升。
35 个投影共1,291,845,632权重元素，相对GGUF全部 *.weight 元素分别40.753%/38.532%；
分母包含GGUF中未执行的权重，不能与此前按实际驻留量估计的专家约6%直接等同。
依旧是混合精度，不是全网络INT8。模式错误、缺少前缀层、多余未执行层、参数SHA损坏均在CPU预检拒绝。
接下来冻结 V6 对全部90开发观测做ABBA，尚未跑留出集；以上不是最终验收。

## V6 完整开发集与留出验收进度

两模型各90开发观测，A0/B0/B1/A1新进程交替均通过；重复动作数组完全一致。
精简最低余弦0.9999318798，native P50降幅11.986% / 12.623%；
完整最低余弦0.9999742928，native P50降幅11.290% / 10.756%。
按episode分组bootstrap的平均提速95%区间分别[12.027%,12.375%]、[10.795%,11.359%]。
V6的90观测浮点基线动作与V5逐元素完全相同，见v6-baseline-preservation.json。

留出首次运行在精简BF16参考生成后被验收脚本断言中断：source_sha256包含输入manifest，
开发/留出manifest理应不同。7个共享来源（checkpoint、配置、框架与参考脚本）全部相同。
修正仅作用于归档验收驱动：各自核对manifest自身哈希，并严格比较其余来源；
额外检查留出split、396条记录，以及input/noise/token/task/episode逐项配对。
模型、量化参数、参考脚本、runtime及冻结库均未变化；保存失败驱动、日志与反例自检。
复用已完成的BF16结果，未重新选择候选或调整量化参数。

精简396留出观测、19,800动作向量全部通过，最低余弦0.9998076803；
native P50降幅12.498%，predict P50降幅12.522%。
完整版留出、runtime回放、完整回归仍在进行，目标尚未完成。

## 最终结果：本次预先定义的离线验收通过

候选代码提交80f6693，配置precision=int8-main-smooth。每模型120校准、90开发、396留出观测；
开发/留出episode分离，候选在留出前冻结，未根据留出结果调整模型或量化参数。
使用SmoothQuant逐通道缩放和原生W8A8/INT32融合内核，18专家+17前缀GateUp共1,291,845,632权重元素；
按GGUF全部*.weight计约40.75%/38.53%，其余仍浮点，不能称全网络INT8。

| 模型 | 留出逐时刻7维动作最低BF16余弦 | native P50 浮点→INT8 | 耗时下降 | 公共API P50 浮点→INT8 |
| --- | --- | --- | --- | --- |
| 精简 | 0.9998076803 | 469.232→410.589ms | 12.498% | 473.828→414.495ms |
| 完整 | 0.9997207221 | 772.510→687.207ms | 11.042% | 778.502→691.825ms |

每模型19,800个留出动作向量均通过0.999门槛。性能参考为当前C++浮点运行时，精度参考为原checkpoint实际BF16前向。
开发集ABBA提供重复性能证据；runtime离线ABBA每进程30次：
精简两对P50下降12.011%/11.612%，完整10.371%/9.876%；8个进程均正常退出、动作50×7且有限。
完整C++回归42/42通过，相关Python测试23/23通过。最终冻结源码、库、校准参数哈希均一致。
生产main仍67f47a0、4份真机配置与生产库SHA不变，runtime既有未提交修改保留；未部署、未合并、未推送、未下发机械臂动作。

本次门槛明确为每个时刻的7维动作向量，不保证每个关节的50步时间轨迹余弦。
该诊断的最低值精简0.510771、完整0.615521；对应BF16轨迹范围分别[-0.195,0.441]、
[-0.473,0.605]，MAE分别0.129、0.209（checkpoint输出单位，尚未独立确认物理单位）。
诊断细节见v6-joint-trajectory-diagnostic.json，原始逐关节误差和余弦保留在comparison文件；
不能只凭高7维余弦宣称真机闭环或所有单关节指标达标。

一次性加载仍较慢：留出测试精简14.581→40.583s、完整15.986→42.046s。
上述提速是模型驻留后的完整推理/软件调用，包含动态激活量化与预后处理，不包含重新加载。
未做真机闭环验证。这里完成的是当前用户要求对应的离线目标，不改变此前暂停的20%性能目标。

复现入口：pruned-main-native-policy.json / full-main-native-policy.json；
公共API选择precision=int8-main-smooth；离线运行时配置runtime-v6-config.json（设备全部null，execute_robot=false）。
精确命令、源文件哈希和数组见v6-abba-commands.json、v6-heldout-commands.json、v6-runtime-commands.json、
candidate-v6-freeze.json、v6-final-acceptance.json。保留全部失败/中断记录及修正说明。
