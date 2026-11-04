# 低精度实现与验收（进行中）

用户已授权持续迭代量化实验及 robot_runtime 精度参数接入；未授权真机动作。
隔离工作树：vla.cpp-quant / robot_runtime-quant；生产主分支保持原样。
原始代码基线：vla.cpp 3dc88d7，robot_runtime 22e5223。基线权重不修改。

## 实施进度

- 转换器：显式 prefix-mlp / prefix / vision / prefix-mixed / expert-mlp / expert / all-linear 配方及精确张量映射；保护头与投影、融合组一致性、F16输入、临时文件与读回校验已实现，三类往返与失败测试通过。
- 公共 API 与 robot_runtime：精度配置选择已实现，两个模型/三精度的6组资产检查及pruned Q8真实模型离线回放通过。实验配置禁止直接用于真机执行。
- 开发数据：202段真实Piper轨迹，按episode切分为120校准/90开发/396留出观测。训练来源匹配未确认，不视为完整任务覆盖或闭环证据。
- pruned Q8 prefix-mlp：已生成并逐张量读回验证，54矩阵，载荷7042473504→5343780384字节；修复MMQ后的性能/动作误差见第三轮结果。
- FP16激活与Q8前缀MLP已筛选，均未通过性能门槛；其他Q8分区、混合Q4、动作专家及校准仍在迭代。
- 最终门槛仍为原方案的驻留P50至少20%、P95至少15%、软件全链路P50至少15%改善；不能用体积变化代替速度证据。
- 物理误差预算及训练单位等待用户确认；真机未连接，闭环验收未完成。

权重与观测位于 validation/quant（不纳入Git）；生成文件的SHA与配方、原始性能/精度报告保存本目录。

## 第一轮筛选：原生产动态库，12个开发观测

- pruned baseline resident P50=472.063ms；FP16激活493.565ms（退步约4.55%），动作最大差异0.07238（源单位待确认）。不能通过性能门槛。
- pruned Q8 prefix-mlp P50=278.885ms，但动作最大差异99.69142，明确拒绝；表面的40.9%降时不能计作成功。
- 独立Q8_0 MUSA GEMM测试M64/N50/K256：3200项数学参考校验3154项失败，最大误差12.1571；M32/N32/K32还出现非有限值。优先排查MMQ，暂不扩大错误内核上的模型量化。
- 初始隔离构建30/30 CTest通过，但发现相较生产缺少muDNN attention开关。已纠正为ON重建；上面的模型筛选统一用原生产动态库，不能与未匹配构建混为一谈。
- 新增可追踪auto/BLAS对照调度和独立数学参考；两次构建错误（头文件命名空间、补丁幂等性）已修复，失败日志保留，重复应用补丁SHA不变。
- 当前筛选有并行CPU编译，且观察到新token长度触发Graph重捕获，因此不是最终稳定性能验收。

## 第二轮：内核正确性修复

- 实测M1000 warpSize=128，源码设备内核按32线程子组索引；同时主机vendor-offset CC误选Ampere配置、设备端误选Pascal配置。
- 补丁统一主机/设备MMQ使用DP4A配置，MUSA的MMQ launch x维明确32；不更改其他后端、不绕过上游架构安全guard。
- 仅修配置仍错误（mmq-fixed-*.txt）；同时修正launch后，小矩阵3200项全通过(max1.26e-6)，真实Gate/Up形状M32768/N320/K2048抽查4096项全通过(max4.29e-6)。
- 真实大矩阵MMQ约180.75ms，仅正确性通过，性能尚不合格；后续需硬件矩阵单元实现，不能把先前错误计算的40.9%“加速”算作收益。
- primitive探针：ARM默认char无符号构建会造成__dp4a错误；按ggml实际-fsigned-char构建，dp4a/激活量化/scale全部通过。保留两次报告，避免误认硬件dot指令错误。
- BLAS对照Graph捕获失败，禁Graph的小矩阵数学参考检查通过；该路径也需显式处理Graph兼容性，不能静默改变基线。
- 提示词静态资产缓存：两个模型各100组提示词/token逐元素一致；pruned P50 2.36→0.215ms，full 90.25→0.427ms（独立CPU测量，不是整模型加速）。
- robot_runtime全量复测805项通过，39项可选依赖跳过；6组真实模型/精度资产检查通过，CLI已支持precision参数，生产两库仍无改动。

## 第三轮：正确内核的整模型成本与运行时接入

- 匹配生产muDNN构建后，33项CTest全部通过（ctest-mmq-fixed.txt）。
- pruned固定12开发观测：B0 native P50=468.191419ms；Q8-MLP=5006.466827ms，延迟约为B0的10.69倍，明确性能失败。Q8动作每维最大误差为[0.167984,0.480431,0.620461,0.188438,0.359436,0.225742,0.489464]，尚无经确认物理预算，不能判部署精度通过。
- 两模型FP16激活筛选均无性能收益；full约9.21%退步，最大动作差异约0.08844。原始screen报告保留，不与修复后的运行混用。
- API支持命名precision_profiles、baseline默认值、量化manifest一致性与加载前完整SHA校验；robot_runtime的CLI/回放/网页字段传递precision，experimental配置拒绝真机执行。
- --dry-run和离线回放在资源预检查前生效。真实pruned Q8离线回放输出50×7，记录resolved_parameters与约5008ms延迟；没有CAN/相机/机械臂I/O。
- 最新runtime串行回归807项成功，39项可选依赖跳过（robot-runtime-tests-serial.txt）。前一轮并行回放占用HardwareLease导致14失败/10错误，串行复跑全部消除，保留robot-runtime-tests-dryrun.txt以免隐瞒失败。

## 第四轮：矩阵单元可行性探针（进行中）

- 块32 Q8权重/激活INT8 WMMA：小矩阵和真实Gate/Up形状数学参考全部通过，真实形状GPU P50约94.06ms，包含激活量化，仍无收益。
- 当前muBLAS的I8/I8→I32 GemmEx返回MUBLAS_STATUS_NOT_SUPPORTED(8)，不是证明所有MUSA卡不支持；本机当前SDK调用证据在row-mublas-not-supported.txt。
- 每通道权重/每token激活INT8 WMMA直接读取原型：真实形状4096抽查全部通过(max1.04e-6)，GPU P50=15.871599ms；同形状F16 BLAS含输入cast=6.712520ms。此原型仍慢，未接入正式模型，不宣称速度收益。
- 开启-ffast-math时CPU/GPU分界舍入有差异；精确舍入参考构建消除差异，保留失败报告。后续生产接入必须固定量化舍入契约。
- 新原型微基准仅为可行性检查；与GGML wall时间、整模型时延分开记录。
