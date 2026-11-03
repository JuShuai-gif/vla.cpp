# 八项优化最终验收

2026-11-03，服务器skysi，仓库/home/skysi/code/vla.cpp，基于2c271b0。两个延迟目标均已通过，源码、运行入口与中文记录一并提交Git。八项均有实施或实测结论，只启用正确且有收益的组合。

## 固定基准交替验收

M1000/MP22、MUSA 5.1.x、performance电源模式、4线程、双相机、chunk50、剪枝3步/完整10步。F16权重及Tensor乘法，F32累加/输出/残差，不是FP4/FP8量化。

固定6-token输入；每段预热3次，A→B→C→C→B→A，每段15次，每配置共30次。wall包含Python绑定到动作读回，不含模型加载、磁盘解码/resize和tokenizer。

|配置|剪枝平均/P95 ms|完整平均/P95 ms|
|---|---:|---:|
|同轮基线|511.478/512.113|880.299/885.294|
|未用新小GEMM|485.583/486.380|833.408/838.483|
|最终选择|467.927/471.363|770.523/771.171|

最终相对同轮基线下降8.51%/12.47%。原始证据：accept-*/、accept-status.json、正式验收耗时汇总.json。

## 真实输入30次

真实480×640图片，保持LeRobot float32 resize/pad，使用各自保存的tokenizer、state统计和相机顺序，预热3次后各计时30次。以下也是常驻predict wall，全部样本均低于目标，输出逐元素一致。

|模型|提示token|平均 ms|P95 ms|最大 ms|
|---|---:|---:|---:|---:|
|剪枝|43|481.444|482.143|482.963|
|完整|42|779.535|786.475|791.099|

来源：real-input30/。自动推理入口干净环境验证亦通过，见optimized-launcher-summary.json。

## 八项逐条结论

|项|实施/实测结论|最终选择|
|---|---|---|
|1 QKV拆分、RoPE/cast/KV写入|拆分本身为零拷贝VIEW，保留已有cast+KV尾写融合。K RoPE直接F16输出/尾写已实施并重测；late-k-rope-v2为465.737/768.535ms，但新增relative_peak为0.000135/0.000325，收益不足0.5%且没有稳定收益验收。|保持逐元素一致路径，不再启用额外RoPE融合|
|2 RoPE表|按每次当前positions生成prefix/suffix sin/cos表，各层共享；F32输出与原生路径逐元素一致，变位置/bucket检查通过。|ROPE_TABLE=1|
|3 U8 LUT+patch|原始HWC字节上传，精确F16 LUT及patch布局在一次GPU打包完成，直接进入embedding GEMM；覆盖全部256值和1/2视图。F32 resize输入保留原精度。|U8_PATCH=1，F32自动回退|
|4 固定RMS gamma折叠|加载时折叠到F16合并QKV/GateUp已实施，实测501.599/858.268ms；现有RMS+gamma已融合，额外舍入没有稳定收益。专家动态AdaRMS未当常量折叠。|关闭|
|5 最后前缀层KV-only|跳过无人使用的Q/O/FFN，只保留动作网络需要的K/V；独立和整模型对照通过。|LAST_KV_ONLY=1|
|6 视觉布局/projector出口|U8 embedding免旧布局复制；剪枝双相机批处理、分别评分/top-k并保留空间顺序；完整projector输出取消强制reshape+cont，直接写目标。既有bias/GELU/residual融合保留，未采用已知错误的Lt batch bias。|PRUNED_BATCH、VISION_NO_OUT_COPY及既有融合|
|7 GEMM/Attention|共享K/V的QK和AV由8批合并。实现MP22原生32×32×16 MMA、128线程协作；小GEMM选择M32/N64，O投影M64/N64，共享内存ld68。私有输入补零到64，但仍输出50个真实token。布局、数学模式、拆分、MMA原型和硬件剖析均有记录。|SMALL_GEMM=auto、FLAT_QK/AV=denoise|
|8 真实整网预取|旁路流在当前Down时预取下一层QKV，事件join保证使用前完成；实测64/256/4096KiB，不只测合成GEMM。最终64KiB为467.280/774.903ms，4096KiB为471.131/782.732ms，没有两模型稳定正收益。|关闭|

第1项没有另外增加处理混合dtype/多目标输出的大QKV节点：拆分为VIEW，已有cast/尾写融合，新增RoPE融合实测没有满足收益/数值取舍。第7项的GEMM+GeGLU合并原型也测试过，最终选择实测更好的独立小GEMM及现有GeGLU。未将微基准收益冒充整网收益，也未声称证明了所有可能实现的全局最优。

## 三部分诊断

下表是同步分段诊断，与普通wall的同步开销/样本数不同，不能把分段和代替上面的正式验收。

|模型/配置|视觉 ms|前缀 ms|动作 ms|分段wall ms|
|---|---:|---:|---:|---:|
|pruned/初始5样本|108.551|291.541|112.063|513.469|
|full/初始5样本|132.865|368.429|381.801|884.269|
|pruned/最终8样本|97.572|284.710|85.865|469.199|
|full/最终8样本|129.628|357.285|285.272|773.205|

异步完整入口JSON里ms_vision约0.5ms是主机提交，不是GPU视觉时长。

## 正确性

28/28 CTest通过且无GPU跳过；两模型各10组随机、32次U8/F32/视图切换、encode_images与precomputed/直接predict三轮重绑均通过，本轮新增选定路径对本轮基线逐元素一致。普通/bucket、生命周期、完整入口及旧参考对照10项均通过。证据：ctest-small-gemm-all-v2.log、random-candidate/、formats-candidate/、embedding-candidate/、candidate-validation-summary.json。

基线relative_peak 0.1%和既有CPU F32参考0.5%的原门限不变。完整入口结果：

|模型|逻辑激活|相对既有CPU F32参考 relative_peak|重复差异|
|---|---|---:|---:|
|剪枝|f32|0.001391513|0.0|
|剪枝|f16|0.000980917|0.0|
|完整|f32|0.000306570|0.0|
|完整|f16|0.000429625|0.0|

CPU单算子新增参考显式采用F16输入、F32累加，避免ARM原生F16 dot累加精度与内核契约混淆；原生GPU四种真实GEMM形状也独立对照。严格1e-5单算子门限和旧CPU整模型参考均未修改。更早full ActionLinear变化沿用上轮已验收范围。

同时修复CUSTOM userdata地址复用、语义常量未进Graph签名的问题，dt/epsilon/base复制进op_params；同图dt3→10→3回归通过。无效开关、错误fixture与撤回方案见实验汇总.md，未将未覆盖的早期测试冒称有效覆盖。

## 搬运与硬件证据

U8基准H2D总量1,259,776B→356,608B；剪枝4→3次，完整保持3次；D2H均一次6400B。显式Graph D2D copy节点：剪枝2个/2,359,296B、完整1个/4,194,304B，均降至0。每次各重放三段Graph。来源io-baseline/summary.json、io-candidate/summary.json。AV结果还原等GPU kernel仍有访存，copy节点为0不等于无GPU搬运。

MCU：前缀大GateUp原生库约90.53%计算吞吐，继续用库；新动作Down/GateUp约89.28%/90.29%，49寄存器/线程、8724B静态共享内存，独立kernel约245.34us/416.21us。不是整网时间。L2 API报0不代表物理没有L2。日志mcu-ffn.log、mcu-small-*.log；大型原始报告在/home/skysi/VLA/eight-target-20261103，校验见验收环境与产物.json。

## 运行和复现

现有两类GGUF直接复用。convert_pi05_local.py支持直接checkpoint/pretrained_model子目录、15/27层视觉及保存的SPM/JSON tokenizer。新--musa-optimized按GGUF vision_token_prune自动选两类性能环境，按元数据处理相机、状态与tokenizer。

~~~bash
cd /home/skysi/code/vla.cpp
VLA_LIBRARY=$PWD/build-musa-official/libvla.so VLA_N_THREADS=4 .venv-converter/bin/python scripts/run_pi05.py --musa-optimized  --model /home/skysi/VLA/models/pi05_cube_merge_cksize50_step3_0929/gguf/model.gguf  --image top=/path/top.png --image wrist=/path/wrist.png  --text 'pick up the cube' --state '0,0,0,0,0,0,0' --warmup 3  --output /tmp/pi05-actions.json
~~~

换完整模型GGUF路径即可。--musa-optimized选择验收过的F16权重/F32逻辑激活/FA；普通入口仍可做精度对照。环境见scripts/pi05_musa_profiles.json和本目录selection.json；早期RoPE+KV环境保存在selection-rope-kv.json。

~~~bash
BENCH_CONFIG=selection.json BENCH_RUNS=30 .venv-converter/bin/python  优化记录/2026-11-03_八项优化与延迟目标/run_bench.py 新的实验目录名
.venv-converter/bin/python 优化记录/2026-11-03_八项优化与延迟目标/bench_real_input.py  pruned /tmp/pi05-real-pruned-new
.venv-converter/bin/python 优化记录/2026-11-03_八项优化与延迟目标/bench_real_input.py  full /tmp/pi05-real-full-new
~~~

使用新输出目录以保存旧证据。Conda若使系统powerprofilesctl缺gi，应使用系统Python/系统PATH；该启动失败保留在late-k-rope.log，late-k-rope-v2已正常重测。

逐阶段实测表.md列出每步候选，实验汇总.md保留失败/无效实验，原始命令/环境/source.patch/动作均在本目录。仅保存服务器，未同步Mac。验收限本机M1000和记录中的输入，未做其他GPU或真实机器人成功率实测。
