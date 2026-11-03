# FlashRT 剩余六类候选：实施、验证与取舍

基线 d262af8。保留前轮全部推荐设置，完成本轮列出的六类候选实验。采用经回归和配对测量通过的路径；失败或负收益原型保留用于复现。

|候选|结论|
|---|---|
|跨层 FFN 残差 + 下一层 AdaRMS|两模型采用，复用已验证内核|
|Action 输入/输出融合|pruned 采用 bias/AdaRMS、bias/dt/update 外围融合；full 采用输入投影/AdaRMS 和最终 AdaRMS/投影/update 完整单核|
|GeGLU GEMM epilogue|已实现 WMMA 原型：16×16正确但慢；128线程多片失败；32×32计时低但数值严重错误，均不采用|
|分模块及全 F16 主激活|当前合成场景数值均通过，但没有胜过已融合配置；视觉缩窄 GEMM 输出明显变慢，保留实验选项，不推荐|
|Swapped GEMM|7个形状均输出一致但变慢，不采用|
|L2侧流 load-touch|Graph fork/join 正确；7个形状及12组参数未得到稳定收益，不采用|

## 六轮交替复测

baseline→exact→linear→linear→exact→baseline，每轮3次预热、15次测量；每模型每配置共30次。F16权重/F32逻辑主激活，两视角，固定预处理观测，Graph开启。数字仅与同轮基线配对比较。

|模型|基线均值 ms|选择后 ms|下降|P95基线→选择后 ms|
|---|---:|---:|---:|---:|
|pruned|513.463|512.099|0.27%|514.283 → 512.816|
|full|889.260|881.618|0.86%|890.156 → 883.286|

详细轮次和选择依据见 [六轮复测](03_最终交错复测.md)，单项、精度及GEMM原型见 [候选计时](01_单项与精度候选.md)。

## 正确性验收

25/25 CTest通过，无GPU跳过；12项变输入/完整入口/Graph任务及4项随机观测任务通过。随机覆盖每模型10组图片、文本、状态、噪声，每组重复结果完全一致。

pruned 保持相对已验收基线逐元素一致。full 完整投影融合改变FP32归约顺序，不声称逐元素相同：
- pruned：所有已汇总场景的最大基线相对峰值差异 0.00000000%。
- full：所有已汇总场景的最大基线相对峰值差异 0.00887508%。

full门限收紧至相对基线0.1%；完整入口CPU参考和历史MUSA F32合成参考仍保持原0.5%门限。F32/F16完整入口均通过，分词、相机顺序和重复结果不变。详见 精度汇总.json、summary-full-entry.json、完整入口基线一致性.json。

Graph：每帧3次launch、1次最终同步，稳态无malloc/free/重新实例化。外部D2D=0；图内D2D仍为pruned 2个/full 1个视觉布局拷贝，未增加。Profiler节点duration不相加作为性能收益。

## 三阶段同步诊断

独立5次同步计时，用于定位，不代替上述正常端到端计时。

|模型/配置|视觉 ms|前缀 ms|去噪 ms|
|---|---:|---:|---:|
|pruned/final-baseline-phases|108.797|292.455|114.246|
|full/final-baseline-phases|133.094|368.279|390.266|
|pruned/final-selected-phases|108.741|291.804|112.548|
|full/final-selected-phases|132.853|368.169|382.262|

## 两种模型运行配置

```bash
cd /home/skysi/code/vla.cpp
source 优化记录/2026-11-03_FlashRT剩余优化逐项验收/推荐环境.sh pruned
# 完整模型将 pruned 换成 full；随后使用原 scripts/run_pi05.py 推理入口。
.venv-converter/bin/python scripts/bench_pi05.py --model "$VLA_MODEL" --weight-dtype f16 --act-dtype f32 --flash-attn --warmup 3 --runs 15 --out-dir /tmp/pi05-flashrt-verify
```

推荐环境脚本会清除其他实验开关后设置按模型验收的配置。必须在加载模型前使用；两模型不同tokenizer/checkpoint布局的完整调用见 full-entry/*/*-command.json。转换脚本及GGUF权重没有修改。

## 归档与边界

方案、成功/失败日志、图节点、各阶段源码patch随Git保存；大数组和msys原始报告保留服务器，见外部文件清单。command.json记录完整环境，新开关未纳入旧bench元数据白名单；源码和库校验见 源码与构建校验.json。早期完整Action原型补充头文件见 原型源码/说明.json。

原型和实验开关不等于推荐配置；其验证范围见 [候选实施与边界](候选实施与边界.md)。WMMA并未穷尽所有tile/pipeline优化，load-touch是合成相邻GEMM试验，不宣称整个项目已达硬件理论最优。

没有验证真实机器人任务成功率；所有新记录仅保存服务器，没有同步到Mac。

计时说明：A/B/C 使用同一最终二进制；A 关闭本轮开关并保留上轮推荐设置，是功能对照，不是重新运行 d262af8 的历史二进制。代码起点、最终差异和库哈希均已记录。
