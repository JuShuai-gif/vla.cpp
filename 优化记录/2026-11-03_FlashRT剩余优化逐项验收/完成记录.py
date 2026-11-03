from pathlib import Path
import json
d=Path(__file__).resolve().parent
for file,n in [('validation-summary.json',12),('random-status.json',4),('abc-status.json',6)]:
 a=json.loads((d/file).read_text());assert len(a)==n and all(x['returncode']==0 for x in a)
rows=json.loads((d/'最终交错对比.json').read_text());accuracy=json.loads((d/'精度汇总.json').read_text())
lines=['# FlashRT 剩余六类候选：实施、验证与取舍','','基线 d262af8。保留前轮全部推荐设置，完成本轮列出的六类候选实验。采用经回归和配对测量通过的路径；失败或负收益原型保留用于复现。','',
'|候选|结论|','|---|---|',
'|跨层 FFN 残差 + 下一层 AdaRMS|两模型采用，复用已验证内核|',
'|Action 输入/输出融合|pruned 采用 bias/AdaRMS、bias/dt/update 外围融合；full 采用输入投影/AdaRMS 和最终 AdaRMS/投影/update 完整单核|',
'|GeGLU GEMM epilogue|已实现 WMMA 原型：16×16正确但慢；128线程多片失败；32×32计时低但数值严重错误，均不采用|',
'|分模块及全 F16 主激活|当前合成场景数值均通过，但没有胜过已融合配置；视觉缩窄 GEMM 输出明显变慢，保留实验选项，不推荐|',
'|Swapped GEMM|7个形状均输出一致但变慢，不采用|',
'|L2侧流 load-touch|Graph fork/join 正确；7个形状及12组参数未得到稳定收益，不采用|','',
'## 六轮交替复测','','baseline→exact→linear→linear→exact→baseline，每轮3次预热、15次测量；每模型每配置共30次。F16权重/F32逻辑主激活，两视角，固定预处理观测，Graph开启。数字仅与同轮基线配对比较。','',
'|模型|基线均值 ms|选择后 ms|下降|P95基线→选择后 ms|','|---|---:|---:|---:|---:|']
for r in rows:
 a=r['variants']['baseline'];b=r['variants'][r['chosen']]
 lines.append(f"|{r['model']}|{a['mean_ms']:.3f}|{b['mean_ms']:.3f}|{r['saved_percent']:.2f}%|{a['p95_ms']:.3f} → {b['p95_ms']:.3f}|")
lines+=['','详细轮次和选择依据见 [六轮复测](03_最终交错复测.md)，单项、精度及GEMM原型见 [候选计时](01_单项与精度候选.md)。','',
'## 正确性验收','','25/25 CTest通过，无GPU跳过；12项变输入/完整入口/Graph任务及4项随机观测任务通过。随机覆盖每模型10组图片、文本、状态、噪声，每组重复结果完全一致。','',
'pruned 保持相对已验收基线逐元素一致。full 完整投影融合改变FP32归约顺序，不声称逐元素相同：']
for a in accuracy:
 lines.append(f"- {a['model']}：所有已汇总场景的最大基线相对峰值差异 {100*a['max_baseline_relative_peak']:.8f}%。")
lines+=['','full门限收紧至相对基线0.1%；完整入口CPU参考和历史MUSA F32合成参考仍保持原0.5%门限。F32/F16完整入口均通过，分词、相机顺序和重复结果不变。详见 精度汇总.json、summary-full-entry.json、完整入口基线一致性.json。','',
'Graph：每帧3次launch、1次最终同步，稳态无malloc/free/重新实例化。外部D2D=0；图内D2D仍为pruned 2个/full 1个视觉布局拷贝，未增加。Profiler节点duration不相加作为性能收益。','',
'## 三阶段同步诊断','','独立5次同步计时，用于定位，不代替上述正常端到端计时。','',
'|模型/配置|视觉 ms|前缀 ms|去噪 ms|','|---|---:|---:|---:|']
for stage in ['final-baseline-phases','final-selected-phases']:
 for r in json.loads((d/stage/'summary.json').read_text()):
  p=r['phase_mean_ms'];lines.append(f"|{r['model']}/{stage}|{p['vision_ms']:.3f}|{p['prefix_ms']:.3f}|{p['denoise_ms']:.3f}|")
fence=chr(96)*3
lines+=['','## 两种模型运行配置','',fence+'bash',
'cd /home/skysi/code/vla.cpp',
'source 优化记录/2026-11-03_FlashRT剩余优化逐项验收/推荐环境.sh pruned',
'# 完整模型将 pruned 换成 full；随后使用原 scripts/run_pi05.py 推理入口。',
'.venv-converter/bin/python scripts/bench_pi05.py --model "$VLA_MODEL" --weight-dtype f16 --act-dtype f32 --flash-attn --warmup 3 --runs 15 --out-dir /tmp/pi05-flashrt-verify',
fence,'',
'推荐环境脚本会清除其他实验开关后设置按模型验收的配置。必须在加载模型前使用；两模型不同tokenizer/checkpoint布局的完整调用见 full-entry/*/*-command.json。转换脚本及GGUF权重没有修改。','',
'## 归档与边界','','方案、成功/失败日志、图节点、各阶段源码patch随Git保存；大数组和msys原始报告保留服务器，见外部文件清单。command.json记录完整环境，新开关未纳入旧bench元数据白名单；源码和库校验见 源码与构建校验.json。早期完整Action原型补充头文件见 原型源码/说明.json。','',
'原型和实验开关不等于推荐配置；其验证范围见 [候选实施与边界](候选实施与边界.md)。WMMA并未穷尽所有tile/pipeline优化，load-touch是合成相邻GEMM试验，不宣称整个项目已达硬件理论最优。','',
'没有验证真实机器人任务成功率；所有新记录仅保存服务器，没有同步到Mac。']
(d/'README.md').write_text('\n'.join(lines)+'\n')
(d/'待做与验收.md').write_text('''# 本轮待办验收状态

六项均已完成本轮实施/实验和取舍：
1. 跨层融合：采用，两模型回归通过。
2. Action边缘：两条路径均实现；pruned选逐值一致外围融合，full选完整投影融合并通过更严数值门限。
3. GeGLU GEMM epilogue：三种WMMA原型已测，正确但慢/运行失败/更快但错误，均不采用。
4. 分模块低精度及全F16主激活：已重新测两模型，当前合成数值通过，性能不优，不推荐。
5. Swapped GEMM：7个形状均慢，不采用。
6. L2侧流load-touch：首轮7形状、调参12组，未加速，不采用。

本轮结果和验证范围以 README.md 为准，不宣称穷尽任意未来内核优化。
''')
p=d/'03_最终交错复测.md';s=p.read_text().replace('须通过额外数值与随机输入回归后才验收','已通过额外数值与随机输入回归并验收');p.write_text(s)
p=d.parent/'README.md';s=p.read_text();entry='| [2026-11-03 FlashRT剩余候选](2026-11-03_FlashRT剩余优化逐项验收/README.md) | 跨层/Action融合、三种WMMA原型、F16分区、swapped/侧流预取；30样本及随机输入验收 |\n'
assert '2026-11-03_FlashRT剩余优化逐项验收' not in s
p.write_text(s.replace('| [2026-11-03 借鉴 Thor 融合]',entry+'| [2026-11-03 借鉴 Thor 融合]',1))
p=d.parent/'方案与验证说明.md';p.write_text(p.read_text()+'\n## 2026-11-03：FlashRT剩余候选验收\n\n最新按模型推荐环境与逐项实验见 [剩余候选记录](2026-11-03_FlashRT剩余优化逐项验收/README.md)。pruned采用跨层与Action外围融合，full采用跨层与完整Action投影融合；full有少量FP32归约差异，已通过基线0.1%及CPU参考0.5%门限。完整入口、视角切换及随机输入回归通过。\n')
p=d.parent/'2026-11-03_借鉴Thor融合逐项实施/README.md';p.write_text(p.read_text()+'\n续轮已执行这里列出的剩余候选并按模型完成取舍，最新配置见 [FlashRT剩余候选验收](../2026-11-03_FlashRT剩余优化逐项验收/README.md)。\n')
