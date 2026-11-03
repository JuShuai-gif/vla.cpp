from pathlib import Path
import json
d=Path(__file__).resolve().parent
lines=['# 单项与精度候选','','每项3次预热、5次同步诊断；非正式端到端收益。以各阶段 report.json、command.json、source.patch.gz 为具体版本和命令证据。','',
'|阶段|模型|视觉 ms|前缀 ms|去噪 ms|wall ms|历史参考相对峰值误差|相对基线差异|','|---|---|---:|---:|---:|---:|---:|---:|']
for stage in ['baseline','cross','edges','combined','action-linear','linear-cross','f16-prefix','f16-expert','f16-vision-v2','f16-all-pruned','f16-all-full','final-baseline-phases','final-selected-phases']:
 p=d/stage/'summary.json'
 if not p.exists():continue
 for r in json.loads(p.read_text()):
  p=r['phase_mean_ms'];lines.append(f"|{stage}|{r['model']}|{p['vision_ms']:.3f}|{p['prefix_ms']:.3f}|{p['denoise_ms']:.3f}|{r['mean_ms']['wall_ms']:.3f}|{r['error']['relative_peak']:.9f}|{r.get('baseline_relative_peak',0):.9f}|")
lines+=['','F16_PART=vision 的第一版断言失败，已保留原始日志；v2 为 F16 GEMM 输出、F32 残差/归一化，成功测试两个模型，但明显变慢。prefix/expert 分模块及全 F16 主激活在本轮两个模型的合成输入上均通过原误差门限，不再沿用旧版本失败结论；这些结果不代表随机输入或实机任务全部验证通过。','',
'精度对照：本轮推荐仍使用 F16 权重/F32 逻辑激活，并按模型选择融合。缩窄更多激活没有胜过已融合配置。完整 Action 单核改变 FP32 累加归约顺序，独立采用基线相对峰值0.1%及历史CPU参考0.5%的门限；逐元素一致路径仍要求差异为0。','',
'## GEMM 与预取微基准','','|形状 M,N,K|原路径 ms|swapped+布局恢复 ms|相邻双GEMM ms|加侧流 touch ms|','|---|---:|---:|---:|---:|']
for r in json.loads((d/'剩余GEMM候选结果.json').read_text()):
 if r['args'][3] or r['returncode']:continue
 a=r['measurements'];lines.append('|'+str(r['args'][:3])+'|'+'|'.join(f"{v['mean_ms']:.5f}" for v in a)+'|')
lines+=['','|GeGLU M,N,K|muBLAS+GeGLU ms|自定义 WMMA 融合 ms|最大输出差|','|---|---:|---:|---:|']
for r in json.loads((d/'剩余GEMM候选结果.json').read_text()):
 if not r['args'][3] or r['returncode']:continue
 a,b=r['measurements'];lines.append(f"|{r['args'][:3]}|{a['mean_ms']:.5f}|{b['mean_ms']:.5f}|{b['max_abs']}|")
lines+=['','第二轮 128线程多片 WMMA 原型运行失败；侧流 CTA/覆盖范围12组调参均未加速，完整结果见 调参候选结果.json。此处是已测原型结论，不宣称穷尽未来自定义 GEMM 或预取实现。']
(d/'01_单项与精度候选.md').write_text('\n'.join(lines)+'\n')
