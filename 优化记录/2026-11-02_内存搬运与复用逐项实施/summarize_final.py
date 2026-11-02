from pathlib import Path
import json, numpy as np
d=Path(__file__).resolve().parent
result=[];lines=['# 最终交错复测','',
'每轮 3 次预热、15 次正常测量，顺序 A→B→B→A；每模型每配置合计 30 次。A 保留上轮已验收的全部优化，B 采用 selection.json。下方分段耗时来自另测的 5 次同步诊断，不用于计算端到端收益。','',
'|模型|基线均值 ms|本轮均值 ms|减少 ms|改善|基线/本轮 P95 ms|','|---|---:|---:|---:|---:|---:|']
for model in ['pruned','full']:
 samples={}
 for variant in ['baseline','final']:
  reports=[json.loads((d/(variant+'-r'+str(i))/model/'report.json').read_text()) for i in [1,2]]
  vals=np.array([x['wall_ms'] for r in reports for x in r['runs']])
  assert len(vals)==30 and all(r['repeat_max_abs']==0 for r in reports)
  samples[variant]={'n':len(vals),'mean_ms':float(vals.mean()),'median_ms':float(np.median(vals)),'p95_ms':float(np.percentile(vals,95)),
                    'std_ms':float(vals.std()),'round_mean_ms':[float(np.mean([x['wall_ms'] for x in r['runs']])) for r in reports]}
 a,b=samples['baseline'],samples['final'];gain=a['mean_ms']-b['mean_ms'];pct=100*gain/a['mean_ms']
 phases={}
 for variant in ['baseline','final']:
  phases[variant]=next(x['phase_mean_ms'] for x in json.loads((d/('compare-'+variant+'-phases')/'summary.json').read_text()) if x['model']==model)
 row={'model':model,**samples,'saved_ms':gain,'saved_percent':pct,'phases':phases}
 result.append(row)
 lines.append(f"|{model}|{a['mean_ms']:.3f}|{b['mean_ms']:.3f}|{gain:.3f}|{pct:.2f}%|{a['p95_ms']:.3f} / {b['p95_ms']:.3f}|")
lines+=['','|模型/配置|视觉 ms|前缀 ms|动作去噪 ms|','|---|---:|---:|---:|']
for row in result:
 for variant,p in row['phases'].items():
  lines.append('|'+row['model']+'/'+variant+'|'+'|'.join(f'{p[k]:.3f}' for k in ['vision_ms','prefix_ms','denoise_ms'])+'|')
lines+=['','完整的各轮均值、标准差及 P95 见 最终对比.json；原始样本在 baseline-r1/r2、final-r1/r2 的 report.json。输出与本轮 baseline/actions.npy 比较，完整入口另与上轮验收输出逐值比较。']
(d/'最终对比.json').write_text(json.dumps(result,indent=2)+'\n')
(d/'05_最终交错复测.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
