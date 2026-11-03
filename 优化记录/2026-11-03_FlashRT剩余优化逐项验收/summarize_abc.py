from pathlib import Path
import json,numpy as np
d=Path(__file__).resolve().parent;rows=[]
for model in ['pruned','full']:
 variants={}
 for v in ['baseline','exact','linear']:
  rr=[json.loads((d/(v+'-r'+str(i))/model/'report.json').read_text()) for i in [1,2]]
  a=np.array([r['wall_ms'] for q in rr for r in q['runs']])
  assert len(a)==30 and all(q['repeat_max_abs']==0 for q in rr)
  variants[v]={'mean_ms':float(a.mean()),'std_ms':float(a.std()),'p95_ms':float(np.percentile(a,95)),'n':len(a),'round_ms':[float(np.mean([r['wall_ms'] for r in q['runs']])) for q in rr]}
 chosen='exact' if model=='pruned' else 'linear';a=variants['baseline'];b=variants[chosen]
 rows.append({'model':model,'variants':variants,'chosen':chosen,'saved_ms':a['mean_ms']-b['mean_ms'],'saved_percent':100*(a['mean_ms']-b['mean_ms'])/a['mean_ms']})
(d/'最终交错对比.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# 六轮交错复测','','顺序 baseline→exact→linear→linear→exact→baseline。每轮预热3次，测量15次；每模型每配置30次。exact=跨层+Action外围融合，linear=跨层+完整Action投影融合。全部复用前轮的完整推荐开关。','',
'|模型|配置|均值 ms|P95 ms|标准差 ms|两轮均值 ms|','|---|---|---:|---:|---:|---|']
for r in rows:
 for v,a in r['variants'].items():lines.append(f"|{r['model']}|{v}|{a['mean_ms']:.3f}|{a['p95_ms']:.3f}|{a['std_ms']:.3f}|{a['round_ms']}|")
lines+=['','pruned 的 linear 第二轮退步且整体慢于 exact，因此选择保持逐元素一致的 exact。full 的 linear 两轮均比 exact 快约2.2ms，须通过额外数值与随机输入回归后才验收；基线差异门限收紧为0.1%，历史CPU参考门限仍为0.5%。']
(d/'03_最终交错复测.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
