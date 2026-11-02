from pathlib import Path
import json,numpy as np
d=Path(__file__).resolve().parent
result={}
for model in ['pruned','full']:
 row={}
 fingerprints=[]
 for mode in ['baseline','final']:
  reports=[json.loads((d/(mode+'-r'+str(i))/model/'report.json').read_text()) for i in [1,2]]
  vals=[x['wall_ms'] for r in reports for x in r['runs']]
  assert len(vals)==30 and all(r['repeat_max_abs']==0 for r in reports)
  for r in reports:
   m=r['metadata'];fingerprints.append((m['library_sha256'],m['core_sha256'],json.dumps(m['backend_sha256'],sort_keys=True)))
  row[mode]={'n':len(vals),'mean_ms':float(np.mean(vals)),'median_ms':float(np.median(vals)),
             'p95_ms':float(np.percentile(vals,95)),
             'round_means_ms':[float(np.mean([x['wall_ms'] for x in r['runs']])) for r in reports],
             'attention_mode':reports[0]['metadata']['environment'].get('VLA_MUSA_GEMMA_ATTN'),
             'error_vs_musa_f32':reports[0]['error_vs_reference']}
 assert len(set(fingerprints))==1,'Comparison used different runtime binaries'
 row['latency_reduction_percent']=100*(1-row['final']['mean_ms']/row['baseline']['mean_ms'])
 row['speedup']=row['baseline']['mean_ms']/row['final']['mean_ms']
 row['phases']={}
 for mode in ['baseline','final']:
  item=next(x for x in json.loads((d/(mode+'-phases')/'summary.json').read_text()) if x['model']==model)
  row['phases'][mode]=item['phase_mean_ms']
 result[model]=row
(d/'final-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
