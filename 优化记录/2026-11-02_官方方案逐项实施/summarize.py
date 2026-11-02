from pathlib import Path
import json
import numpy as np
d=Path(__file__).resolve().parent
result={}
for model in ['pruned','full']:
 row={}
 for mode in ['baseline','final','nocache']:
  reports=[json.loads((d/(mode+'-r'+str(i))/model/'report.json').read_text()) for i in [1,2]]
  values=[r['wall_ms'] for report in reports for r in report['runs']]
  assert len(values)==30 and all(r['repeat_max_abs']==0 for r in reports)
  row[mode]={'n':len(values),'mean_ms':float(np.mean(values)),'median_ms':float(np.median(values)),
             'p95_ms':float(np.percentile(values,95)),
             'round_means_ms':[float(np.mean([r['wall_ms'] for r in report['runs']])) for report in reports],
             'error_vs_musa_f32':reports[0]['error_vs_reference']}
 for mode in ['final','nocache']:
  row[mode]['latency_reduction_percent']=100*(1-row[mode]['mean_ms']/row['baseline']['mean_ms'])
  row[mode]['speedup']=row['baseline']['mean_ms']/row[mode]['mean_ms']
 row['cache_delta_ms']=row['final']['mean_ms']-row['nocache']['mean_ms']
 result[model]=row
(d/'final-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
