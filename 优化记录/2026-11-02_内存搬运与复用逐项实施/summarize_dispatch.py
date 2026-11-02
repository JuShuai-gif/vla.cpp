from pathlib import Path
import json,collections
d=Path(__file__).resolve().parent;rows=[]
for model in ['pruned','full']:
 source=d/'dispatch'/model/'run.log'
 records=[json.loads(x.split(' ',1)[1]) for x in source.read_text().splitlines() if x.startswith('VLA_MUSA_GEMM ')]
 counts=collections.Counter(json.dumps(x,sort_keys=True) for x in records)
 unique=[]
 for key,count in counts.items():
  x=json.loads(key);x['trace_calls']=count
  x['flatten_eligible']=x['flatten_requested']==1 and x['compute_type']==1 and x['out_type']==0 and x['a_batch']==1 and x['batch']>1 and x['b_outer']==1 and x['stride_b']==x['ldb']*x['n']
  unique.append(x)
 rows.append({'model':model,'trace_calls':len(records),'eligible_calls':sum(x['trace_calls'] for x in unique if x['flatten_eligible']),'shapes':unique})
(d/'GEMM实际分派.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print([(x['model'],x['trace_calls'],x['eligible_calls']) for x in rows])
