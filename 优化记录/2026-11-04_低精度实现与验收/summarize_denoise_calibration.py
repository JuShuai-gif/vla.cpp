"""Summarize captured calibration distributions by weight and denoise occurrence."""
import json,math,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
base=Path('/home/skysi/code/vla.cpp-quant/validation/quant/calibration')
result={'scope':'calibration only; actual denoise occurrences; no heldout access','models':{}}
for name,steps in [('pruned',3),('full',10)]:
    stats={};files=sorted((base/f'{name}-120/raw').glob('*-denoise.jsonl'))
    for path in files:
        with path.open() as stream:
            for line in stream:
                d=json.loads(line)
                if d.get('status')!='captured':continue
                key=(d['weight'],d['occurrence'])
                v=stats.setdefault(key,dict(samples=0,rows=0,amax_max=0.,squared_sum=0.,elements=0,input_types=[]))
                v['samples']+=1;v['rows']+=d['rows']
                v['amax_max']=max(v['amax_max'],max(d['amax']))
                v['squared_sum']+=sum(d['sum_squares'])
                v['elements']+=d['rows']*d['channels']
                if d['input_type'] not in v['input_types']:v['input_types'].append(d['input_type'])
    rows=[]
    for (weight,step),v in sorted(stats.items()):
        assert 0<=step<steps and v['samples']==120,(weight,step,v['samples'])
        rows.append(dict(weight=weight,denoise_step=step,rms=math.sqrt(v['squared_sum']/v['elements']),**v))
    result['models'][name]={'files':len(files),'expected_steps':steps,'rows':rows}
    print(name,len(files),len(rows),flush=True)
(root/'denoise-calibration-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
