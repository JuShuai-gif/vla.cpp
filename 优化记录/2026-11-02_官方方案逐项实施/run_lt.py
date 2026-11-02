from pathlib import Path
import subprocess,json,os
d=Path(__file__).resolve().parent; out=d/'lt';out.mkdir(exist_ok=True)
shapes={}
for model in ['pruned','full']:
 for s in json.loads((d/'baseline'/model/'shapes.json').read_text()):
  if s['op']!='MUL_MAT':continue
  dst,a,b=s['tensors'][:3]
  if a['type']!='f16':continue
  key=(a['ne'][1],b['ne'][1],a['ne'][0],b['ne'][2]*b['ne'][3])
  shapes.setdefault(key,[]).append({'model':model,'phase':s['phase'],'count':s['count']})
summary=[]
for shape,usage in shapes.items():
 name='x'.join(map(str,shape));print('LT',name,flush=True)
 cmd=['/tmp/pi05-probe-lt',*map(str,shape)]
 env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/musa/lib')
 with (out/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 lines=(out/(name+'.log')).read_text().splitlines()
 rows=[json.loads(x) for x in lines if x.startswith('{')]
 summary.append({'shape':shape,'usage':usage,'returncode':p.returncode,'results':rows})
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary[-1]),flush=True)
