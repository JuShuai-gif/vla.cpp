from pathlib import Path
import subprocess,sys,json
d=Path(__file__).resolve().parent
jobs=[('control',[]),('kv-direct',['VLA_PI05_KV_DIRECT=1']),('geglu-f16',['VLA_MUSA_GEGLU_F16=1']),('res-adarms',['VLA_PI05_RES_ADARMS=1']),('combined',['VLA_PI05_KV_DIRECT=1','VLA_MUSA_GEGLU_F16=1','VLA_PI05_RES_ADARMS=1'])]
rows=[]
for name,flags in jobs:
 print('CANDIDATE',name,flush=True)
 with (d/(name+'.log')).open('w') as f:
  p=subprocess.run([sys.executable,str(d/'run_stage.py'),name,'BENCH_RUNS=5','BENCH_SPLIT=1','VLA_PI05_DUMP_GRAPH=1']+flags,stdout=f,stderr=subprocess.STDOUT)
 rows.append({'name':name,'flags':flags,'returncode':p.returncode})
 (d/'candidate-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
