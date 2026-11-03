from pathlib import Path
import sys,os,subprocess,json
d=Path(__file__).resolve().parent
rows=[]
for part in ['vision','prefix','expert']:
 name='f16-'+part
 print('PRECISION',name,flush=True)
 with (d/(name+'.log')).open('w') as f:
  p=subprocess.run([sys.executable,str(d/'run_stage.py'),name,'BENCH_RUNS=5','BENCH_SPLIT=1','BENCH_ALLOW_PRECISION=1','VLA_PI05_F16_PART='+part],stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'part':part,'returncode':p.returncode})
 (d/'precision-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,p.returncode,flush=True)
