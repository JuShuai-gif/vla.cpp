from pathlib import Path
import subprocess,sys,json
d=Path(__file__).resolve().parent;rows=[]
for model in ['pruned','full']:
 stage='f16-all-'+model
 print(stage,flush=True)
 with (d/(stage+'.log')).open('w') as f:
  p=subprocess.run([sys.executable,str(d/'run_stage.py'),stage,'BENCH_MODEL='+model,'BENCH_ACT=f16','BENCH_ALLOW_PRECISION=1','BENCH_RUNS=5','BENCH_SPLIT=1'],stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'model':model,'returncode':p.returncode})
 (d/'legacy-f16-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(stage,p.returncode,flush=True)
