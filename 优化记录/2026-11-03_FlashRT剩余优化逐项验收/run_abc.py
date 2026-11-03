from pathlib import Path
import sys,subprocess,json
d=Path(__file__).resolve().parent
variants={'baseline':[],'exact':['VLA_PI05_CROSS_ADARMS=1','VLA_PI05_ACTION_EDGES=1'],'linear':['VLA_PI05_CROSS_ADARMS=1','VLA_PI05_ACTION_LINEAR=1','BENCH_ALLOW_PRECISION=1']}
rows=[]
for name,round in [('baseline',1),('exact',1),('linear',1),('linear',2),('exact',2),('baseline',2)]:
 stage=name+'-r'+str(round)
 print(stage,flush=True)
 with (d/(stage+'.log')).open('w') as f:
  p=subprocess.run([sys.executable,str(d/'run_stage.py'),stage,'BENCH_RUNS=15']+variants[name],stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'stage':stage,'returncode':p.returncode})
 (d/'abc-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(stage,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
