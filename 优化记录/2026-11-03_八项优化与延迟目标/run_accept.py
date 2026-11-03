from pathlib import Path
import os,subprocess,sys,json
d=Path(__file__).resolve().parent
c=json.loads((d/'candidate.json').read_text());b=json.loads(json.dumps(c))
for e in b.values():e.pop('VLA_MUSA_SMALL_GEMM',None)
(d/'before-small.json').write_text(json.dumps(b,indent=2)+'\n')
variants={'baseline':'baseline-env.json','before-small':'before-small.json','candidate':'candidate.json'}
rows=[]
for name,rnd in [('baseline',1),('before-small',1),('candidate',1),('candidate',2),('before-small',2),('baseline',2)]:
 stage='accept-'+name+'-r'+str(rnd);print(stage,flush=True)
 e=dict(os.environ,BENCH_RUNS='15',BENCH_CONFIG=variants[name]);e.pop('BENCH_SPLIT',None)
 with (d/(stage+'.log')).open('w')as f:
  p=subprocess.run([sys.executable,str(d/'run_bench.py'),stage],env=e,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'stage':stage,'returncode':p.returncode})
 (d/'accept-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)
