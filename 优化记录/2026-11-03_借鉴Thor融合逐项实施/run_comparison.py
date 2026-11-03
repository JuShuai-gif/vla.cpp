from pathlib import Path
import os,sys,subprocess,json
r=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
base=dict(os.environ)
for k in ['VLA_PI05_KV_DIRECT','VLA_MUSA_GEGLU_F16','VLA_PI05_RES_ADARMS']:base.pop(k,None)
base.update(json.loads((d/'baseline-env.json').read_text()))
checks=json.loads((d/'validation-summary.json').read_text())
assert len(checks)==12 and all(x['returncode']==0 for x in checks)
rows=[]
for stage in ['baseline-r1','final-r1','final-r2','baseline-r2']:
 env=dict(base,BENCH_RUNS='15')
 if 'final' in stage:env['BENCH_SELECTED']='1'
 if stage.endswith('phases'):env.update(BENCH_RUNS='5',BENCH_SPLIT='1',VLA_PI05_DUMP_GRAPH='1')
 print('MEASURE',stage,flush=True)
 with (d/(stage+'.log')).open('w') as f:
  p=subprocess.run([sys.executable,str(d/'run_bench.py'),stage],env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'stage':stage,'returncode':p.returncode})
 (d/'comparison-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(stage,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
