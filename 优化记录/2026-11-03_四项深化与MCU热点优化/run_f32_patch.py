from pathlib import Path
import json,os,sys,subprocess
d=Path(__file__).resolve().parent;repo=d.parents[1]
cfg=json.loads((d/'baseline-env.json').read_text())
for v in cfg.values():v['VLA_PI05_F32_PATCH']='1'
(d/'f32-patch-env.json').write_text(json.dumps(cfg,indent=2)+'\n')
rows=[]
for variant,config in [('control-f32','baseline-env.json'),('f32-patch','f32-patch-env.json')]:
 for s in ['pruned','full']:
  out=d/variant/s;out.parent.mkdir(exist_ok=True)
  env=dict(os.environ,BENCH_CONFIG=config,BENCH_RUNS='8')
  cmd=[sys.executable,str(d/'bench_real.py'),s,str(out)]
  (out.parent/(s+'-command.json')).write_text(json.dumps({'argv':cmd,'config':config},indent=2)+'\n')
  with (d/(variant+'-'+s+'.log')).open('w')as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  rows.append({'stage':variant,'model':s,'returncode':p.returncode})
  (d/'f32-patch-status.json').write_text(json.dumps(rows,indent=2)+'\n')
  if p.returncode:raise SystemExit(p.returncode)
  a=json.loads((out/'report.json').read_text());print(variant,s,a['mean_ms'],a['reference_relative_peak'],flush=True)
