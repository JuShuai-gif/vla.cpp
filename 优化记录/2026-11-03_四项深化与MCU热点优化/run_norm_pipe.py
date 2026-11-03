from pathlib import Path
import json,os,sys,subprocess
d=Path(__file__).resolve().parent
baseline=json.loads((d/'f32-patch-env.json').read_text());rows=[]
for stage,extra in [('norm-pack',{'VLA_PI05_NORM_PACK':'1'}),('pipe-1',{'VLA_MUSA_SMALL_PIPE':'1'}),('pipe-4',{'VLA_MUSA_SMALL_PIPE':'4'}),('pipe-8',{'VLA_MUSA_SMALL_PIPE':'8'})]:
 cfg={k:dict(v,**extra)for k,v in baseline.items()}
 config=stage+'-env.json';(d/config).write_text(json.dumps(cfg,indent=2)+'\n')
 for short in ['pruned','full']:
  out=d/stage/short;out.parent.mkdir(exist_ok=True)
  env=dict(os.environ,BENCH_CONFIG=config,BENCH_RUNS='8')
  cmd=[sys.executable,str(d/'bench_real.py'),short,str(out)]
  (out.parent/(short+'-command.json')).write_text(json.dumps({'argv':cmd,'config':config},indent=2)+'\n')
  with(d/(stage+'-'+short+'.log')).open('w')as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  row={'stage':stage,'model':short,'returncode':r.returncode}
  if r.returncode==0:
   report=json.loads((out/'report.json').read_text());row.update({k:report[k]for k in ['mean_ms','reference_relative_peak']})
  rows.append(row);(d/'norm-pipe-status.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(row,flush=True)
