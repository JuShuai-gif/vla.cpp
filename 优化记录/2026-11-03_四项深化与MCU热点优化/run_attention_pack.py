from pathlib import Path
import json,os,sys,subprocess,time
d=Path(__file__).resolve().parent;repo=d.parents[1]
while Path('/proc/26330').exists():time.sleep(1)
with(d/'build-attention-pack.log').open('w')as f:r=subprocess.run(['cmake','--build','build-musa-official','-j','8'],cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=900)
if r.returncode:raise SystemExit(r.returncode)
with(d/'test-attention-pack.log').open('w')as f:r=subprocess.run(['ctest','--test-dir','build-musa-official','-R','musa_(packed_norm|prune_score|tiled_attention256)','--output-on-failure'],cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=600)
print('focused-tests',r.returncode,flush=True)
baseline=json.loads((d/'f32-patch-env.json').read_text());rows=[]
for stage,extra in [
 ('geglu-pack',{'VLA_PI05_GEGLU_PACK':'1'}),
 ('prune-mean',{'VLA_PI05_PRUNE_MEAN':'1'}),
 ('tiled-denoise',{'VLA_MUSA_GEMMA_TILED':'denoise'}),
 ('tiled-all',{'VLA_MUSA_GEMMA_TILED':'all'}),
 ('pack-combined',{'VLA_PI05_NORM_PACK':'1','VLA_PI05_GEGLU_PACK':'1','VLA_PI05_PRUNE_MEAN':'1'})]:
 cfg={k:dict(v,**extra)for k,v in baseline.items()}
 config=stage+'-env.json';(d/config).write_text(json.dumps(cfg,indent=2)+'\n')
 for short in ['pruned','full']:
  if stage=='prune-mean'and short=='full':continue
  out=d/stage/short;out.parent.mkdir(exist_ok=True)
  env=dict(os.environ,BENCH_CONFIG=config,BENCH_RUNS='8')
  cmd=[sys.executable,str(d/'bench_real.py'),short,str(out)]
  (out.parent/(short+'-command.json')).write_text(json.dumps({'argv':cmd,'config':config},indent=2)+'\n')
  with(d/(stage+'-'+short+'.log')).open('w')as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  row={'stage':stage,'model':short,'returncode':r.returncode}
  if r.returncode==0:
   report=json.loads((out/'report.json').read_text());row.update({k:report[k]for k in ['mean_ms','reference_relative_peak']})
  rows.append(row);(d/'attention-pack-status.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(row,flush=True)
