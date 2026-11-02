from pathlib import Path
import os,sys,subprocess,json
d=Path(__file__).resolve().parent;repo=d.parents[1]
base=dict(os.environ,VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_MUSA_VISION_EPILOGUE='all')
for k in ['VLA_PI05_INPUT_CACHE','VLA_PI05_HOST_INPUT','VLA_PI05_ASYNC_VISION','VLA_PI05_VISION_DIRECT','VLA_PI05_KV_REUSE','VLA_PI05_NO_KV_REUSE','VLA_PI05_LANG_BUCKET','VLA_PI05_SPLIT_PHASES','BENCH_SPLIT','VLA_MUSA_ATTN_DIRECT_OUT','VLA_MUSA_VISION_PACK']:base.pop(k,None)
layout={'VLA_MUSA_ATTN_DIRECT_OUT':'1','VLA_MUSA_VISION_PACK':'1'}
io={'VLA_PI05_HOST_INPUT':'pinned','VLA_PI05_ASYNC_VISION':'1','VLA_PI05_VISION_DIRECT':'1'}
configs=[
 ('output',{'VLA_MUSA_ATTN_DIRECT_OUT':'1'}),
 ('vision-pack',{'VLA_MUSA_VISION_PACK':'1'}),
 ('layout',layout),
 ('combined',dict(layout,**io)),
 ('combined-kv',dict(layout,**io,VLA_PI05_KV_REUSE='1'))]
rows=[]
for stage,config in configs:
 for split in [False,True]:
  name=stage+('-phases' if split else '')
  env=dict(base,**config,BENCH_RUNS='3' if split else '5')
  if split:env['BENCH_SPLIT']='1'
  print('MEASURE',name,flush=True)
  with (d/(name+'.log')).open('w') as f:
   p=subprocess.run([sys.executable,str(d/'run_bench.py'),name],env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  rows.append({'stage':name,'returncode':p.returncode})
  (d/'layout-matrix-status.json').write_text(json.dumps(rows,indent=2))
  print(name,p.returncode,flush=True)
  if p.returncode:raise SystemExit(p.returncode)
