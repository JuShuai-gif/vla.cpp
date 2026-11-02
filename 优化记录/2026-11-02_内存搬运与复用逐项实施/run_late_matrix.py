from pathlib import Path
import os,sys,subprocess,json
d=Path(__file__).resolve().parent;r=d.parents[1]
base=dict(os.environ,VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_MUSA_VISION_EPILOGUE='all',
 VLA_MUSA_VISION_PACK='1',VLA_MUSA_ATTN_DIRECT_OUT='1',VLA_PI05_HOST_INPUT='pinned',
 VLA_PI05_ASYNC_VISION='1',VLA_PI05_VISION_DIRECT='1',VLA_PI05_KV_REUSE='1')
rows=[]
for stage,config in [('late-control',{}),('main-pin',{'VLA_PI05_PIN_MAIN_INPUT':'1'}),('flatten',{'VLA_MUSA_FLATTEN_GEMM':'1'}),('pin-flatten',{'VLA_PI05_PIN_MAIN_INPUT':'1','VLA_MUSA_FLATTEN_GEMM':'1'})]:
 for split in [False,True]:
  name=stage+('-phases' if split else '');env=dict(base,**config,BENCH_RUNS='3' if split else '5')
  if split:env['BENCH_SPLIT']='1'
  print('MEASURE',name,flush=True)
  with (d/(name+'.log')).open('w') as f:p=subprocess.run([sys.executable,str(d/'run_bench.py'),name],env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=900)
  rows.append({'stage':name,'returncode':p.returncode});(d/'late-matrix-status.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(name,p.returncode,flush=True)
  if p.returncode:raise SystemExit(p.returncode)
