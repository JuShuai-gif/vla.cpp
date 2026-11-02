from pathlib import Path
import os,sys,subprocess,json
d=Path(__file__).resolve().parent;r=d.parents[1]
base=dict(os.environ,VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_MUSA_VISION_EPILOGUE='all',VLA_MUSA_VISION_PACK='1',VLA_MUSA_ATTN_DIRECT_OUT='1')
configs=[
 ('pageable',{'VLA_PI05_HOST_INPUT':'pageable'}),
 ('pinned',{'VLA_PI05_HOST_INPUT':'pinned'}),
 ('mapped',{'VLA_PI05_HOST_INPUT':'mapped'}),
 ('async',{'VLA_PI05_HOST_INPUT':'pinned','VLA_PI05_ASYNC_VISION':'1'}),
 ('direct',{'VLA_PI05_HOST_INPUT':'pinned','VLA_PI05_ASYNC_VISION':'1','VLA_PI05_VISION_DIRECT':'1'}),
 ('mapped-direct',{'VLA_PI05_HOST_INPUT':'mapped','VLA_PI05_ASYNC_VISION':'1','VLA_PI05_VISION_DIRECT':'1'}),
 ('kv',{'VLA_PI05_HOST_INPUT':'pinned','VLA_PI05_ASYNC_VISION':'1','VLA_PI05_VISION_DIRECT':'1','VLA_PI05_KV_REUSE':'1'})]
# Recheck zero-copy after the internal packing optimization, before lifecycle tests.
for split in [False,True]:
 stage='mapped-combined'+('-phases' if split else '')
 env=dict(base,VLA_PI05_HOST_INPUT='mapped',VLA_PI05_ASYNC_VISION='1',VLA_PI05_VISION_DIRECT='1',BENCH_RUNS='3' if split else '5')
 if split:env['BENCH_SPLIT']='1'
 print('MEASURE',stage,flush=True)
 with (d/(stage+'.log')).open('w') as f:
  subprocess.run([sys.executable,str(d/'run_bench.py'),stage],env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
rows=[]
for mode,config in configs:
 for short,name in [('pruned','pi05_cube_merge_cksize50_step3_0929'),('full','cup_cube_joint_061_a800_50k_20261022')]:
  env=dict(base,**config,VLA_MUSA_GEMMA_ATTN='fused-denoise' if short=='pruned' else 'fused-all')
  title=mode+'-'+short
  print('CHECK',title,flush=True)
  cmd=[sys.executable,str(d/'check_lifecycle.py'),name,mode]
  (d/('lifecycle-'+title+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_'))}},indent=2)+'\n')
  with (d/('lifecycle-'+title+'.log')).open('w') as f:
   p=subprocess.run(cmd,env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  rows.append({'name':title,'returncode':p.returncode});(d/'lifecycle-status.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(title,p.returncode,flush=True)
  if p.returncode:raise SystemExit(p.returncode)
