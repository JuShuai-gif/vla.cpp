from pathlib import Path
import os,subprocess,sys,json
repo=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
env=dict(os.environ,VLA_DEVICE='0',VLA_N_THREADS='4',VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_MUSA_GEMMA_ATTN='fused-all')
for k in ['VLA_PI05_INPUT_CACHE','VLA_PI05_DUMP_PRUNE','VLA_PI05_DUMP_GRAPH','VLA_MUSA_NO_VECTOR_PACK','GGML_CUDA_DISABLE_GRAPHS','VLA_MUSA_PROFILE_OPS','VLA_PI05_LANG_BUCKET']:
 env.pop(k,None)
jobs=[
 ('full-all-inputs',[sys.executable,str(d/'check_full_model_inputs.py'),'full-fused-all'],env),
 ('full-all-bucket',[sys.executable,str(d/'check_full_model_inputs.py'),'full-fused-all-bucket'],dict(env,VLA_PI05_LANG_BUCKET='16')),
 ('full-all-entry',[sys.executable,str(d/'check_full_model_entry.py')],env),
 ('full-all-traces',[sys.executable,str(d/'trace_full_model.py')],env),
]
summary=[]
for name,cmd,e in jobs:
 print('CHECK',name,flush=True)
 (d/(name+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in e.items() if k.startswith(('VLA_','GGML_'))}},indent=2)+'\n')
 with (d/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,cwd=repo,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 summary.append({'name':name,'returncode':p.returncode});(d/'validation-full-all-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
