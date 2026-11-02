from pathlib import Path
import os,subprocess,sys,json
repo=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
env=dict(os.environ,VLA_DEVICE='0',VLA_N_THREADS='4',VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_MUSA_GEMMA_ATTN='fused-denoise')
for k in ['VLA_PI05_INPUT_CACHE','VLA_PI05_DUMP_PRUNE','VLA_PI05_DUMP_GRAPH','VLA_MUSA_NO_VECTOR_PACK','GGML_CUDA_DISABLE_GRAPHS','VLA_MUSA_PROFILE_OPS','VLA_PI05_LANG_BUCKET']:
 env.pop(k,None)
jobs=[
 ('ctest',['ctest','--test-dir','build-musa-official','--output-on-failure'],env),
 ('changing-final',[sys.executable,str(d/'check_inputs.py'),'final'],env),
 ('changing-final-bucket',[sys.executable,str(d/'check_inputs.py'),'final-bucket'],dict(env,VLA_PI05_LANG_BUCKET='16')),
 ('full-entry',[sys.executable,str(d/'check_full_entry.py')],env),
 ('trace-graphs',[sys.executable,str(d/'trace_graphs.py')],env),
]
summary=[]
for name,cmd,e in jobs:
 print('CHECK',name,flush=True)
 (d/(name+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in e.items() if k.startswith(('VLA_','GGML_'))}},indent=2)+'\n')
 with (d/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,cwd=repo,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 summary.append({'name':name,'returncode':p.returncode});(d/'validation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
