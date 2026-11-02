from pathlib import Path
import os,subprocess,sys,json
repo=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
base=dict(os.environ,VLA_DEVICE='0',VLA_N_THREADS='4')
for k in ['VLA_MUSA_NO_VECTOR_PACK','VLA_MUSA_VISION_SDPA','VLA_PI05_VISION_QKV','VLA_PI05_INPUT_CACHE','VLA_PI05_DUMP_PRUNE','VLA_PI05_LANG_BUCKET','VLA_MUSA_PROFILE_OPS','GGML_CUDA_DISABLE_GRAPHS']:
 base.pop(k,None)
final=dict(base,VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_PI05_INPUT_CACHE='1')
jobs=[
 ('ctest',[ 'ctest','--test-dir','build-musa-official','--output-on-failure'],final),
 ('changing-baseline',[sys.executable,str(d/'check_inputs.py'),'baseline'],dict(base,VLA_MUSA_NO_VECTOR_PACK='1',VLA_PI05_DUMP_PRUNE='1')),
 ('changing-final',[sys.executable,str(d/'check_inputs.py'),'final'],dict(final,VLA_PI05_DUMP_PRUNE='1')),
 ('changing-baseline-bucket',[sys.executable,str(d/'check_inputs.py'),'baseline-bucket'],dict(base,VLA_MUSA_NO_VECTOR_PACK='1',VLA_PI05_DUMP_PRUNE='1',VLA_PI05_LANG_BUCKET='16')),
 ('changing-final-bucket',[sys.executable,str(d/'check_inputs.py'),'final-bucket'],dict(final,VLA_PI05_DUMP_PRUNE='1',VLA_PI05_LANG_BUCKET='16')),
 ('full-entry',[sys.executable,str(d/'check_full_entry.py')],final),
 ('trace-graphs',[sys.executable,str(d/'trace_graphs.py')],final),
]
summary=[]
for name,cmd,env in jobs:
 print('CHECK',name,flush=True)
 (d/(name+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_'))}},indent=2)+'\n')
 with (d/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 summary.append({'name':name,'returncode':p.returncode});(d/'validation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
