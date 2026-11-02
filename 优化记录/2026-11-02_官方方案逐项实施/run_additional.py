from pathlib import Path
import subprocess,os,sys,json
repo=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
base=dict(os.environ,VLA_DEVICE='0',VLA_N_THREADS='4')
for k in ['VLA_MUSA_NO_VECTOR_PACK','VLA_MUSA_VISION_SDPA','VLA_PI05_VISION_QKV','VLA_PI05_INPUT_CACHE','VLA_PI05_DUMP_PRUNE','VLA_PI05_DUMP_GRAPH','VLA_PI05_LANG_BUCKET','VLA_MUSA_PROFILE_OPS','GGML_CUDA_DISABLE_GRAPHS','BENCH_ACT','BENCH_SPLIT']:
 base.pop(k,None)
final=dict(base,VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_PI05_INPUT_CACHE='1')
jobs=[
 ('production',[sys.executable,str(d/'check_inputs.py'),'final-production'],final),
 ('production-bucket',[sys.executable,str(d/'check_inputs.py'),'final-production-bucket'],dict(final,VLA_PI05_LANG_BUCKET='16')),
 ('baseline-f16',[sys.executable,str(d/'run_bench.py'),'baseline-f16'],dict(base,VLA_MUSA_NO_VECTOR_PACK='1',BENCH_ACT='f16',BENCH_RUNS='5',BENCH_BASELINE_STAGE='baseline-f16')),
 ('final-f16-matched',[sys.executable,str(d/'run_bench.py'),'final-f16-matched'],dict(final,BENCH_ACT='f16',BENCH_RUNS='5',BENCH_BASELINE_STAGE='baseline-f16')),
 ('fallback',[sys.executable,str(d/'run_bench.py'),'fallback'],dict(base,VLA_PI05_VISION_QKV='1',VLA_PI05_INPUT_CACHE='1',BENCH_RUNS='3')),
]
summary=[]
for name,cmd,env in jobs:
 print('CHECK',name,flush=True)
 (d/(name+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_','BENCH_'))}},indent=2)+'\n')
 with (d/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 summary.append({'name':name,'returncode':p.returncode});(d/'additional-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(name,p.returncode,flush=True)
# Keep running independent checks and preserve any failed candidates.
raise SystemExit(any(x['returncode'] for x in summary))
