from pathlib import Path
import os,subprocess,sys,json
repo=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
base=dict(os.environ,VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',BENCH_RUNS='5')
for k in ['VLA_MUSA_GEMMA_ATTN','VLA_PI05_INPUT_CACHE','VLA_MUSA_NO_VECTOR_PACK','VLA_PI05_DUMP_PRUNE','VLA_PI05_DUMP_GRAPH','GGML_CUDA_DISABLE_GRAPHS','VLA_MUSA_PROFILE_OPS','VLA_PI05_LANG_BUCKET','BENCH_SPLIT','BENCH_ACT']:
 base.pop(k,None)
summary=[]
for mode in ['fused-denoise','fused-all']:
 env=dict(base)
 if mode!='baseline':env['VLA_MUSA_GEMMA_ATTN']=mode
 cmd=[sys.executable,str(d/'run_bench.py'),mode]
 print('PILOT',mode,flush=True)
 with (d/(mode+'.log')).open('w') as f:
  p=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 summary.append({'stage':mode,'returncode':p.returncode});(d/'fused-status.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(mode,p.returncode,flush=True)
 # An accuracy failure rejects this candidate but does not stop independent modes.
raise SystemExit(any(x['returncode'] for x in summary))
