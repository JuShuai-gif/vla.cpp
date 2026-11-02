from pathlib import Path
import os,subprocess,sys,json
repo=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
base=dict(os.environ,VLA_DEVICE='0',VLA_N_THREADS='4',VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',BENCH_RUNS='15')
for k in ['VLA_MUSA_GEMMA_ATTN','VLA_PI05_INPUT_CACHE','VLA_PI05_DUMP_PRUNE','VLA_PI05_DUMP_GRAPH','VLA_MUSA_NO_VECTOR_PACK','GGML_CUDA_DISABLE_GRAPHS','VLA_MUSA_PROFILE_OPS','VLA_PI05_LANG_BUCKET','BENCH_ACT','BENCH_SPLIT','BENCH_FULL_ATTN']:
 base.pop(k,None)
full_checks=json.loads((d/'validation-full-all-summary.json').read_text())
full_all_ok=len(full_checks)==4 and all(x['returncode']==0 for x in full_checks)
summary=[]
for stage in ['baseline-r1','final-r1','final-r2','baseline-r2','baseline-phases','final-phases']:
 env=dict(base)
 if stage.startswith('final'):
  env['VLA_MUSA_GEMMA_ATTN']='fused-denoise'
  if full_all_ok:env['BENCH_FULL_ATTN']='fused-all'
 if stage.endswith('-phases'):env.update(BENCH_RUNS='5',BENCH_SPLIT='1')
 print('MEASURE',stage,flush=True)
 with (d/(stage+'.log')).open('w') as f:
  p=subprocess.run([sys.executable,str(d/'run_bench.py'),stage],cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 summary.append({'stage':stage,'returncode':p.returncode});(d/'comparison-status.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(stage,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
