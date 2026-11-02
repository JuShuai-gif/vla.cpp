from pathlib import Path
import os,subprocess,sys,json
r=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
base=dict(os.environ,VLA_DEVICE='0',VLA_N_THREADS='4',VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_MUSA_VISION_EPILOGUE='all',BENCH_RUNS='15')
for k in ['VLA_PI05_PIN_MAIN_INPUT','VLA_MUSA_FLATTEN_GEMM','VLA_MUSA_ATTN_DIRECT_OUT','VLA_MUSA_VISION_PACK','VLA_PI05_HOST_INPUT','VLA_PI05_ASYNC_VISION','VLA_PI05_VISION_DIRECT','VLA_PI05_KV_REUSE','VLA_PI05_INPUT_CACHE','BENCH_SPLIT','VLA_PI05_LANG_BUCKET']:base.pop(k,None)
checks=json.loads((d/'validation-summary.json').read_text());life=json.loads((d/'lifecycle-status.json').read_text())
assert len(checks)==13 and all(x['returncode']==0 for x in checks)
assert len(life)==14 and all(x['returncode']==0 for x in life)
rows=[]
for stage in ['baseline-r1','final-r1','final-r2','baseline-r2','compare-baseline-phases','compare-final-phases']:
 env=dict(base)
 if 'final' in stage:env['BENCH_SELECTED']='1'
 if stage.endswith('phases'):env.update(BENCH_RUNS='5',BENCH_SPLIT='1',VLA_PI05_DUMP_GRAPH='1')
 print('MEASURE',stage,flush=True)
 with (d/(stage+'.log')).open('w') as f:
  p=subprocess.run([sys.executable,str(d/'run_bench.py'),stage],env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'stage':stage,'returncode':p.returncode});(d/'comparison-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(stage,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
