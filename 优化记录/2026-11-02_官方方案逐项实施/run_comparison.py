from pathlib import Path
import subprocess,os,sys,json
repo=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
base=dict(os.environ,VLA_DEVICE='0',VLA_N_THREADS='4',BENCH_RUNS='15')
for key in ['VLA_MUSA_NO_VECTOR_PACK','VLA_MUSA_VISION_SDPA','VLA_PI05_VISION_QKV','VLA_PI05_INPUT_CACHE','VLA_MUSA_PROFILE_OPS','VLA_PI05_DUMP_GRAPH','VLA_PI05_DUMP_PRUNE','GGML_CUDA_DISABLE_GRAPHS','VLA_PI05_LANG_BUCKET']:
 base.pop(key,None)
modes={
 'baseline':dict(VLA_MUSA_NO_VECTOR_PACK='1'),
 'final':dict(VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_PI05_INPUT_CACHE='1'),
 'nocache':dict(VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1'),
 'fallback':dict(VLA_PI05_VISION_QKV='1',VLA_PI05_INPUT_CACHE='1',BENCH_RUNS='3'),
}
stages=[('baseline-r1','baseline'),('final-r1','final'),('nocache-r1','nocache'),('nocache-r2','nocache'),('final-r2','final'),('baseline-r2','baseline'),
        ('final-phases','final'),('final-f16','final'),('fallback','fallback')]
summary=[]
for stage,mode in stages:
 env=dict(base,**modes[mode])
 if stage=='final-phases':env.update(BENCH_RUNS='5',BENCH_SPLIT='1')
 if stage=='final-f16':env.update(BENCH_RUNS='5',BENCH_ACT='f16')
 print('MEASURE',stage,flush=True)
 cmd=[sys.executable,str(d/'run_bench.py'),stage]
 with (d/(stage+'.log')).open('w') as log:
  r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 summary.append({'stage':stage,'returncode':r.returncode});(d/'comparison-status.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(stage,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
