from pathlib import Path
import os,sys,subprocess,json,time
d=Path(__file__).resolve().parent;repo=d.parents[1]
raw=Path('/home/skysi/VLA/four-deep-20261103');raw.mkdir(exist_ok=False)
msys='/opt/moore-threads/moore-perf-system/1.7.2/target-linux-aarch64/msys'
rows=[]
for short in ['pruned','full']:
 for mode in ['baseline','msys','mcu']:
  stage=mode+'-'+short;out=d/stage
  env=dict(os.environ,PATH='/usr/bin:/bin:'+os.environ.get('PATH',''),BENCH_CONFIG='baseline-env.json',BENCH_WARMUP='3',BENCH_RUNS='8')
  cmd=[sys.executable,str(d/'bench_real.py'),short,str(out)]
  if mode!='baseline':
   env.update(GGML_CUDA_DISABLE_GRAPHS='1',PROFILE_MARKERS='/tmp/pi05-r2-markers.so',BENCH_WARMUP='1',BENCH_RUNS='1')
   if mode=='msys':cmd=[msys,'profile','-t','musa','-o',str(raw/('baseline-'+short)),*cmd]
   else:cmd=['/usr/bin/mcu','--dump-mode','normal','-o',str(raw/('baseline-'+short)),*cmd]
  (d/(stage+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items()if k.startswith(('VLA_','GGML_','BENCH_','PROFILE_'))}},indent=2)+'\n')
  print('START',stage,flush=True);begin=time.time()
  with (d/(stage+'.log')).open('w')as f:
   p=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
  rows.append({'stage':stage,'returncode':p.returncode,'elapsed_seconds':time.time()-begin})
  (d/'profile-status.json').write_text(json.dumps(rows,indent=2)+'\n')
  print('END',stage,p.returncode,flush=True)
  if p.returncode:raise SystemExit(p.returncode)
  if mode=='msys':
   with (d/(stage+'-export.log')).open('w')as f:
    subprocess.run([msys,'export','-t','json','-o',str(raw/('baseline-'+short+'.json')),str(raw/('baseline-'+short+'.msys-rep'))],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
print('BASELINE PROFILE COMPLETE',flush=True)
