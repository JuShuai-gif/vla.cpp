from pathlib import Path
import json,os,sys,subprocess,time,collections
d=Path(__file__).resolve().parent;repo=d.parents[1]
while not(d/'final-validation-complete.json').exists():time.sleep(1)
assert json.loads((d/'final-validation-complete.json').read_text())['passed']
msys='/opt/moore-threads/moore-perf-system/1.7.2/target-linux-aarch64/msys';raw=Path('/home/skysi/VLA/four-deep-20261103')
rows=[]
for short in ['pruned','full']:
 stage='selected-msys-'+short
 cmd=[msys,'profile','-t','musa','-o',str(raw/stage),sys.executable,str(d/'bench_real.py'),short,str(d/stage)]
 env=dict(os.environ,BENCH_CONFIG='selection.json',BENCH_WARMUP='1',BENCH_RUNS='1',
          GGML_CUDA_DISABLE_GRAPHS='1',PROFILE_MARKERS=str(raw/'profile_markers.so'),
          LD_LIBRARY_PATH='/usr/local/musa/lib:/usr/local/musa/lib64:'+os.environ.get('LD_LIBRARY_PATH',''))
 (d/(stage+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items()if k.startswith(('VLA_','GGML_','BENCH_','PROFILE_'))}},indent=2)+'\n')
 with(d/(stage+'.log')).open('w')as f:r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 assert r.returncode==0
 with(d/(stage+'-export.log')).open('w')as f:
  subprocess.run([msys,'export','-t','json','-o',str(raw/(stage+'.json')),str(raw/(stage+'.msys-rep'))],cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
 a=json.loads((raw/(stage+'.json')).read_text())['traceEvents']
 begin=[x for x in a if x.get('ph')=='X'and 'vla_r2_begin'in x.get('name','')]
 end=[x for x in a if x.get('ph')=='X'and 'vla_r2_end'in x.get('name','')]
 assert len(begin)==len(end)==1
 lo=begin[0]['ts']+begin[0]['dur'];hi=end[0]['ts']
 ev=[x for x in a if x.get('ph')=='X'and 'Block'in x.get('args',{})and lo<=x['ts']<hi]
 counts=collections.Counter(x['name']for x in ev)
 result={'model':short,'kernel_count':len(ev),'timing_valid':False,'scope':'marked warmed Graph OFF real predict, counts only',
         'kernel_counts':dict(counts)}
 (d/(stage+'-counts.json')).write_text(json.dumps(result,indent=2)+'\n')
 rows.append({'stage':stage,'returncode':r.returncode,'kernel_count':len(ev)})
 (d/'selected-msys-status.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True)
