from pathlib import Path
import json,os,sys,subprocess,time
d=Path(__file__).resolve().parent;repo=d.parents[1]
while not(d/'focused-mcu-status.json').exists()or len(json.loads((d/'focused-mcu-status.json').read_text()))<12:time.sleep(1)
rows=[{'stage':'focus-attention-'+v,'returncode':0,'valid_kernel_report':True}for v in ['all','all-parallel']]
for v in ['all','all-parallel']:
 raw=(d/('focus-attention-'+v+'.log')).read_text()
 assert 'Duration'in raw and '==ERROR=='not in raw and 'No kernels were profiled'not in raw
for short in ['pruned','full']:
 name='hot-combined-r2-'+short
 cmd=['/usr/bin/mcu','--dump-mode','normal','--kernel-name','regex:(small_gemm_kernel|packed_norm_kernel|packed_geglu_kernel|vision_scores_kernel|vla_r2_begin|vla_r2_end)','-o','/home/skysi/VLA/four-deep-20261103/'+name,sys.executable,str(d/'bench_real.py'),short,str(d/('mcu-'+name))]
 env=dict(os.environ,BENCH_CONFIG='combined-pipe4-env.json',BENCH_WARMUP='1',BENCH_RUNS='1',GGML_CUDA_DISABLE_GRAPHS='1',LD_LIBRARY_PATH='/usr/local/musa/lib:/usr/local/musa/lib64:'+os.environ.get('LD_LIBRARY_PATH',''),PROFILE_MARKERS='/home/skysi/VLA/four-deep-20261103/profile_markers.so')
 (d/(name+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:env[k]for k in ['BENCH_CONFIG','BENCH_WARMUP','BENCH_RUNS','GGML_CUDA_DISABLE_GRAPHS','PROFILE_MARKERS']}},indent=2)+'\n')
 with(d/(name+'.log')).open('w')as f:r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 raw=(d/(name+'.log')).read_text()
 valid='Duration'in raw and '==ERROR=='not in raw and 'No kernels were profiled'not in raw
 rows.append({'stage':name,'returncode':r.returncode,'valid_kernel_report':valid});print(rows[-1],flush=True)
 (d/'hot-mcu-status-valid.json').write_text(json.dumps(rows,indent=2)+'\n')

assert all(x['valid_kernel_report']for x in rows)
