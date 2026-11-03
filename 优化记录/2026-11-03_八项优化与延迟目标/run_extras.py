from pathlib import Path
import json,os,sys,subprocess
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1];rows=[]
for short,name in [('pruned','pi05_cube_merge_cksize50_step3_0929'),('full','cup_cube_joint_061_a800_50k_20261022')]:
 x=json.loads((d/'candidate-full-entry'/name/'f32-command.json').read_text());cmd=x['argv']
 dest=d/'optimized-launcher'/name/'f32.json';dest.parent.mkdir(parents=True,exist_ok=True)
 cmd[cmd.index('--output')+1]=str(dest);cmd.append('--musa-optimized')
 env={k:v for k,v in os.environ.items()if not k.startswith(('VLA_','GGML_'))}
 (dest.parent/'command.json').write_text(json.dumps({'argv':cmd,'env_profile':'--musa-optimized (clean VLA_/GGML_ environment)'},indent=2)+'\n')
 with (dest.parent/'run.log').open('w')as f:subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
 a=json.loads(dest.read_text());b=json.loads((d/'candidate-full-entry'/name/'f32.json').read_text())
 assert a['musa_optimized_profile']==short and a['repeat_max_abs']==0
 assert a['weight_dtype']=='f16' and a['act_dtype']=='f32' and a['flash_attn']
 assert a['token_ids']==b['token_ids'] and a['camera_order']==b['camera_order']
 assert np.array_equal(a['actions'],b['actions'])
 rows.append({'model':short,'profile':a['musa_optimized_profile'],'timing_ms':a['timing_ms'],'max_abs':0})
 (d/'optimized-launcher-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for short in ['pruned','full']:
 with (d/('embedding-'+short+'.log')).open('w')as f:subprocess.run([sys.executable,str(d/'check_embedding.py'),short],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
stages=[('late-k-rope',{'VLA_PI05_K_ROPE_DIRECT':'1'}),
 ('late-prefetch64',{'VLA_PI05_PREFETCH_KIB':'64'}),
 ('late-prefetch4096',{'VLA_PI05_PREFETCH_KIB':'4096'}),
 ('selected-split',{})]
for stage,options in stages:
 e=dict(os.environ,BENCH_CONFIG='candidate.json',BENCH_RUNS='8')
 if stage=='selected-split':e['BENCH_SPLIT']='1'
 cmd=[sys.executable,str(d/'run_bench.py'),stage]+[k+'='+v for k,v in options.items()]
 with (d/(stage+'.log')).open('w')as f:p=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'stage':stage,'returncode':p.returncode})
 (d/'extras-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)
print('EXTRAS COMPLETE',flush=True)
