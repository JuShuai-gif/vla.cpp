from pathlib import Path
import os,sys,json,subprocess,time,re
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1]
while not(d/'final-acceptance.json').exists():time.sleep(1)
rows=json.loads((d/'final-validation-status.json').read_text())
def run(stage,cmd,env):
 if any(x['stage']==stage and x['returncode']==0 for x in rows):return
 (d/(stage+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items()if k.startswith(('VLA_','GGML_'))}},indent=2)+'\n')
 with(d/(stage+'.log')).open('w')as f:r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'stage':stage,'returncode':r.returncode})
 (d/'final-validation-status.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True)
 assert r.returncode==0,rows[-1]
for mode in ['baseline','selected']:
 for short in ['pruned','full']:
  run('random-'+mode+'-'+short,[sys.executable,str(d/'check_random_f32.py'),mode,short],dict(os.environ))
for short in ['pruned','full']:
 run('mixed-lifecycle-r2-'+short,[sys.executable,str(d/'check_mixed_lifecycle.py'),short],
     dict(os.environ,LIFECYCLE_TAG='mixed-lifecycle-r2',VLA_PI05_SPLIT_PHASES='1'))
 log=(d/('mixed-lifecycle-r2-'+short+'.log')).read_text()
 phases=[json.loads(x.split(' ',1)[1])for x in log.splitlines()if x.startswith('VLA_PI05_PHASES ')]
 assert len(phases)>=11,len(phases)
 phases=phases[3:11]
 (d/('phases-'+short+'.json')).write_text(json.dumps({'protocol':'separate synchronized phase instrumentation, not ordinary wall','runs':phases,
    'means':{k:float(np.mean([x[k]for x in phases]))for k in ['vision_ms','prefix_ms','denoise_ms']}},indent=2)+'\n')
selection=json.loads((d/'selection.json').read_text())
for short in ['pruned','full']:
 name='pi05_cube_merge_cksize50_step3_0929'if short=='pruned'else 'cup_cube_joint_061_a800_50k_20261022'
 base=Path('/home/skysi/VLA/models')/name/'gguf'
 state=json.loads((base/'model.stats.json').read_text())['observation.state']['mean']
 common=[sys.executable,str(repo/'scripts/run_pi05.py'),'--model',str(base/'model.gguf'),
         '--image','top='+str(base/'validation/top-480x640.png'),'--image','wrist='+str(base/'validation/wrist-480x640.png'),
         '--text','pick up the cube','--state',','.join(map(str,state)),'--device','0','--warmup','3']
 for mode in ['f32','f16']:
  out=d/('final-cli-'+mode)/short;out.parent.mkdir(exist_ok=True)
  env=dict(os.environ)
  for k in list(env):
   if k.startswith('VLA_'):env.pop(k)
  if mode=='f32':args=['--musa-optimized']
  else:
   env.update(selection[short],VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'))
   args=['--weight-dtype','f16','--act-dtype','f16','--flash-attn']
  run('final-cli-'+mode+'-'+short,common+args+['--output',str(out.with_suffix('.json'))],env)
  got=np.asarray(json.loads(out.with_suffix('.json').read_text())['actions'],np.float32)
  refpath=d.parent/'2026-11-03_八项优化与延迟目标'/'candidate-full-entry'/name/(mode+'.json')
  expected=np.asarray(json.loads(refpath.read_text())['actions'],np.float32)
  err=float(abs(got-expected).max());assert err==0,(mode,short,err)
  print('CLI exact',mode,short,flush=True)
(d/'final-validation-complete.json').write_text(json.dumps({'passed':True,'stages':rows},indent=2)+'\n')
print('FINAL VALIDATION COMPLETE',flush=True)
