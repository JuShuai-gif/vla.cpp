from pathlib import Path
import json,sys,subprocess,os,time
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1]
while not(d/'acceptance.json').exists():time.sleep(1)
for p in d.glob('entry-*.json'):
 if p.name.endswith(('-command.json','.actions.json')):continue
 a=json.loads(p.read_text());assert len(a['loaded_core'])==1,a
with(d/'build-diagnostics.log').open('w')as f:r=subprocess.run(['cmake','--build','build-musa-official','-j','8'],cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=600)
assert r.returncode==0
# Only diagnostic formatting changed since the complete 32-test run.
with(d/'test-weights-final.log').open('w')as f:
 r=subprocess.run(['ctest','--test-dir','build-musa-official','-R','^weight_loading$','--output-on-failure'],cwd=repo,stdout=f,stderr=subprocess.STDOUT)
assert r.returncode==0
rows=[]
for short in ['pruned','full']:
 cmd=[sys.executable,str(d/'check_random.py'),'selected',short]
 with(d/('random-'+short+'.log')).open('w')as f:r=subprocess.run(cmd,cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 assert r.returncode==0
 a=json.loads((d/'random-f32-selected'/short/'summary.json').read_text())
 assert len(a)==14 and max(x['max_abs']for x in a)==0,a
 rows.append({'stage':'random-'+short,'cases':len(a),'max_abs':0})
 name='pi05_cube_merge_cksize50_step3_0929'if short=='pruned'else 'cup_cube_joint_061_a800_50k_20261022'
 base=Path('/home/skysi/VLA/models')/name/'gguf'
 state=json.loads((base/'model.stats.json').read_text())['observation.state']['mean']
 env=dict(os.environ,**json.loads((repo/'scripts/pi05_musa_profiles.json').read_text())[short],
          VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'))
 out=d/('final-f16-'+short+'.json')
 cmd=[sys.executable,str(repo/'scripts/run_pi05.py'),'--model',str(base/'model.gguf'),
      '--image','top='+str(base/'validation/top-480x640.png'),'--image','wrist='+str(base/'validation/wrist-480x640.png'),
      '--text','pick up the cube','--state',','.join(map(str,state)),'--weight-dtype','f16','--act-dtype','f16','--flash-attn',
      '--warmup','3','--output',str(out)]
 with(d/('final-f16-'+short+'.log')).open('w')as f:r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 assert r.returncode==0
 ref=d.parent/'2026-11-03_四项深化与MCU热点优化/final-cli-f16'/(short+'.json')
 assert np.array_equal(np.asarray(json.loads(out.read_text())['actions'],np.float32),
                       np.asarray(json.loads(ref.read_text())['actions'],np.float32))
 rows.append({'stage':'f16-entry-'+short,'max_abs':0})
 (d/'verification-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(rows[-2:],flush=True)
(d/'verification-complete.json').write_text(json.dumps({'passed':True,'stages':rows},indent=2)+'\n')
