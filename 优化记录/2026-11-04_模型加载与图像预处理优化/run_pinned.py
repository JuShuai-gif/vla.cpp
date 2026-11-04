from pathlib import Path
import os,sys,json,subprocess
d=Path(__file__).resolve().parent;repo=d.parents[1];rows=[]
with(d/'build-pinned.log').open('w')as f:r=subprocess.run(['cmake','--build','build-musa-official','-j','8'],cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=600)
assert r.returncode==0
for pin in [0,32]:
 for short in ['pruned','full']:
  stage='pin-'+str(pin);cmd=[sys.executable,str(d/'measure_pipeline.py'),short,stage]
  env=dict(os.environ,VLA_LOAD_THREADS='4',VLA_LOAD_PINNED_MIB=str(pin),VLA_PI05_PROFILE_LOAD='1')
  (d/(stage+'-'+short+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:env[k]for k in ['VLA_LOAD_THREADS','VLA_LOAD_PINNED_MIB','VLA_PI05_PROFILE_LOAD']}},indent=2)+'\n')
  with(d/(stage+'-'+short+'.log')).open('w')as f:r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
  rows.append({'stage':stage,'model':short,'returncode':r.returncode});(d/'pin-status.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True)
  assert r.returncode==0
