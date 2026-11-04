from pathlib import Path
import os,sys,json,subprocess
d=Path(__file__).resolve().parent;repo=d.parents[1];rows=[]
with(d/'build-lut.log').open('w')as f:r=subprocess.run(['cmake','--build','build-musa-official','-j','8'],cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=600)
assert r.returncode==0
for lut in [0,1]:
 for short in ['pruned','full']:
  stage='lut-'+str(lut);cmd=[sys.executable,str(d/'measure_pipeline.py'),short,stage]
  env=dict(os.environ,VLA_LOAD_THREADS='4',VLA_LOAD_PINNED_MIB='0',VLA_LOAD_BF16_LUT=str(lut),VLA_PI05_PROFILE_LOAD='1')
  (d/(stage+'-'+short+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:env[k]for k in ['VLA_LOAD_THREADS','VLA_LOAD_BF16_LUT','VLA_PI05_PROFILE_LOAD']}},indent=2)+'\n')
  with(d/(stage+'-'+short+'.log')).open('w')as f:r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
  rows.append({'stage':stage,'model':short,'returncode':r.returncode});(d/'lut-status.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True)
  assert r.returncode==0
