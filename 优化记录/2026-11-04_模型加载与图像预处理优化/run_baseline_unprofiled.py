from pathlib import Path
import subprocess,sys,json,os
d=Path(__file__).resolve().parent;rows=[]
for short in ['pruned','full']:
 cmd=[sys.executable,str(d/'measure_pipeline.py'),short,'baseline-clean']
 (d/('baseline-clean-'+short+'-command.json')).write_text(json.dumps({'argv':cmd},indent=2)+'\n')
 with(d/('baseline-clean-'+short+'.log')).open('w')as f:r=subprocess.run(cmd,env=dict(os.environ,MEASURE_LIBRARY='/home/skysi/VLA/startup-preprocess-20261104/baseline-lib/libvla.so',LD_LIBRARY_PATH='/home/skysi/VLA/startup-preprocess-20261104/baseline-lib:/usr/local/musa/lib'),stdout=f,stderr=subprocess.STDOUT,timeout=600)
 rows.append({'model':short,'returncode':r.returncode});(d/'baseline-clean-status.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True)
 assert r.returncode==0
