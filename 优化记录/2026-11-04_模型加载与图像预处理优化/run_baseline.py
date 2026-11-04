from pathlib import Path
import subprocess,sys,json
d=Path(__file__).resolve().parent;rows=[]
for short in ['pruned','full']:
 cmd=[sys.executable,str(d/'measure_pipeline.py'),short,'baseline']
 (d/('baseline-'+short+'-command.json')).write_text(json.dumps({'argv':cmd},indent=2)+'\n')
 with(d/('baseline-'+short+'.log')).open('w')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 rows.append({'model':short,'returncode':r.returncode});(d/'baseline-status.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True)
 assert r.returncode==0
