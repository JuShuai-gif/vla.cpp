from pathlib import Path
import sys,json,subprocess
d=Path(__file__).resolve().parent;rows=[]
for mode in ['baseline','selected']:
 for model in ['pruned','full']:
  name='random-'+mode+'-'+model
  print(name,flush=True)
  with (d/(name+'.log')).open('w') as f:
   p=subprocess.run([sys.executable,str(d/'check_random.py'),mode,model],stdout=f,stderr=subprocess.STDOUT,timeout=900)
  rows.append({'name':name,'returncode':p.returncode});(d/'random-status.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(name,p.returncode,flush=True)
  if p.returncode:raise SystemExit(p.returncode)
