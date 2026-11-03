from pathlib import Path
import subprocess,json
d=Path(__file__).resolve().parent;rows=[]
jobs=[(8192,50,1024,1),(32768,278,2048,1),(32768,518,2048,1)]
jobs += [(m,n,k,0,blocks,mb) for m,n,k in [(2560,50,1024),(32768,518,2048)] for blocks in [8,32] for mb in [1,4,8]]
for args in jobs:
 name='tuned-'+'-'.join(map(str,args));cmd=['/tmp/pi05-probe-remaining-tuned',*map(str,args)]
 print(name,flush=True)
 with (d/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 data=[json.loads(s) for s in (d/(name+'.log')).read_text().splitlines() if s.startswith('{')]
 rows.append({'command':cmd,'args':args,'returncode':p.returncode,'measurements':data})
 (d/'调参候选结果.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(p.returncode,data[-1] if data else None,flush=True)
