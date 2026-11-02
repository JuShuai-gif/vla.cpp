from pathlib import Path
import os,subprocess,sys,json
d=Path(__file__).resolve().parent;repo=d.parents[1]
jobs=[
 ('before-main-pin-profile',[sys.executable,str(d/'profile_io.py'),'before-main-pin'],dict(os.environ,PROFILE_SELECTED='1')),
 ('before-main-pin-summary',[sys.executable,str(d/'summarize_io.py'),'before-main-pin'],os.environ.copy()),
 ('mixed-q-final',['/tmp/vla-probe-mixed-q-final'],os.environ.copy()),
 ('gemm-shapes',[sys.executable,str(d/'run_gemm_shapes.py')],os.environ.copy()),
]
rows=[]
for name,cmd,env in jobs:
 print('CHECK',name,flush=True)
 with (d/(name+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 rows.append({'name':name,'returncode':p.returncode});(d/'extra-probes-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
