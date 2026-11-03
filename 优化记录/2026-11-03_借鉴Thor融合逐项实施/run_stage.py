import os,sys,json,subprocess
from pathlib import Path
d=Path(__file__).resolve().parent
e=dict(os.environ)
for k in ['VLA_PI05_KV_DIRECT','VLA_MUSA_GEGLU_F16','VLA_PI05_RES_ADARMS']:e.pop(k,None)
e.update(json.loads((d/'baseline-env.json').read_text()))
for arg in sys.argv[2:]:
    k,v=arg.split('=',1);e[k]=v
subprocess.run([sys.executable,str(d/'run_bench.py'),sys.argv[1]],env=e,check=True)
