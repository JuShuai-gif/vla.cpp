import os,sys,json,subprocess
from pathlib import Path
d=Path(__file__).resolve().parent
e=dict(os.environ);e.update(json.loads((d/'baseline-env.json').read_text()))
for k in ['VLA_PI05_CROSS_ADARMS','VLA_PI05_ACTION_EDGES','VLA_PI05_ACTION_LINEAR','VLA_PI05_F16_PART']:e.pop(k,None)
for arg in sys.argv[2:]:
 k,v=arg.split('=',1);e[k]=v
subprocess.run([sys.executable,str(d/'run_bench.py'),sys.argv[1]],env=e,check=True)
