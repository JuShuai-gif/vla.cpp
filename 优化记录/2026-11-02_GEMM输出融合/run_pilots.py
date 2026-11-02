from pathlib import Path
import os,subprocess,sys
d=Path(__file__).resolve().parent;repo=d.parents[1]
base=dict(os.environ,VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1')
base.pop('VLA_PI05_INPUT_CACHE',None)
for mode in ['gelu','residual','all']:
 print('PILOT',mode,flush=True)
 with (d/('pilot-'+mode+'.log')).open('w') as f:
  subprocess.run([sys.executable,str(d/'run_bench.py'),'pilot-'+mode],cwd=repo,env=dict(base,VLA_MUSA_VISION_EPILOGUE=mode),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
 print((d/('pilot-'+mode)/'summary.json').read_text(),flush=True)
