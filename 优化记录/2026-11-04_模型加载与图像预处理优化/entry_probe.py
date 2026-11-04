import time
entry_start=time.perf_counter()
from pathlib import Path
import sys,os,json,importlib.util,resource
d=Path(__file__).resolve().parent;repo=d.parents[1];variant,short,dest=sys.argv[1:4]
sys.path.insert(0,str(repo/'scripts'));sys.path.insert(0,str(repo/'bindings/python'))
timing={}
if variant=='A':
 spec=importlib.util.spec_from_file_location('before',d/'run_pi05_before.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.REPO=repo;original_reader=module.gguf.GGUFReader
 def reader(*a,**k):
  t=time.perf_counter();v=original_reader(*a,**k);timing['metadata_ms']=(time.perf_counter()-t)*1000;return v
 module.gguf.GGUFReader=reader
else:
 import run_pi05 as module
 original_reader=module.GGUFReader
 def reader(*a,**k):
  t=time.perf_counter();v=original_reader(*a,**k);timing['metadata_ms']=(time.perf_counter()-t)*1000;return v
 module.GGUFReader=reader
import vla_cpp
load=vla_cpp.load
def loading(*a,**k):
 t=time.perf_counter();v=load(*a,**k);timing['cpp_load_ms']=(time.perf_counter()-t)*1000;return v
vla_cpp.load=loading
predict=vla_cpp.Model.predict
def predicted(self,*a,**k):
 t=time.perf_counter();v=predict(self,*a,**k)
 timing['first_predict_ms']=(time.perf_counter()-t)*1000
 timing['first_action_from_entry_ms']=(time.perf_counter()-entry_start)*1000
 # Check steady resident latency without image decode or tokenizer work.
 hot=[]
 for i in range(23):
  t=time.perf_counter();r=predict(self,*a,**k);ms=(time.perf_counter()-t)*1000
  import numpy as np
  assert np.array_equal(v,r)
  if i>=3:hot.append(ms)
 timing['resident_ms']=hot
 return v
vla_cpp.Model.predict=predicted
name='pi05_cube_merge_cksize50_step3_0929'if short=='pruned'else 'cup_cube_joint_061_a800_50k_20261022'
base=Path('/home/skysi/VLA/models')/name/'gguf'
state=json.loads((base/'model.stats.json').read_text())['observation.state']['mean']
sys.argv=['run_pi05.py','--model',str(base/'model.gguf'),'--image','top='+str(base/'validation/top-480x640.png'),
          '--image','wrist='+str(base/'validation/wrist-480x640.png'),'--text','pick up the cube','--state',','.join(map(str,state)),
          '--musa-optimized','--output',str(Path(dest).with_suffix('.actions.json'))]
module.main()
timing.update(variant=variant,model=short,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              loaded_core=sorted(set(x.split()[-1]for x in Path('/proc/self/maps').read_text().splitlines()if 'libvla_core'in x)))
Path(dest).write_text(json.dumps(timing,indent=2)+'\n');print(timing,flush=True)
