from pathlib import Path
import importlib.util,sys,json
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1];sys.path.insert(0,str(repo/'scripts'))
from pi05_gguf import GGUFReader
import run_pi05
spec=importlib.util.spec_from_file_location('before',d/'run_pi05_before.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rows=[]
for name in ['pi05_cube_merge_cksize50_step3_0929','cup_cube_joint_061_a800_50k_20261022']:
 base=Path('/home/skysi/VLA/models')/name/'gguf';r=GGUFReader(base/'model.gguf')
 q={t.name:t.data.reshape(-1).copy()for t in r.tensors if t.name in ['state_q01','state_q99']}
 mean=np.asarray(json.loads((base/'model.stats.json').read_text())['observation.state']['mean'],np.float32)
 for state in [mean,q['state_q01'],q['state_q99'],q['state_q01']-100,q['state_q99']+100]:
  for text in ['pick_up\nthe cube','','long task '*100]:
   a=old.prepare_prompt(r,text,state);b=run_pi05.prepare_prompt(r,text,state);assert a==b
   rows.append({'model':name,'task':text[:20],'tokens':len(b[1]),'equal':True})
(d/'prompt-correctness.json').write_text(json.dumps(rows,indent=2)+'\n');print('prompt cases',len(rows),flush=True)
