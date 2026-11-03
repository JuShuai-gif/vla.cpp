from pathlib import Path
import json,hashlib,os
import numpy as np
d=Path(__file__).resolve().parent;old=d.parent/'2026-11-02_前缀与动作Attention优化'
rows=[]
for name,folder in [('pi05_cube_merge_cksize50_step3_0929','full-entry'),('cup_cube_joint_061_a800_50k_20261022','full-entry-fused-all')]:
 for dtype in ['f32','f16']:
  src=old/folder/name/(dtype+'.json');dest=d/(os.environ.get('CHECK_TAG','')+'full-entry')/name/(dtype+'.json')
  ref=json.loads(src.read_text());got=json.loads(dest.read_text())
  a,b=np.array(got['actions']),np.array(ref['actions'])
  assert got['token_ids']==ref['token_ids'] and got['camera_order']==ref['camera_order']
  assert (float(abs(a-b).max())/max(float(abs(b).max()),1e-12)<.001 if json.loads((d/os.environ.get('CONFIG_SELECTION','selection.json')).read_text())['pruned' if name.startswith('pi05_') else 'full'].get('VLA_PI05_ACTION_LINEAR')=='1' else np.array_equal(a,b)),(name,dtype,float(abs(a-b).max()))
  rows.append({'model':name,'act_dtype':dtype,'max_abs':float(abs(a-b).max()),'reference':str(src),'reference_sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
(d/(os.environ.get('CHECK_TAG','')+'完整入口基线一致性.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rows,ensure_ascii=False,indent=2))
