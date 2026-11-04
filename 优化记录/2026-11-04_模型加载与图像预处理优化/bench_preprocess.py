from pathlib import Path
import sys,time,json,importlib.util
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1];sys.path.insert(0,str(repo/'scripts'))
spec=importlib.util.spec_from_file_location('before',d/'run_pi05_before.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from pi05_preprocess import prepare_image,prepare_images,prepare_rgb,_read_rgb
paths=[Path('/home/skysi/VLA/models/pi05_cube_merge_cksize50_step3_0929/gguf/validation')/(s+'-480x640.png')for s in ['top','wrist']]
expected=[old.prepare_image(p,224)for p in paths];frames=[_read_rgb(p)for p in paths]
out=[np.empty((224,224,3),np.float32)for _ in paths]
functions={'A':lambda:[old.prepare_image(p,224)for p in paths],
           'B':lambda:[prepare_image(p,224)for p in paths],
           'C':lambda:prepare_images(paths,224),
           'D':lambda:[prepare_rgb(a,224,b)for a,b in zip(frames,out)]}
rows=[]
for key in ['A','B','C','D','D','C','B','A']:
 for _ in range(3):functions[key]()
 times=[]
 for _ in range(30):
  t=time.perf_counter();a=functions[key]();times.append((time.perf_counter()-t)*1000)
  assert all(np.array_equal(x,y)for x,y in zip(a,expected))
 rows.append({'variant':key,'wall_ms':times,'pixel_max_abs':0})
report={'scope':'warmed double-image preprocessing, A/B/C file decode included, D raw RGB excludes decode; imports excluded','runs':rows,'summary':{}}
for key in functions:
 a=[x for row in rows if row['variant']==key for x in row['wall_ms']]
 report['summary'][key]={'samples':len(a),'mean_ms':float(np.mean(a)),'p95_ms':float(np.percentile(a,95)),'max_ms':max(a)}
(d/'preprocess-acceptance.json').write_text(json.dumps(report,indent=2)+'\n');print(report['summary'],flush=True)
