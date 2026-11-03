from pathlib import Path
import os,sys,json
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1];mode,short=sys.argv[1:3]
env=json.loads((d/('baseline-env.json'if mode=='baseline'else 'selection.json')).read_text())[short]
os.environ.update(env,VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
sys.path.insert(0,str(repo/'bindings/python'));import vla_cpp
name='pi05_cube_merge_cksize50_step3_0929'if short=='pruned'else 'cup_cube_joint_061_a800_50k_20261022'
out=d/('random-f32-'+mode)/short;out.mkdir(parents=True,exist_ok=False)
runtime=out/'runtime.json';runtime.write_text(json.dumps({'runtime':{'weight_dtype':'f16','act_dtype':'f32','flash_attn':True}}))
rows=[]
with vla_cpp.load('/home/skysi/VLA/models/'+name+'/gguf/model.gguf',config_path=str(runtime))as m:
 cfg=m.config
 for seed in range(14):
  rng=np.random.default_rng(2026110300+seed)
  images=[rng.random((224,224,3),dtype=np.float32)for _ in range(2)] if seed<10 else [
     rng.integers(0,256,(224,224,3),dtype=np.uint8)for _ in range(2)]
  noise=rng.standard_normal(cfg.n_suffix*cfg.max_action_dim).astype(np.float32)
  state=np.zeros(cfg.max_state_dim,np.float32);state[:cfg.real_state_dim]=rng.uniform(-.3,.3,cfg.real_state_dim)
  tokens=[1]+rng.integers(10,1000,[4,7,40,53,20][seed%5]).tolist()+[2]
  kw=dict(tokens=tokens,state=state,noise=noise,pixel_format=vla_cpp.PIXEL_F32_RGB_01 if seed<10 else vla_cpp.PIXEL_U8)
  a=m.predict(images,**kw);b=m.predict(images,**kw)
  assert np.isfinite(a).all()and np.array_equal(a,b)
  np.save(out/(str(seed)+'.npy'),a);row={'seed':seed,'tokens':len(tokens),'repeat_max_abs':0}
  if mode!='baseline':
   ref=np.load(d/'random-f32-baseline'/short/(str(seed)+'.npy'))
   err=float(abs(a-ref).max());peak=err/max(float(abs(ref).max()),1e-12)
   row.update(max_abs=err,relative_peak=peak);assert peak<.001,row
  rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(short,mode,rows,flush=True)
