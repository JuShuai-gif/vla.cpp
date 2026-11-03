from pathlib import Path
import json,os,sys
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1]
short=sys.argv[1];env=json.loads((d/'candidate.json').read_text())[short]
os.environ.update(env,VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
sys.path.insert(0,str(repo/'bindings/python'));import vla_cpp
name='pi05_cube_merge_cksize50_step3_0929' if short=='pruned' else 'cup_cube_joint_061_a800_50k_20261022'
out=d/'formats-candidate'/short;out.mkdir(parents=True,exist_ok=False)
runtime=out/'runtime.json';runtime.write_text(json.dumps({'runtime':{'weight_dtype':'f16','act_dtype':'f32','flash_attn':True}}))
rows=[]
with vla_cpp.load('/home/skysi/VLA/models/'+name+'/gguf/model.gguf',config_path=str(runtime)) as m:
 cfg=m.config
 for seed in range(2):
  rng=np.random.default_rng(20261103+seed)
  images=[rng.integers(0,256,(224,224,3),dtype=np.uint8) for _ in range(2)]
  floats=[np.ascontiguousarray(x.astype(np.float32)/np.float32(255)) for x in images]
  state=np.zeros(cfg.max_state_dim,np.float32);state[:cfg.real_state_dim]=rng.uniform(-.3,.3,cfg.real_state_dim)
  noise=rng.normal(0,1,cfg.n_suffix*cfg.max_action_dim).astype(np.float32)
  tokens=[1,100,200,300,400,2]+([500,600,2]if seed else [])
  refs={}
  os.environ['VLA_PI05_U8_PATCH']='0'
  for nv in [1,2]:
   refs[nv]=m.predict(images[:nv],tokens=tokens,state=state,noise=noise)
  os.environ['VLA_PI05_U8_PATCH']='1'
  for nv,fmt in [(2,'u8'),(2,'f32'),(1,'u8'),(1,'f32'),(2,'u8'),(1,'u8'),(2,'f32'),(2,'u8')]:
   data=images if fmt=='u8' else floats
   a=m.predict(data[:nv],tokens=tokens,state=state,noise=noise,
               pixel_format=vla_cpp.PIXEL_U8 if fmt=='u8' else vla_cpp.PIXEL_F32_RGB_01)
   error=float(abs(a-refs[nv]).max());assert np.isfinite(a).all() and error==0,(short,seed,nv,fmt,error)
   rows.append({'seed':seed,'views':nv,'format':fmt,'max_abs':error})
   (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(short,len(rows),'format/view changes exact',flush=True)
