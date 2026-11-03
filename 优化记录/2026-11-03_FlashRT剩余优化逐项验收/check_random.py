from pathlib import Path
import os,sys,json
import numpy as np
d=Path(__file__).resolve().parent;r=d.parents[1];mode,short=sys.argv[1:]
os.environ.update(VLA_LIBRARY=str(r/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
env=json.loads((d/('baseline-env.json' if mode=='baseline' else 'selection.json')).read_text())
if mode!='baseline':env=env[short]
for k in ['VLA_PI05_CROSS_ADARMS','VLA_PI05_ACTION_EDGES','VLA_PI05_ACTION_LINEAR','VLA_PI05_F16_PART']:os.environ.pop(k,None)
os.environ.update(env);os.environ['VLA_MUSA_GEMMA_ATTN']='fused-denoise' if short=='pruned' else 'fused-all'
name='pi05_cube_merge_cksize50_step3_0929' if short=='pruned' else 'cup_cube_joint_061_a800_50k_20261022'
out=d/('random-'+mode)/short;out.mkdir(parents=True,exist_ok=False)
runtime=out/'runtime.json';runtime.write_text(json.dumps({'runtime':{'weight_dtype':'f16','act_dtype':'f32','flash_attn':True}}))
sys.path.insert(0,str(r/'bindings/python'));import vla_cpp
rows=[]
with vla_cpp.load('/home/skysi/VLA/models/'+name+'/gguf/model.gguf',config_path=str(runtime)) as m:
 cfg=m.config
 for seed in range(10):
  rng=np.random.default_rng(20261103+seed)
  images=[rng.integers(0,256,(224,224,3),dtype=np.uint8) for _ in range(2)]
  state=np.zeros(cfg.max_state_dim,np.float32);state[:cfg.real_state_dim]=rng.uniform(-.3,.3,cfg.real_state_dim)
  noise=rng.normal(0,1,(cfg.n_suffix*cfg.max_action_dim)).astype(np.float32)
  tokens=[1]+rng.integers(10,1000,4+(seed%2)*3).tolist()+[2]
  a=m.predict(images,tokens=tokens,state=state,noise=noise)
  b=m.predict(images,tokens=tokens,state=state,noise=noise);assert np.array_equal(a,b) and np.isfinite(a).all()
  np.save(out/(str(seed)+'.npy'),a)
  row={'seed':seed,'repeat_max_abs':0}
  if mode!='baseline':
   ref=np.load(d/'random-baseline'/short/(str(seed)+'.npy'))
   row.update(max_abs=float(abs(a-ref).max()),relative_peak=float(abs(a-ref).max()/max(abs(ref).max(),1e-12)),rmse=float(np.sqrt(np.mean((a-ref)**2))))
   assert (row['relative_peak']<.001 if env.get('VLA_PI05_ACTION_LINEAR')=='1' else row['max_abs']==0),row
  rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(short,mode,rows,flush=True)
