from pathlib import Path
import os,json,sys
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1];short=sys.argv[1]
os.environ.update(json.loads((d/'candidate.json').read_text())[short],
 VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
sys.path.insert(0,str(repo/'bindings/python'));import vla_cpp
name='pi05_cube_merge_cksize50_step3_0929' if short=='pruned' else 'cup_cube_joint_061_a800_50k_20261022'
out=d/'embedding-candidate'/short;out.mkdir(parents=True,exist_ok=False)
cfgpath=out/'runtime.json';cfgpath.write_text(json.dumps({'runtime':{'weight_dtype':'f16','act_dtype':'f32','flash_attn':True}}))
rows=[]
with vla_cpp.load('/home/skysi/VLA/models/'+name+'/gguf/model.gguf',config_path=str(cfgpath)) as m:
 cfg=m.config;rng=np.random.default_rng(9847)
 images=[rng.integers(0,256,(224,224,3),dtype=np.uint8) for _ in range(2)]
 state=np.zeros(cfg.max_state_dim,np.float32);noise=rng.normal(size=cfg.n_suffix*cfg.max_action_dim).astype(np.float32)
 kwargs=dict(tokens=[1,100,200,300,400,2],state=state,noise=noise)
 ref=m.predict(images,**kwargs)
 for iteration in range(3):
  emb=m.encode_images(images)
  assert emb.shape==(cfg.n_img*2,cfg.hidden) and np.isfinite(emb).all(),emb.shape
  a=m.predict(None,img_emb=emb,n_img_views=2,**kwargs)
  b=m.predict(images,**kwargs)
  assert np.array_equal(a,ref) and np.array_equal(b,ref)
  rows.append({'iteration':iteration,'embedding_shape':list(emb.shape),'max_abs':0})
  (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(short,'encode/predict/direct destination rebinding exact',flush=True)
