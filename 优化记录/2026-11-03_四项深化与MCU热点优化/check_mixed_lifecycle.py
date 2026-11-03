from pathlib import Path
import os,sys,json
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1];short=sys.argv[1]
env=json.loads((d/'selection.json').read_text())[short]
os.environ.update(env,VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
sys.path.insert(0,str(repo/'bindings/python'));sys.path.insert(0,str(repo/'scripts'))
import gguf,vla_cpp
from run_pi05 import prepare_prompt,prepare_image
name='pi05_cube_merge_cksize50_step3_0929'if short=='pruned'else 'cup_cube_joint_061_a800_50k_20261022'
base=Path('/home/skysi/VLA/models')/name/'gguf'
reader=gguf.GGUFReader(str(base/'model.gguf'))
state=np.asarray(json.loads((base/'model.stats.json').read_text())['observation.state']['mean'],np.float32)
prompt,tokens=prepare_prompt(reader,'pick up the cube',state)
cameras=reader.fields['pi05.camera_keys'].contents()
images=[prepare_image(base/'validation'/(c.split('.')[-1]+'-480x640.png'),224)for c in cameras]
reference=np.asarray(json.loads((d.parent/'2026-11-03_八项优化与延迟目标'/'candidate-full-entry'/name/'f32.json').read_text())['actions'],np.float32)
out=d/os.environ.get('LIFECYCLE_TAG','mixed-lifecycle')/short;out.mkdir(parents=True,exist_ok=False)
runtime=out/'runtime.json';runtime.write_text(json.dumps({'runtime':{'weight_dtype':'f16','act_dtype':'f32','flash_attn':True}}))
with vla_cpp.load(str(base/'model.gguf'),config_path=str(runtime))as m:
 cfg=m.config;padded=np.zeros(cfg.max_state_dim,np.float32);padded[:len(state)]=state
 noise=np.random.default_rng(0).standard_normal(cfg.n_suffix*cfg.max_action_dim).astype(np.float32)
 def predict(img=images,fmt=vla_cpp.PIXEL_F32_RGB_01,**kwargs):
  return m.predict(img,tokens=tokens,state=padded,noise=noise,pixel_format=fmt,**kwargs)
 a=predict()
 assert np.array_equal(a[:,:cfg.real_action_dim],reference)
 for _ in range(2):assert np.array_equal(a,predict())
 for _ in range(8):assert np.array_equal(a,predict(timing=vla_cpp.TIMING_PHASE))
 u8=[np.ascontiguousarray(np.rint(x*255).astype(np.uint8))for x in images]
 b=predict(u8,fmt=vla_cpp.PIXEL_U8)
 assert np.array_equal(a,predict())and np.array_equal(b,predict(u8,fmt=vla_cpp.PIXEL_U8))
 one=predict(images[:1]);assert np.array_equal(a,predict())
 emb=m.encode_images(images,pixel_format=vla_cpp.PIXEL_F32_RGB_01)
 assert np.array_equal(a,predict(None,img_emb=emb,n_img_views=2))
 e1=m.encode_images(images[:1],pixel_format=vla_cpp.PIXEL_F32_RGB_01)
 assert np.array_equal(one,predict(None,img_emb=e1,n_img_views=1))
 assert np.array_equal(b,predict(u8,fmt=vla_cpp.PIXEL_U8))
 changed=[np.ascontiguousarray(x*.875+.02,dtype=np.float32)for x in images]
 c=predict(changed);assert np.isfinite(c).all()and float(abs(c-a).max())>1e-5
 assert np.array_equal(c,predict(changed))and np.array_equal(a,predict())
 for _ in range(3):assert np.array_equal(a,predict())
 np.save(out/'actions.npy',a)
 row={'model':short,'reference_max_abs':0,'f32_u8_f32_roundtrip_exact':True,
      'one_two_one_two_views':True,'precomputed_embedding_exact':True,
      'changed_image_roundtrip_exact':True,'phase_replay_runs':8,'tokens':len(tokens)}
 (out/'summary.json').write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True)
