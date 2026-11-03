from pathlib import Path
import os,sys,json
import numpy as np
repo=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
name,mode=sys.argv[1:3]
os.environ.update(VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
sys.path.insert(0,str(repo/'bindings/python'))
import vla_cpp
out=d/('lifecycle-'+mode)/name;out.mkdir(parents=True,exist_ok=False)
runtime=out/'runtime.json';runtime.write_text(json.dumps({'runtime':{'weight_dtype':'f16','act_dtype':'f32','flash_attn':True,'mm_prec':'f32'}}))
with vla_cpp.load('/home/skysi/VLA/models/'+name+'/gguf/model.gguf',config_path=str(runtime)) as m:
 cfg=m.config;y,x,c=np.indices((224,224,3))
 images=[np.ascontiguousarray(((x+2*y+40*c+17*v)&255).astype(np.uint8)) for v in range(2)]
 state=np.zeros(cfg.max_state_dim,np.float32);state[:cfg.real_state_dim]=np.arange(1,cfg.real_state_dim+1)*.01
 noise=np.ascontiguousarray((np.arange(cfg.n_suffix*cfg.max_action_dim,dtype=np.uint64)*2654435761%1000).astype(np.float32)*np.float32(.001)-np.float32(.5))
 tokens=[1,100,200,300,400,2]
 def predict(img=images,**kw):return m.predict(img,tokens=tokens,state=state,noise=noise,**kw)
 # Allocate one-view staging before the larger two-view graph, then exercise cache rebuilds.
 e1=m.encode_images(images[:1]);a1=predict(images[:1])
 a=predict();b=predict();assert np.array_equal(a,b)
 changed=[np.ascontiguousarray(((im.astype(np.uint16)+37)%256).astype(np.uint8)) for im in images]
 c1=predict(changed);c2=predict(changed)
 assert np.array_equal(c1,c2) and float(abs(a-c1).max())>1e-5
 assert np.array_equal(a,predict())
 emb=m.encode_images(images)
 via=predict(None,img_emb=emb,n_img_views=2)
 assert np.array_equal(a,via),float(abs(a-via).max())
 m.encode_images(images[:1]);again1=predict(images[:1]);assert np.array_equal(a1,again1)
 # Return from one view to two after output and input addresses have changed.
 for _ in range(3):assert np.array_equal(a,predict())
 refdir='changing-final' if name.startswith('pi05_') else 'changing-full-fused-all'
 ref=np.load(d.parent/'2026-11-02_前缀与动作Attention优化'/refdir/(name+'-0.npy'))
 err=float(abs(a-ref).max());peak=err/max(float(abs(ref).max()),1e-12)
 assert (peak<.001 if os.environ.get('CHECK_NUMERIC')=='1' else np.array_equal(a,ref)),peak
 np.save(out/'actions.npy',a)
 row={'model':name,'mode':mode,'one_two_one_two_views':True,'encode_then_predict':True,'precomputed_embeddings_exact':True,'changed_image_roundtrip_exact':True,'repeat_max_abs':0,'baseline_max_abs':err,'baseline_relative_peak':peak}
 (out/'summary.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
