#!/usr/bin/env python3
"""Validate both pi05 variants: tensor-derived vision reference and CPU/MUSA inference.

Run --mode vision-ref, then --mode run --device cpu, --mode run --device 0,
and --mode compare. Uses synthetic inputs; this is not a robot task evaluation.
"""
import argparse
import os,sys,json,time
import struct
from convert_pi05_local import checkpoint_dir
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model',type=Path,required=True)
ap.add_argument('--ckpt',type=Path,required=True)
ap.add_argument('--out-dir',type=Path)
ap.add_argument('--mode',choices=('vision-ref','run','compare'),required=True)
ap.add_argument('--device',default='0')
ap.add_argument('--weight-dtype',default='f32',choices=('f32','bf16','f16'))
opt=ap.parse_args()
base=opt.out_dir or opt.model.parent/'validation'
base.mkdir(parents=True,exist_ok=True)
ckpt=checkpoint_dir(opt.ckpt)
source_cfg=json.loads((ckpt/'config.json').read_text())
# Read only the safetensors header; tokenizer blobs are irrelevant to graph validation.
with (ckpt/'model.safetensors').open('rb') as f:
 header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
prefix='model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.'
conv=header[prefix+'embeddings.patch_embedding.weight']['shape']
patch_tokens,hidden=header[prefix+'embeddings.position_embedding.weight']['shape']
layers=len({int(k.split('.encoder.layers.')[1].split('.')[0]) for k in header if k.startswith(prefix+'encoder.layers.')})
heads=source_cfg.get('vision_config',{}).get('num_attention_heads',16)
patch=conv[2]; size=int(round(patch_tokens**0.5))*patch
keep=source_cfg.get('vision_token_prune',0) or patch_tokens
lm_hidden=2048; head_dim=hidden//heads
views=sum(v['type']=='VISUAL' for v in source_cfg['input_features'].values())
repo=Path(__file__).resolve().parents[1]
os.environ['VLA_DEVICE']=opt.device
if opt.mode=='compare':
 def metrics(a,b):
  d=np.abs(a-b)
  return {'max_abs':float(d.max()),'rmse':float(np.sqrt(np.mean(d*d))),'relative_peak':float(d.max()/max(float(np.abs(b).max()),1e-12))}
 ref=np.load(base/'vision-reference.npz')['emb']
 cpu=np.load(base/'cpu-f32-vision.npy'); musa=np.load(base/'musa-f32-vision.npy')
 report={'synthetic_inputs':True,'vision_cpu_vs_torch':metrics(cpu,ref),'vision_musa_vs_torch':metrics(musa,ref),'vision_musa_vs_cpu':metrics(musa,cpu)}
 real=source_cfg['output_features']['action']['shape'][0]
 a=np.load(base/'musa-f32-actions.npy')[:,:real]; b=np.load(base/'cpu-f32-actions.npy')[:,:real]
 report['actions_musa_vs_cpu']=metrics(a,b)
 report['threshold_relative_peak']=0.005
 report['passed']=all(v['relative_peak']<0.005 for v in report.values() if isinstance(v,dict))
 (base/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
 raise SystemExit(0 if report['passed'] else 1)
os.environ.setdefault("VLA_LIBRARY",str(repo/"build-musa-m1000/libvla.so"))
os.environ.setdefault("VLA_N_THREADS","4")
sys.path.insert(0,str(repo/"bindings/python"))
y,x,c=np.indices((size,size,3))
images=[np.ascontiguousarray(((x+2*y+40*c+17*v)&255).astype(np.uint8)) for v in range(views)]
for v,im in enumerate(images):
 (base/f"synthetic_{v}.ppm").write_bytes(f"P6\n{size} {size}\n255\n".encode()+im.tobytes())
if opt.mode=="vision-ref":
 import torch
 import torch.nn.functional as F
 from safetensors import safe_open
 torch.set_num_threads(4)
 root="model.paligemma_with_expert.paligemma.model."
 with safe_open(ckpt/"model.safetensors",framework="pt") as sf:
  w={k[len(root):]:sf.get_tensor(k).float() for k in sf.keys() if k.startswith(root+"vision_tower.") or k.startswith(root+"multi_modal_projector.")}
 p="vision_tower.vision_model."
 def get(s): return w[p+s]
 def lin(z,pfx): return F.linear(z,get(pfx+".weight"),get(pfx+".bias"))
 z=torch.tensor(np.stack(images).transpose(0,3,1,2)).float()/255.0*2-1
 with torch.inference_mode():
  z=F.conv2d(z,get("embeddings.patch_embedding.weight"),get("embeddings.patch_embedding.bias"),stride=patch).flatten(2).transpose(1,2)
  z=z+get("embeddings.position_embedding.weight")
  for i in range(layers):
   pre=f"encoder.layers.{i}."
   n=F.layer_norm(z,(hidden,),get(pre+"layer_norm1.weight"),get(pre+"layer_norm1.bias"),1e-6)
   q,k,v=[lin(n,pre+"self_attn."+t+"_proj").reshape(views,patch_tokens,heads,head_dim).transpose(1,2) for t in ("q","k","v")]
   aw=(q@k.transpose(-1,-2)/(head_dim**0.5)).softmax(-1)
   z=z+lin((aw@v).transpose(1,2).reshape(views,patch_tokens,hidden),pre+"self_attn.out_proj")
   n=F.layer_norm(z,(hidden,),get(pre+"layer_norm2.weight"),get(pre+"layer_norm2.bias"),1e-6)
   z=z+lin(F.gelu(lin(n,pre+"mlp.fc1"),approximate="tanh"),pre+"mlp.fc2")
  z=F.layer_norm(z,(hidden,),get("post_layernorm.weight"),get("post_layernorm.bias"),1e-6)
  scores=aw.mean(1).mean(1)
  ids=scores.topk(keep,dim=-1,sorted=False).indices.sort(-1).values
  selected=z.gather(1,ids[:,:,None].expand(-1,-1,hidden))
  emb=F.linear(selected,w["multi_modal_projector.linear.weight"],w["multi_modal_projector.linear.bias"]).numpy().reshape(views*keep,lm_hidden)
  np.savez(base/"vision-reference.npz",emb=emb,scores=scores.numpy(),ids=ids.numpy())
 print("PyTorch original safetensors vision reference:",emb.shape,"finite",bool(np.isfinite(emb).all()),flush=True)
else:
 import vla_cpp
 label=("cpu" if opt.device=="cpu" else "musa")+"-"+opt.weight_dtype
 start=time.monotonic()
 dtype=opt.weight_dtype
 runtime=base/(label+"-runtime.json")
 runtime.write_text(json.dumps({"runtime":{"weight_dtype":dtype,"act_dtype":"f32","flash_attn":False,"mm_prec":"f32"}}))
 with vla_cpp.load(str(opt.model),config_path=str(runtime)) as m:
  cfg=m.config
  assert (cfg.n_img,cfg.n_suffix,cfg.num_steps,cfg.real_state_dim,cfg.real_action_dim)==(keep,source_cfg["chunk_size"],source_cfg["num_inference_steps"],source_cfg["input_features"]["observation.state"]["shape"][0],source_cfg["output_features"]["action"]["shape"][0])
  print("CONFIG OK; load seconds",time.monotonic()-start,flush=True)
  emb=m.encode_images(images)
  assert emb.shape==(views*keep,lm_hidden) and np.isfinite(emb).all()
  np.save(base/(label+"-vision.npy"),emb)
  print("VISION OK",m.last_stats().ms_total,flush=True)
  state=np.zeros(cfg.max_state_dim,dtype=np.float32);state[:cfg.real_state_dim]=np.arange(1,cfg.real_state_dim+1)*0.01
  noise=(np.arange(cfg.n_suffix*cfg.max_action_dim,dtype=np.uint64)*2654435761%1000).astype(np.float32)*np.float32(.001)-np.float32(.5)
  args=dict(tokens=[1,100,200,300,400,2],state=state,noise=noise,timing=vla_cpp.TIMING_PHASE)
  actions=m.predict(images,**args)
  st=m.last_stats()
  timing={k:float(getattr(st,k)) for k in ("ms_total","ms_vision","ms_inference","ms_prefill","ms_denoise")}
  assert actions.shape==(cfg.n_suffix,cfg.max_action_dim) and np.isfinite(actions).all()
  np.save(base/(label+"-actions.npy"),actions)
  print("PREDICT OK",timing,flush=True)
  # Cached graph + encode/precomputed path must agree with the direct image path.
  repeated=m.predict(None,img_emb=emb,n_img_views=views,**args)
  repeat_max=float(np.max(np.abs(actions-repeated)))
  assert np.allclose(actions,repeated,atol=1e-5,rtol=1e-5),repeat_max
  report={"backend":os.environ.get("VLA_DEVICE"),"weight_dtype":dtype,"shape":list(actions.shape),"real_action_dim":cfg.real_action_dim,"timing_ms":timing,"direct_vs_precomputed_max_abs":repeat_max,"elapsed_s":time.monotonic()-start}
  (base/(label+"-report.json")).write_text(json.dumps(report,indent=2)+"\n")
  print("REPEAT/PRECOMPUTED OK",repeat_max,flush=True)
