from pathlib import Path
import os,sys,json,time,subprocess,hashlib
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1];short=sys.argv[1]
assert subprocess.check_output(['/usr/bin/python3','/usr/bin/powerprofilesctl','get'],text=True).strip()=='performance'
env=json.loads((d/os.environ.get('BENCH_CONFIG','baseline-env.json')).read_text())[short]
os.environ.update(env,VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
sys.path.insert(0,str(repo/'scripts'));sys.path.insert(0,str(repo/'bindings/python'))
from run_pi05 import prepare_prompt,prepare_image
import gguf,vla_cpp
name='pi05_cube_merge_cksize50_step3_0929' if short=='pruned' else 'cup_cube_joint_061_a800_50k_20261022'
model=Path('/home/skysi/VLA/models')/name;reference=d.parent/'2026-11-03_八项优化与延迟目标/candidate-full-entry'/name/'f32.json'
expected=json.loads(reference.read_text());refdir=model/'gguf/validation'
state=np.asarray(json.loads((model/'gguf/model.stats.json').read_text())['observation.state']['mean'],np.float32)
reader=gguf.GGUFReader(str(model/'gguf/model.gguf'));prompt,tokens=prepare_prompt(reader,'pick up the cube',state)
cameras=reader.fields['pi05.camera_keys'].contents();size=int(reader.fields['pi05.image_size'].contents())
images=[prepare_image(refdir/(key.split('.')[-1]+'-480x640.png'),size)for key in cameras];del reader
assert tokens==expected['token_ids'] and cameras==expected['camera_order']
out=Path(sys.argv[2])
if os.environ.get('PROFILE_MARKERS'):out=out/('pid-'+str(os.getpid()))
out.mkdir(parents=True,exist_ok=False)
cfgpath=out/'runtime.json';cfgpath.write_text(json.dumps({'runtime':{'weight_dtype':'f16','act_dtype':'f32','flash_attn':True}}))
times=[];first=None;errors=[]
warmup=int(os.environ.get('BENCH_WARMUP','3'));runs=int(os.environ.get('BENCH_RUNS','8'))
marker=None
if os.environ.get('PROFILE_MARKERS'):
 import ctypes
 marker=ctypes.CDLL(os.environ['PROFILE_MARKERS'])
 marker.r2_begin.restype=ctypes.c_int;marker.r2_end.restype=ctypes.c_int

with vla_cpp.load(str(model/'gguf/model.gguf'),config_path=str(cfgpath))as m:
 cfg=m.config;padded=np.zeros(cfg.max_state_dim,np.float32);padded[:len(state)]=state
 noise=np.random.default_rng(0).standard_normal(cfg.n_suffix*cfg.max_action_dim).astype(np.float32)
 for i in range(warmup+runs):
  if marker and i==warmup:assert marker.r2_begin()==0
  begin=time.perf_counter()
  a=m.predict(images,tokens=tokens,state=padded,noise=noise,pixel_format=vla_cpp.PIXEL_F32_RGB_01)
  elapsed=(time.perf_counter()-begin)*1000
  if marker and i==warmup:assert marker.r2_end()==0
  if first is None:first=a.copy()
  assert np.isfinite(a).all() and np.array_equal(a,first)
  ref=np.asarray(expected['actions'],np.float32);got=a[:,:cfg.real_action_dim]
  err=float(abs(got-ref).max()/max(float(abs(ref).max()),1e-12))
  assert err<.001,(short,err)
  cpu=np.asarray(json.loads((refdir/'inference-cpu.json').read_text())['actions'],np.float32)
  assert float(abs(got-cpu).max()/max(float(abs(cpu).max()),1e-12))<.005
  errors.append(err)
  if i>=warmup:times.append(elapsed)
 np.save(out/'actions.npy',first)
report={'model':short,'model_path':str(model/'gguf/model.gguf'),'warmup':warmup,'samples':runs,
 'text_tokens':len(tokens),'token_ids':tokens,'camera_order':cameras,'num_steps':cfg.num_steps,
 'pixel_format':'F32_RGB_01 (saved LeRobot float resize)','wall_ms':times,'mean_ms':float(np.mean(times)),
 'p95_ms':float(np.percentile(times,95)),'max_ms':float(np.max(times)),'min_ms':float(np.min(times)),
 'repeat_max_abs':0,'reference_relative_peak':max(errors),'reference':str(reference),
 'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'env':env,
 'scope':'resident predict Python binding wall; excludes checkpoint load, image file decode/resize and tokenizer'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ['wall_ms','env','token_ids']},indent=2),flush=True)

