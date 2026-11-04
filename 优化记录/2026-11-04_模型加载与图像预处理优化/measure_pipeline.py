from pathlib import Path
import os,sys,time,json,hashlib,cProfile,pstats,io,resource
d=Path(__file__).resolve().parent;repo=d.parents[1];short,stage=sys.argv[1:3]
out=d/stage/short;out.mkdir(parents=True,exist_ok=False)
times={}
def measure(key,fn):
 t=time.perf_counter();v=fn();times[key]=(time.perf_counter()-t)*1000;return v
t=time.perf_counter()
import numpy as np,gguf
times['import_numpy_gguf_ms']=(time.perf_counter()-t)*1000
sys.path.insert(0,str(repo/'scripts'));sys.path.insert(0,str(repo/'bindings/python'))
if stage.startswith('baseline'):
 import importlib.util
 spec=importlib.util.spec_from_file_location('before',d/'run_pi05_before.py')
 run_pi05=importlib.util.module_from_spec(spec);spec.loader.exec_module(run_pi05)
 run_pi05.REPO=repo
else:import run_pi05
name='pi05_cube_merge_cksize50_step3_0929'if short=='pruned'else 'cup_cube_joint_061_a800_50k_20261022'
base=Path('/home/skysi/VLA/models')/name/'gguf';path=base/'model.gguf'
prof=cProfile.Profile()
if os.environ.get('PROFILE_METADATA')=='1':prof.enable()
reader=measure('python_gguf_metadata_ms',lambda:(gguf.GGUFReader if stage.startswith('baseline')else run_pi05.GGUFReader)(str(path)))
if os.environ.get('PROFILE_METADATA')=='1':
 prof.disable();s=io.StringIO();pstats.Stats(prof,stream=s).sort_stats('cumtime').print_stats(25);(out/'metadata-profile.txt').write_text(s.getvalue())
state=np.asarray(json.loads((base/'model.stats.json').read_text())['observation.state']['mean'],np.float32)
prompt,tokens=measure('prepare_prompt_ms',lambda:run_pi05.prepare_prompt(reader,'pick up the cube',state))
cameras=reader.fields['pi05.camera_keys'].contents()
impaths=[base/'validation'/(k.split('.')[-1]+'-480x640.png')for k in cameras]
images=measure('first_image_pair_ms',lambda:run_pi05.prepare_images(impaths,224)if hasattr(run_pi05,'prepare_images')else [run_pi05.prepare_image(p,224)for p in impaths])
warm=[]
for _ in range(30):
 t=time.perf_counter();a=run_pi05.prepare_images(impaths,224)if hasattr(run_pi05,'prepare_images')else [run_pi05.prepare_image(p,224)for p in impaths];warm.append((time.perf_counter()-t)*1000)
 assert all(np.array_equal(x,y)for x,y in zip(a,images))
times['warm_image_pair_mean_ms']=float(np.mean(warm));times['warm_image_pair_p95_ms']=float(np.percentile(warm,95))
if not stage.startswith('baseline'):
 base_report=json.loads((d/'baseline-clean'/short/'report.json').read_text())
 assert tokens==base_report['tokens']and cameras==base_report['camera_order']
 for i,a in enumerate(images):assert np.array_equal(a,np.load(d/'baseline-clean'/short/(str(i)+'.npy')))
for i,a in enumerate(images):np.save(out/(str(i)+'.npy'),a)
dtype_counts={}
for tensor in reader.tensors:dtype_counts[str(tensor.tensor_type)]=dtype_counts.get(str(tensor.tensor_type),0)+1
del reader
env=json.loads((d/'baseline-profiles.json').read_text())[short]
os.environ.update(env,VLA_LIBRARY=os.environ.get('MEASURE_LIBRARY',str(repo/'build-musa-official/libvla.so')),VLA_DEVICE='0',VLA_N_THREADS='4')
measure('binding_import_ms',lambda:__import__('vla_cpp'));import vla_cpp
cfgpath=out/'runtime.json';cfgpath.write_text(json.dumps({'runtime':{'weight_dtype':'f16','act_dtype':'f32','flash_attn':True}}))
m=measure('cpp_model_load_ms',lambda:vla_cpp.load(str(path),config_path=str(cfgpath)))
cfg=m.config;padded=np.zeros(cfg.max_state_dim,np.float32);padded[:len(state)]=state
noise=np.random.default_rng(0).standard_normal(cfg.n_suffix*cfg.max_action_dim).astype(np.float32)
expected=np.asarray(json.loads((d.parent/'2026-11-03_四项深化与MCU热点优化/final-cli-f32'/(short+'.json')).read_text())['actions'],np.float32)
pred=[]
with m:
 for i in range(6):
  t=time.perf_counter();a=m.predict(images,tokens=tokens,state=padded,noise=noise,pixel_format=vla_cpp.PIXEL_F32_RGB_01);pred.append((time.perf_counter()-t)*1000)
  assert np.array_equal(a[:,:cfg.real_action_dim],expected)
 np.save(out/'actions.npy',a)
report={'model':short,'stage':stage,'timings':times,'preprocess_wall_ms':warm,'predict_wall_ms':pred,
        'dtype_counts':dtype_counts,'tokens':tokens,'camera_order':cameras,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'gpu_reference_max_abs':0,'metadata_profiled':os.environ.get('PROFILE_METADATA')=='1','loaded_core':[x for x in Path('/proc/self/maps').read_text().splitlines()if 'libvla_core'in x],
        'note':'Disk cache not dropped; metadata profiled only when PROFILE_METADATA=1.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
