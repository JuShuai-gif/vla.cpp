from pathlib import Path
import os,sys,subprocess,json
d=Path(__file__).resolve().parent;repo=d.parents[1];stage=sys.argv[1]
msys='/opt/moore-threads/moore-perf-system/1.7.2/target-linux-aarch64/msys'
raw=Path('/home/skysi/VLA/thor-fusion-20261103')/stage;raw.mkdir(parents=True,exist_ok=False)
for short,name in [('pruned','pi05_cube_merge_cksize50_step3_0929'),('full','cup_cube_joint_061_a800_50k_20261022')]:
 out=d/('io-'+stage)/short;out.mkdir(parents=True)
 env=dict(os.environ,VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4',VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_MUSA_VISION_EPILOGUE='all',VLA_MUSA_GEMMA_ATTN='fused-denoise' if short=='pruned' else 'fused-all')
 for k in ['VLA_PI05_KV_DIRECT','VLA_MUSA_GEGLU_F16','VLA_PI05_RES_ADARMS']:env.pop(k,None)
 env.update(json.loads((d/'baseline-env.json').read_text()))
 if os.environ.get('PROFILE_SELECTED')=='1':env.update(json.loads((d/'selection.json').read_text())[short])
 if os.environ.get('PROFILE_MAPPED')=='1':env['VLA_PI05_HOST_INPUT']='mapped'
 cmd=[msys,'profile','-t','musa','--musa-graph-trace=node','-o',str(raw/short),sys.executable,str(d/'profile_entry.py'),'--model','/home/skysi/VLA/models/'+name+'/gguf/model.gguf','--out-dir',str(out),'--weight-dtype','f16','--act-dtype','f32','--flash-attn','--warmup','3','--runs','3']
 (out/'command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_'))}},indent=2))
 print(stage,short,flush=True)
 with (out/'profile.log').open('w') as f:
  subprocess.run(cmd,env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
  subprocess.run([msys,'stats','-r','musa_api_trace','-f','json','-o',str(out/'trace'),str(raw/(short+'.msys-rep'))],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)

 with (out/'export.log').open('w') as f:
  subprocess.run([msys,'export','-t','json','-o',str(raw/(short+'.json')),str(raw/(short+'.msys-rep'))],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
