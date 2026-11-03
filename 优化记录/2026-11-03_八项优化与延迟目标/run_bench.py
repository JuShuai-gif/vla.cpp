from pathlib import Path
import os,sys,json,subprocess
import numpy as np
d=Path(__file__).resolve().parent;r=d.parents[1];stage=sys.argv[1];out=d/stage;out.mkdir(exist_ok=False)
overrides=dict(x.split('=',1) for x in sys.argv[2:])
assert subprocess.check_output(['powerprofilesctl','get'],text=True).strip()=='performance'
sel=json.loads((d/os.environ.get('BENCH_CONFIG','baseline-env.json')).read_text());rows=[]
(out/'source.patch').write_bytes(subprocess.check_output(['git','diff','HEAD','--','src','scripts','tests','CMakeLists.txt'],cwd=r))
for short,name in [('pruned','pi05_cube_merge_cksize50_step3_0929'),('full','cup_cube_joint_061_a800_50k_20261022')]:
 env=dict(os.environ,VLA_DEVICE='0',VLA_N_THREADS='4',VLA_LIBRARY=str(r/'build-musa-official/libvla.so'))
 for k in ['VLA_PI05_LAST_KV_ONLY','VLA_PI05_K_ROPE_DIRECT','VLA_PI05_ROPE_TABLE','VLA_PI05_F16_PART','VLA_PI05_U8_PATCH','VLA_PI05_PREFIX_GAMMA_FOLD','VLA_PI05_VISION_NO_OUT_COPY','VLA_PI05_PREFETCH_KIB','VLA_MUSA_GEGLU_FAST','VLA_MUSA_GEMMA_FLAT_QK','VLA_MUSA_GEMMA_FLAT_AV','VLA_PI05_PRUNED_BATCH','VLA_MUSA_SMALL_GEMM']:env.pop(k,None)
 env.update(sel[short]);env.update(overrides)
 dest=out/short;dest.mkdir()
 model=Path('/home/skysi/VLA/models')/name/'gguf'
 cmd=[sys.executable,str(r/'scripts/bench_pi05.py'),'--model',str(model/'model.gguf'),'--out-dir',str(dest),'--weight-dtype','f16','--act-dtype','f32','--flash-attn','--warmup','3','--runs',env.get('BENCH_RUNS','5'),'--reference',str(model/'validation/musa-f32-actions.npy')]
 if env.get('BENCH_SPLIT')=='1':cmd.append('--split-phases')
 (dest/'command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_','BENCH_'))}},indent=2)+'\n')
 print(stage,short,flush=True)
 with (dest/'run.log').open('w') as f:subprocess.run(cmd,env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
 report=json.loads((dest/'report.json').read_text())
 assert report['repeat_max_abs']==0 and report['error_vs_reference']['relative_peak']<.005,report
 a=np.load(dest/'actions.npy');ref=d/'baseline'/short/'actions.npy';err=0
 if ref.exists():
  b=np.load(ref);err=float(abs(a-b).max()/max(float(abs(b).max()),1e-12));assert err<.001,(short,err)
 row={'model':short,'wall_ms':float(np.mean([x['wall_ms'] for x in report['runs']])),'baseline_relative_peak':err,'reference_error':report['error_vs_reference']}
 phases=[json.loads(x.split(' ',1)[1]) for x in (dest/'run.log').read_text().splitlines() if x.startswith('VLA_PI05_PHASES ')]
 if phases:row['phases']={k:float(np.mean([x[k] for x in phases[3:]])) for k in ['vision_ms','prefix_ms','denoise_ms']}
 rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
