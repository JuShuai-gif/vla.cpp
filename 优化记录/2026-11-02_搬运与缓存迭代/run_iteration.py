#!/usr/bin/env python3
"""Serialized two-model measurements. Each stage preserves source diff and outputs."""
from pathlib import Path
import json,os,subprocess,sys
import numpy as np
repo=Path(__file__).resolve().parents[2]; record=Path(__file__).resolve().parent
stage=sys.argv[1]; runs=int(os.environ.get('BENCH_RUNS','7'))
root=record/stage
root.mkdir(exist_ok=False)
env=dict(os.environ,VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
if stage.startswith('baseline'):
    env['VLA_LIBRARY']='/home/skysi/VLA/baseline-dfc0da1/libvla.so'
    env['LD_LIBRARY_PATH']='/home/skysi/VLA/baseline-dfc0da1'
for k in ['GGML_CUDA_DISABLE_GRAPHS','GGML_MUSA_DISABLE_GRAPHS','VLA_PI05_SPLIT_PHASES','VLA_PI05_HOST_SPLIT']:
    env.pop(k,None)
(root/'source.patch').write_bytes(subprocess.check_output(['git','diff','HEAD','--','src','tests','scripts','CMakeLists.txt'],cwd=repo))
assert subprocess.check_output(['powerprofilesctl','get'],text=True).strip()=='performance'
summary=[]
for name in ['pi05_cube_merge_cksize50_step3_0929','cup_cube_joint_061_a800_50k_20261022']:
    model=Path('/home/skysi/VLA/models')/name;out=root/name;out.mkdir()
    cmd=[sys.executable,str(repo/'scripts/bench_pi05.py'),'--model',str(model/'gguf/model.gguf'),
         '--out-dir',str(out),'--weight-dtype','f16','--act-dtype',os.environ.get('BENCH_ACT','f32'),'--flash-attn',
         '--warmup','3','--runs',str(runs),'--reference',str(model/'gguf/validation/musa-f32-actions.npy')]
    if os.environ.get('BENCH_SPLIT')=='1':cmd+=['--split-phases']
    (out/'command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_','LD_LIBRARY'))}},indent=2)+'\n')
    print('START',stage,name,flush=True)
    with (out/'run.log').open('w') as log:
        subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    report=json.loads((out/'report.json').read_text())
    assert report['repeat_max_abs']==0 and report['error_vs_reference']['relative_peak']<.005,report
    row={'model':name,'median_ms':report['median_ms'],'reference_error':report['error_vs_reference']}
    baseline=record/'baseline'/name/'actions.npy'
    if not stage.startswith('baseline'):
        a=np.load(out/'actions.npy');b=np.load(baseline)
        row['baseline_max_abs']=float(np.max(np.abs(a-b)))
        row['baseline_relative_peak']=row['baseline_max_abs']/max(float(np.max(np.abs(b))),1e-12)
        assert row['baseline_relative_peak']<.005,row
    phases=[json.loads(x.split(' ',1)[1]) for x in (out/'run.log').read_text().splitlines() if x.startswith('VLA_PI05_PHASES ')]
    if phases:
        row['phase_median_ms']={k:float(np.median([x[k] for x in phases[3:]])) for k in ['vision_ms','prefix_ms','denoise_ms']}
    summary.append(row);(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(row),flush=True)
