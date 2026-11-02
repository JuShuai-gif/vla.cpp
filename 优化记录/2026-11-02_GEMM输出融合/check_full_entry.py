#!/usr/bin/env python3
"""Regression of both checkpoint layouts through the complete inference entry."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[2]
record=Path(__file__).resolve().parent
models=['pi05_cube_merge_cksize50_step3_0929','cup_cube_joint_061_a800_50k_20261022']
env=dict(os.environ,VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
# Exercise the final graph compatibility fallback without an environment override.
env.pop('GGML_CUDA_DISABLE_GRAPHS',None)
results=[]
for name in models:
    env['VLA_MUSA_GEMMA_ATTN']='fused-denoise' if name.startswith('pi05_') else 'fused-all'
    model=Path('/home/skysi/VLA/models')/name
    refdir=model/'gguf/validation'
    ref=json.loads((refdir/'inference-cpu.json').read_text())
    state=json.loads((model/'gguf/model.stats.json').read_text())['observation.state']['mean']
    for dtype in ['f32','f16']:
        out=record/'full-entry'/name
        out.mkdir(parents=True,exist_ok=True)
        dest=out/(dtype+'.json')
        cmd=[sys.executable,str(repo/'scripts/run_pi05.py'),'--model',str(model/'gguf/model.gguf'),
             '--image',f'top={refdir}/top-480x640.png','--image',f'wrist={refdir}/wrist-480x640.png',
             '--text','pick up the cube','--state',','.join(map(str,state)),
             '--weight-dtype','f16','--act-dtype',dtype,'--flash-attn','--warmup','2','--seed','0','--output',str(dest)]
        (out/(dtype+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_'))}},indent=2)+'\n')
        with (out/(dtype+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        got=json.loads(dest.read_text())
        assert got['warmup']==2 and got['repeat_max_abs']==0
        assert got['token_ids']==ref['token_ids'] and got['camera_order']==ref['camera_order']
        a,b=np.array(got['actions']),np.array(ref['actions'])
        assert a.shape==(50,7) and np.isfinite(a).all()
        delta=a-b
        row={'model':name,'act_dtype':dtype,'camera_order':got['camera_order'],
             'text_tokens':len(got['token_ids']),'warmup':got['warmup'],'repeat_max_abs':got['repeat_max_abs'],'max_abs':float(abs(delta).max()),
             'rmse':float(np.sqrt(np.mean(delta**2))),
             'relative_peak':float(abs(delta).max()/abs(b).max())}
        row['passed']=row['relative_peak']<.005
        results.append(row)
        (record/'summary-full-entry.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        assert row['passed'],row
