#!/usr/bin/env python3
"""Changing-observation/sequence-length regression for resident model reuse."""
from pathlib import Path
import json, os, sys
import numpy as np
repo=Path(__file__).resolve().parents[2]
record=Path(__file__).resolve().parent
os.environ['VLA_LIBRARY']=str(repo/'build-musa-official/libvla.so')
os.environ['VLA_DEVICE']='0';os.environ['VLA_N_THREADS']='4'
os.environ.pop('GGML_CUDA_DISABLE_GRAPHS',None)
sys.path.insert(0,str(repo/'bindings/python'))
import vla_cpp
runtime=record/'changing-input-runtime.json'
runtime.write_text(json.dumps({'runtime':{'weight_dtype':'f16','act_dtype':'f32','flash_attn':True,'mm_prec':'f32'}}))
rows=[]
for name in ['pi05_cube_merge_cksize50_step3_0929','cup_cube_joint_061_a800_50k_20261022']:
    with vla_cpp.load(str(Path('/home/skysi/VLA/models')/name/'gguf/model.gguf'),config_path=str(runtime)) as m:
        cfg=m.config
        y,x,c=np.indices((224,224,3))
        base=[np.ascontiguousarray(((x+2*y+40*c+17*v)&255).astype(np.uint8)) for v in range(2)]
        noise=(np.arange(cfg.n_suffix*cfg.max_action_dim,dtype=np.uint64)*2654435761%1000).astype(np.float32)*np.float32(.001)-np.float32(.5)
        seen={};errors=[]
        for case in [0,0,0,1,1,1,0,0,0,1,1,1]:
            images=base if case==0 else [np.ascontiguousarray(im[:,::-1,:]) for im in base]
            state=np.zeros(cfg.max_state_dim,dtype=np.float32)
            state[:cfg.real_state_dim]=np.arange(1,cfg.real_state_dim+1)*(.01 if case==0 else -.02)
            tokens=[1,100,200,300,400,2] if case==0 else [1,100,200,300,400,500,600,700,2]
            actions=m.predict(images,tokens=tokens,state=state,noise=noise if case==0 else np.ascontiguousarray(-noise))
            assert np.isfinite(actions).all()
            if case in seen: errors.append(float(np.max(np.abs(actions-seen[case]))))
            else: seen[case]=actions.copy()
        change=float(np.max(np.abs(seen[0]-seen[1])))
        row={'model':name,'sequence_lengths':[6,6,6,9,9,9,6,6,6,9,9,9],'repeat_max_abs':errors,'between_input_max_abs':change,
             'passed':max(errors)==0 and change>1e-5}
        assert row['passed'],row
        rows.append(row)
        (record/'summary-changing-inputs.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
