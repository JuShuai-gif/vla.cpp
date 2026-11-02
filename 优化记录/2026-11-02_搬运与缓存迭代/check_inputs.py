#!/usr/bin/env python3
from pathlib import Path
import os,sys,json
import numpy as np
repo=Path(__file__).resolve().parents[2]; record=Path(__file__).resolve().parent
mode=sys.argv[1]; out=record/('changing-'+mode);out.mkdir(exist_ok=False)
os.environ.setdefault('VLA_LIBRARY',str(repo/'build-musa-official/libvla.so'))
os.environ['VLA_DEVICE']='0';os.environ['VLA_N_THREADS']='4'
sys.path.insert(0,str(repo/'bindings/python'))
import vla_cpp
runtime=out/'runtime.json';runtime.write_text(json.dumps({'runtime':{'weight_dtype':'f16','act_dtype':'f32','flash_attn':True,'mm_prec':'f32'}}))
rows=[]
for name in ['pi05_cube_merge_cksize50_step3_0929','cup_cube_joint_061_a800_50k_20261022']:
    with vla_cpp.load(str(Path('/home/skysi/VLA/models')/name/'gguf/model.gguf'),config_path=str(runtime)) as m:
        cfg=m.config;y,x,c=np.indices((224,224,3))
        noise=(np.arange(cfg.n_suffix*cfg.max_action_dim,dtype=np.uint64)*2654435761%1000).astype(np.float32)*np.float32(.001)-np.float32(.5)
        seen={};drift=0.;sequence=[0,0,0,1,1,1,2,2,2,0,0,0]
        for case in sequence:
            images=[np.ascontiguousarray(((x*(1+case)+2*y+40*c+17*v+23*case)&255).astype(np.uint8)) for v in range(2)]
            state=np.zeros(cfg.max_state_dim,dtype=np.float32)
            state[:cfg.real_state_dim]=np.arange(1,cfg.real_state_dim+1)*(.01 if case==0 else -.02)
            tokens=[[1,100,200,300,400,2],[1,500,600,700,800,2],[1,100,200,300,400,500,600,700,2]][case]
            actions=m.predict(images,tokens=tokens,state=state,noise=np.ascontiguousarray(noise if case==0 else -noise))
            assert np.isfinite(actions).all()
            if case in seen:drift=max(drift,float(abs(actions-seen[case]).max()))
            else:seen[case]=actions.copy()
        assert drift==0
        errs=[]
        for case,a in seen.items():
            np.save(out/(name+'-'+str(case)+'.npy'),a)
            if mode!='baseline':
                b=np.load(record/'changing-baseline'/(name+'-'+str(case)+'.npy'))
                errs.append({'case':case,'max_abs':float(abs(a-b).max()),'relative_peak':float(abs(a-b).max()/max(abs(b).max(),1e-12))})
                assert errs[-1]['relative_peak']<.005,errs
        assert all(float(abs(seen[0]-seen[k]).max())>1e-5 for k in [1,2])
        row={'model':name,'cases':sequence,'repeat_max_abs':drift,'baseline_errors':errs}
        rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
