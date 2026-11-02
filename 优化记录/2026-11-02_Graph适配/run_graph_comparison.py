#!/usr/bin/env python3
from pathlib import Path
import argparse, json, os, subprocess
repo=Path(__file__).resolve().parents[2];record=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('stages',nargs='+',choices=['baseline','control','graph'])
ap.add_argument('--act-dtype',choices=['f32','f16'],default='f32')
ap.add_argument('--label',default='')
args=ap.parse_args()
assert subprocess.check_output(['powerprofilesctl','get'],text=True).strip()=='performance'
rows=[]
for stage in args.stages:
    build=Path('/home/skysi/VLA/graph-baseline-47a75fb') if stage=='baseline' else repo/'build-musa-official'
    for name in ['pi05_cube_merge_cksize50_step3_0929','cup_cube_joint_061_a800_50k_20261022']:
        out=record/(args.label+stage+'-'+args.act_dtype)/name
        if (out/'run.log').exists():raise SystemExit('Use a new label; refusing to overwrite '+str(out))
        out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,VLA_LIBRARY=str(build/'libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4',
                 LD_LIBRARY_PATH=str(build/'bin')+':'+str(build)+':/usr/local/musa/lib')
        for k in ['GGML_CUDA_DISABLE_GRAPHS','GGML_MUSA_DISABLE_GRAPHS','VLA_PI05_NO_FUSE','VLA_PI05_NO_PROJ_FUSE']:
            env.pop(k,None)
        if stage!='graph':env['GGML_CUDA_DISABLE_GRAPHS']='1'
        model=Path('/home/skysi/VLA/models')/name
        cmd=[str(repo/'.venv-converter/bin/python'),'scripts/bench_pi05.py','--model',str(model/'gguf/model.gguf'),
             '--out-dir',str(out),'--weight-dtype','f16','--act-dtype',args.act_dtype,'--flash-attn','--warmup','3','--runs','5',
             '--reference',str(model/'gguf/validation/musa-f32-actions.npy')]
        (out/'command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_','LD_LIBRARY_PATH'))}},indent=2)+'\n')
        print('START',stage,args.act_dtype,name,flush=True)
        with (out/'run.log').open('w') as log:
            p=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        row={'stage':stage,'act_dtype':args.act_dtype,'model':name,'exit_code':p.returncode}
        if (out/'report.json').exists():
            report=json.loads((out/'report.json').read_text())
            row.update(median_ms=report['median_ms'],error=report['error_vs_reference'],repeat_max_abs=report['repeat_max_abs'])
            row['passed']=row['error']['relative_peak']<.005 and row['repeat_max_abs']==0
        rows.append(row)
        (record/('summary-'+args.label+args.act_dtype+'-'+'-'.join(args.stages)+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
