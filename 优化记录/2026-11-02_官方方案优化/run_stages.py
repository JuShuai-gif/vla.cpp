#!/usr/bin/env python3
"""Serial A/B benchmarks for this experiment; run after compilation has finished."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

repo = Path(__file__).resolve().parents[2]
record = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('stages', nargs='+', choices=['old','control','graph','fa','graph-fa'])
ap.add_argument('--label', default='')
ap.add_argument('--act-dtype', choices=['f32','f16'], default='f32')
args = ap.parse_args()
if subprocess.check_output(['powerprofilesctl','get'],text=True).strip() != 'performance':
    raise SystemExit('Set performance profile before comparing stages.')
models = ['pi05_cube_merge_cksize50_step3_0929', 'cup_cube_joint_061_a800_50k_20261022']
results = []
for stage in args.stages:
    for model in models:
        build = repo / ('build-musa-m1000' if stage == 'old' else 'build-musa-official')
        out = record / (args.label+stage+'-'+args.act_dtype) / model
        out.mkdir(parents=True,exist_ok=True)
        if (out/'report.json').exists():
            raise SystemExit(f'Refusing to overwrite {out}; use a new experiment directory for repeated trials.')
        env = dict(os.environ, VLA_LIBRARY=str(build/'libvla.so'), VLA_DEVICE='0', VLA_N_THREADS='4')
        for key in ['GGML_CUDA_DISABLE_GRAPHS','GGML_MUSA_DISABLE_GRAPHS','VLA_PI05_NO_FUSE','VLA_PI05_NO_PROJ_FUSE']:
            env.pop(key,None)
        if stage not in ['graph','graph-fa']:
            env['GGML_CUDA_DISABLE_GRAPHS']='1'
        model_dir=Path('/home/skysi/VLA/models')/model
        cmd=[str(repo/'.venv-converter/bin/python'), str(repo/'scripts/bench_pi05.py'),
             '--model',str(model_dir/'gguf/model.gguf'),'--out-dir',str(out),
             '--weight-dtype','f16','--act-dtype',args.act_dtype,'--warmup','3','--runs','5',
             '--reference',str(model_dir/'gguf/validation/musa-f32-actions.npy')]
        if stage in ['fa','graph-fa']: cmd.append('--flash-attn')
        (out/'command.json').write_text(json.dumps({'argv':cmd,'environment':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_'))}},indent=2)+'\n')
        print('START',stage,args.act_dtype,model,flush=True)
        with (out/'run.log').open('w') as log:
            try:
                proc=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
                status=proc.returncode
            except subprocess.TimeoutExpired:
                status='timeout'
        row={'stage':stage,'act_dtype':args.act_dtype,'model':model,'exit_code':status}
        if (out/'report.json').exists():
            report=json.loads((out/'report.json').read_text())
            row.update(median_wall_ms=report['median_ms']['wall_ms'],error=report.get('error_vs_reference'),repeat_max_abs=report['repeat_max_abs'])
            row['numerical_pass']=report['error_vs_reference']['relative_peak']<0.005 and report['repeat_max_abs']==0
        results.append(row)
        (record/('summary-'+args.label+args.act_dtype+'-'+'-'.join(args.stages)+'.json')).write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(row),flush=True)
