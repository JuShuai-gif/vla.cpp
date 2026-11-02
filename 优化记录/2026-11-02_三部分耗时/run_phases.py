#!/usr/bin/env python3
"""Three-phase wall timings; check against unsplit graph outputs and latency."""
from pathlib import Path
import json, os, subprocess, sys
import numpy as np
repo=Path(__file__).resolve().parents[2];record=Path(__file__).resolve().parent
assert subprocess.check_output(['powerprofilesctl','get'],text=True).strip()=='performance'
summaries=[]
for name in ['pi05_cube_merge_cksize50_step3_0929','cup_cube_joint_061_a800_50k_20261022']:
    model=Path('/home/skysi/VLA/models')/name
    reports={}
    for stage in ['whole','split']:
        out=record/name/stage
        if (out/'run.log').exists(): raise SystemExit('Refusing to overwrite '+str(out))
        out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
        for key in ['GGML_CUDA_DISABLE_GRAPHS','GGML_MUSA_DISABLE_GRAPHS','VLA_PI05_SPLIT_PHASES','VLA_PI05_HOST_SPLIT']:
            env.pop(key,None)
        cmd=[sys.executable,str(repo/'scripts/bench_pi05.py'),'--model',str(model/'gguf/model.gguf'),
             '--out-dir',str(out),'--weight-dtype','f16','--act-dtype','f32','--flash-attn',
             '--warmup','3','--runs','5','--reference',str(model/'gguf/validation/musa-f32-actions.npy')]
        if stage=='split':cmd.append('--split-phases')
        (out/'command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_'))}},indent=2)+'\n')
        print('START',name,stage,flush=True)
        with (out/'run.log').open('w') as log:
            subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        reports[stage]=json.loads((out/'report.json').read_text())
        assert reports[stage]['repeat_max_abs']==0 and reports[stage]['error_vs_reference']['relative_peak']<.005
    log=(record/name/'split/run.log').read_text()
    phases=[json.loads(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith('VLA_PI05_PHASES ')]
    assert len(phases)==8,len(phases)
    steady=phases[3:]
    med={key:float(np.median([x[key] for x in steady])) for key in ['vision_ms','prefix_ms','denoise_ms']}
    med['denoise_per_step_ms']=med['denoise_ms']/phases[0]['steps']
    a=np.load(record/name/'whole/actions.npy');b=np.load(record/name/'split/actions.npy')
    error=float(np.max(np.abs(a-b)));assert error==0,error
    whole=reports['whole']['median_ms']['wall_ms'];split=reports['split']['median_ms']['wall_ms']
    row={'model':name,'steps':phases[0]['steps'],'phase_median_ms':med,'phase_samples':steady,
         'whole_graph_wall_ms':whole,'split_graph_wall_ms':split,'split_overhead_ms':split-whole,
         'split_overhead_pct':100*(split/whole-1),'split_vs_whole_max_abs':error,
         'whole_report':reports['whole']['median_ms'],'split_report':reports['split']['median_ms'],
         'error_vs_reference':reports['split']['error_vs_reference']}
    summaries.append(row)
    (record/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(row),flush=True)
