#!/usr/bin/env python3
from pathlib import Path
import os,sys,json,subprocess,collections
import numpy as np
repo=Path(__file__).resolve().parents[2]; record=Path(__file__).resolve().parent
stage=sys.argv[1]; runs=int(os.environ.get('BENCH_RUNS','5'))
assert stage and '/' not in stage and runs>0
root=record/stage;root.mkdir(exist_ok=False)
assert subprocess.check_output(['powerprofilesctl','get'],text=True).strip()=='performance'
env=dict(os.environ,VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
(root/'source.patch').write_bytes(subprocess.check_output(['git','diff','HEAD','--','src','scripts','tests','CMakeLists.txt'],cwd=repo))
summary=[]
for short,name in [('pruned','pi05_cube_merge_cksize50_step3_0929'),('full','cup_cube_joint_061_a800_50k_20261022')]:
    if os.environ.get('BENCH_MODEL') and os.environ['BENCH_MODEL']!=short:continue
    dest=root/short;dest.mkdir()
    if os.environ.get('BENCH_SELECTED')=='1':
        for k in ['VLA_PI05_PIN_MAIN_INPUT','VLA_MUSA_FLATTEN_GEMM','VLA_MUSA_ATTN_DIRECT_OUT','VLA_MUSA_VISION_PACK','VLA_PI05_HOST_INPUT','VLA_PI05_ASYNC_VISION','VLA_PI05_VISION_DIRECT','VLA_PI05_KV_REUSE']:env.pop(k,None)
        env.update(json.loads((record/'selection.json').read_text())[short])
    env['VLA_MUSA_GEMMA_ATTN']='fused-denoise' if short=='pruned' else 'fused-all'
    model=Path('/home/skysi/VLA/models')/name
    cmd=[sys.executable,str(repo/'scripts/bench_pi05.py'),'--model',str(model/'gguf/model.gguf'),
         '--out-dir',str(dest),'--weight-dtype','f16','--act-dtype',os.environ.get('BENCH_ACT','f32'),'--flash-attn',
         '--warmup','3','--runs',str(runs),'--reference',str(model/'gguf/validation/musa-f32-actions.npy')]
    if os.environ.get('BENCH_SPLIT')=='1':cmd+=['--split-phases']
    if os.environ.get('BENCH_PROFILE')=='1':
        raw=Path('/home/skysi/VLA/official-node-traces-20261102');raw.mkdir(exist_ok=True)
        cmd=['/opt/moore-threads/moore-perf-system/1.7.2/target-linux-aarch64/msys','profile','-t','musa','--musa-graph-trace=node','-o',str(raw/(stage+'-'+short))]+cmd
    (dest/'command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_','LD_LIBRARY','BENCH_'))}},indent=2)+'\n')
    print('START',stage,short,flush=True)
    with (dest/'run.log').open('w') as log:
        subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    r=json.loads((dest/'report.json').read_text())
    assert r['repeat_max_abs']==0 and r['error_vs_reference']['relative_peak']<.005,r
    row={'model':short,'mean_ms':{k:float(np.mean([v[k] for v in r['runs']])) for k in r['runs'][0]},
         'median_ms':r['median_ms'],'p95_wall_ms':r['p95_wall_ms'],'error':r['error_vs_reference']}
    baseline=record/os.environ.get('BENCH_BASELINE_STAGE','baseline')/short/'actions.npy'
    if baseline.exists():
        a=np.load(dest/'actions.npy');b=np.load(baseline);peak=float(np.max(np.abs(a-b)))/max(float(np.max(np.abs(b))),1e-12)
        assert peak<.005,peak
        row['baseline_relative_peak']=peak
    lines=(dest/'run.log').read_text().splitlines()
    nodes=[json.loads(x.split(' ',1)[1]) for x in lines if x.startswith('VLA_PI05_NODE ')]
    if nodes:
        assert {x['phase'] for x in nodes}=={'vision','prefix','denoise'}
        (dest/'nodes.json').write_text(json.dumps(nodes,indent=2)+'\n')
        counts=collections.Counter((x['phase'],x['op'],json.dumps(x['tensors'],sort_keys=True)) for x in nodes if x['op'] in ('MUL_MAT','FLASH_ATTN_EXT'))
        shapes=[{'phase':phase,'op':op,'tensors':json.loads(ts),'count':count} for (phase,op,ts),count in counts.items()]
        (dest/'shapes.json').write_text(json.dumps(shapes,indent=2)+'\n')
    phases=[json.loads(x.split(' ',1)[1]) for x in lines if x.startswith('VLA_PI05_PHASES ')]
    if phases:row['phase_mean_ms']={k:float(np.mean([x[k] for x in phases[3:]])) for k in ['vision_ms','prefix_ms','denoise_ms']}
    summary.append(row);(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(row),flush=True)
