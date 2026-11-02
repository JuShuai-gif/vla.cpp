#!/usr/bin/env python3
from pathlib import Path
import json,os,subprocess,sys,collections,gzip
repo=Path(__file__).resolve().parents[2];record=Path(__file__).resolve().parent
msys='/opt/moore-threads/moore-perf-system/1.7.2/target-linux-aarch64/msys'
raw=Path('/home/skysi/VLA/vision-epilogue-graphs-20261102');raw.mkdir(exist_ok=True)
summary=[]
for short,name in [('pruned','pi05_cube_merge_cksize50_step3_0929'),('full','cup_cube_joint_061_a800_50k_20261022')]:
    dest=record/('trace-'+short);dest.mkdir(exist_ok=False)
    env=dict(os.environ,VLA_LIBRARY=str(repo/'build-musa-official/libvla.so'),VLA_DEVICE='0',VLA_N_THREADS='4')
    env['VLA_MUSA_GEMMA_ATTN']='fused-denoise' if short=='pruned' else 'fused-all'
    cmd=[msys,'profile','-t','musa','--musa-graph-trace=graph','-o',str(raw/short),
         sys.executable,str(repo/'scripts/bench_pi05.py'),'--model',str(Path('/home/skysi/VLA/models')/name/'gguf/model.gguf'),
         '--out-dir',str(dest),'--weight-dtype','f16','--act-dtype','f32','--flash-attn','--warmup','2','--runs','2']
    (dest/'command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_'))}},indent=2)+'\n')
    print('TRACE',short,flush=True)
    with (dest/'profile.log').open('w') as log:
        subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        subprocess.run([msys,'stats','-r','musa_api_trace','-f','json','-o',str(dest/'api'),str(raw/(short+'.msys-rep'))],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    trace=dest/'api_musa_api_trace.json'
    rows=json.loads(trace.read_text())
    trace.with_suffix('.json.gz').write_bytes(gzip.compress(trace.read_bytes(),mtime=0))
    trace.unlink()
    allcounts=collections.Counter(x['Name'] for x in rows)
    last=max(i for i,x in enumerate(rows) if 'GraphInstantiate' in x['Name'])
    steady=collections.Counter(x['Name'] for x in rows[last+1:])
    result={'model':name,'all_graph_api_counts':{k:v for k,v in allcounts.items() if 'Graph' in k or 'Capture' in k or k=='musaLaunchKernel'},
            'after_last_instantiate':dict(steady)}
    assert sum(v for k,v in allcounts.items() if 'GraphInstantiate' in k)==3,result
    assert steady['musaGraphLaunch']>=6 and steady['musaLaunchKernel']==0,result
    summary.append(result);(record/'summary-graph-traces.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(result),flush=True)
