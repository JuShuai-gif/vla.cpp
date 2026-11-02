from pathlib import Path
import os,sys,subprocess,json
import numpy as np
d=Path(__file__).resolve().parent;r=d.parents[1]
base=dict(os.environ,VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_MUSA_VISION_EPILOGUE='all',
 VLA_MUSA_VISION_PACK='1',VLA_MUSA_ATTN_DIRECT_OUT='1',VLA_PI05_HOST_INPUT='pinned',
 VLA_PI05_ASYNC_VISION='1',VLA_PI05_VISION_DIRECT='1',VLA_PI05_KV_REUSE='1',VLA_PI05_PIN_MAIN_INPUT='1',
 BENCH_MODEL='full',BENCH_RUNS='10')
for k in ['VLA_MUSA_FLATTEN_GEMM','VLA_MUSA_TRACE_GEMM','BENCH_SPLIT']:base.pop(k,None)
for stage in ['fold-control-r1','fold-enabled-r1','fold-enabled-r2','fold-control-r2']:
 env=dict(base)
 if 'enabled' in stage:env['VLA_MUSA_FLATTEN_GEMM']='1'
 print('MEASURE',stage,flush=True)
 with (d/(stage+'.log')).open('w') as f:subprocess.run([sys.executable,str(d/'run_bench.py'),stage],env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
values={}
for mode in ['control','enabled']:
 values[mode]=[[x['wall_ms'] for x in json.loads((d/f'fold-{mode}-r{i}'/'full/report.json').read_text())['runs']] for i in [1,2]]
a,b=[float(np.mean(values[x])) for x in ['control','enabled']]
gain=(a-b)/a
round_gains=[float(np.mean(values['control'][j])-np.mean(values['enabled'][j])) for j in [0,1]]
adopt=gain>.005 and all(x>0 for x in round_gains)
result={'control_mean_ms':a,'enabled_mean_ms':b,'saved_percent':gain*100,'round_saved_ms':round_gains,'n_per_config':20,'adopt':adopt,'selection_rule':'mean improvement >0.5% and positive in both rounds','samples':values}
(d/'GEMM展平交错复测.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
selected=json.loads((d/'selection.json').read_text())
for config in selected.values():config['VLA_PI05_PIN_MAIN_INPUT']='1';config.pop('VLA_MUSA_FLATTEN_GEMM',None)
if adopt:selected['full']['VLA_MUSA_FLATTEN_GEMM']='1'
(d/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
print(json.dumps(result),flush=True)
