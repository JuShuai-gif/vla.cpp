from pathlib import Path
import os,subprocess,sys,json
d=Path(__file__).resolve().parent;repo=d.parents[1]
env=dict(os.environ,VLA_DEVICE='0',VLA_N_THREADS='4',VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1')
for k in ['VLA_MUSA_GEMMA_ATTN','VLA_PI05_INPUT_CACHE','VLA_PI05_LANG_BUCKET','VLA_MUSA_NO_VECTOR_PACK','GGML_CUDA_DISABLE_GRAPHS','VLA_PI05_DUMP_PRUNE']:
 env.pop(k,None)
cmd=[sys.executable,str(d/'check_inputs.py'),'current-baseline']
(d/'reference-reuse-command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_'))}},indent=2)+'\n')
with (d/'reference-reuse.log').open('w') as f:
 subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
rows=json.loads((d/'changing-current-baseline/summary.json').read_text())
assert all(e['max_abs']==0 for r in rows for e in r['baseline_errors']),rows
(d/'reference-reuse-summary.json').write_text(json.dumps({'exact_match_to_prior_references':True,'models':rows},indent=2)+'\n')
print('PASS: current baseline and historical varying-input references match exactly')
