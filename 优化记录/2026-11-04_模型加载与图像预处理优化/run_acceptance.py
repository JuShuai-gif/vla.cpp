from pathlib import Path
import os,sys,json,subprocess,time
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1]
# GPU tests must finish before timed inference.
while not(d/'ctest-final.log').exists()or 'Total Test time'not in(d/'ctest-final.log').read_text():time.sleep(1)
assert '100% tests passed' in(d/'ctest-final.log').read_text()
for script,log in [('bench_preprocess.py','preprocess-acceptance.log'),('check_prompts.py','prompt-correctness.log')]:
 with(d/log).open('w')as f:r=subprocess.run([sys.executable,str(d/script)],cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 assert r.returncode==0,(script,r.returncode)
rows=[];seen={}
for variant in ['A','B','B','A']:
 seen[variant]=seen.get(variant,0)+1
 for short in ['pruned','full']:
  tag='entry-'+variant+str(seen[variant])+'-'+short;dest=d/(tag+'.json')
  cmd=[sys.executable,str(d/'entry_probe.py'),variant,short,str(dest)]
  env=dict(os.environ)
  for k in list(env):
   if k.startswith('VLA_')or k=='MEASURE_LIBRARY':env.pop(k)
  root=Path('/home/skysi/VLA/startup-preprocess-20261104/baseline-lib')if variant=='A'else repo/'build-musa-official'
  env.update(VLA_LIBRARY=str(root/'libvla.so'),VLA_N_THREADS='4',LD_LIBRARY_PATH=str(root)+':/usr/local/musa/lib')
  (d/(tag+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:env[k]for k in ['VLA_LIBRARY','VLA_N_THREADS','LD_LIBRARY_PATH']}},indent=2)+'\n')
  with(d/(tag+'.log')).open('w')as f:r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
  row={'stage':tag,'returncode':r.returncode};rows.append(row)
  (d/'acceptance-status.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
  assert r.returncode==0
  a=json.loads(dest.read_text());assert str(root/'libvla_core.so')in a['loaded_core'],a['loaded_core']
  ref=d.parent/'2026-11-03_四项深化与MCU热点优化/final-cli-f32'/(short+'.json')
  got=np.asarray(json.loads(dest.with_suffix('.actions.json').read_text())['actions'],np.float32)
  expected=np.asarray(json.loads(ref.read_text())['actions'],np.float32);assert np.array_equal(got,expected)
summary={}
for short in ['pruned','full']:
 summary[short]={}
 for variant in ['A','B']:
  a=[json.loads((d/('entry-'+variant+str(i)+'-'+short+'.json')).read_text())for i in [1,2]]
  summary[short][variant]={k:float(np.mean([x[k]for x in a]))for k in ['metadata_ms','cpp_load_ms','first_predict_ms','first_action_from_entry_ms','max_rss_kib']}
  hot=[v for x in a for v in x['resident_ms']]
  summary[short][variant].update(resident_mean_ms=float(np.mean(hot)),resident_p95_ms=float(np.percentile(hot,95)))
(d/'acceptance.json').write_text(json.dumps(summary,indent=2)+'\n')
print('ACCEPTANCE',summary,flush=True)
