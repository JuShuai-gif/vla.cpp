from pathlib import Path
import json,os,sys,subprocess,time
import numpy as np
d=Path(__file__).resolve().parent;repo=d.parents[1]
while True:
 p=d/'hot-mcu-status-valid.json'
 if p.exists():
  rows=json.loads(p.read_text())
  if len(rows)==4:
   assert all(x['valid_kernel_report']for x in rows),rows
   break
 time.sleep(1)
with(d/'build-final.log').open('w')as f:
 r=subprocess.run(['cmake','--build','build-musa-official','-j','8'],cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=900)
assert r.returncode==0
env=dict(os.environ,VLA_MUSA_SMALL_PIPE='shape')
with(d/'ctest-final.log').open('w')as f:
 r=subprocess.run(['ctest','--test-dir','build-musa-official','--output-on-failure','-j','1'],cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
assert r.returncode==0
configs={'A':'baseline-env.json','B':'pack-combined-v2-env.json','C':'combined-pipe4-env.json','D':'combined-shape-env.json'}
status=[];seen={}
for variant in ['A','B','C','D','D','C','B','A']:
 seen[variant]=seen.get(variant,0)+1
 for short in ['pruned','full']:
  tag='accept-'+variant+'-r'+str(seen[variant]);out=d/tag/short;out.parent.mkdir(exist_ok=True)
  cmd=[sys.executable,str(d/'bench_real.py'),short,str(out)]
  env=dict(os.environ,BENCH_CONFIG=configs[variant],BENCH_WARMUP='3',BENCH_RUNS='15')
  (out.parent/(short+'-command.json')).write_text(json.dumps({'argv':cmd,'config':configs[variant],'warmup':3,'runs':15},indent=2)+'\n')
  with(d/(tag+'-'+short+'.log')).open('w')as f:
   r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  row={'variant':variant,'round':seen[variant],'model':short,'returncode':r.returncode}
  if r.returncode==0:
   a=json.loads((out/'report.json').read_text())
   row.update({k:a[k]for k in ['mean_ms','reference_relative_peak']})
  status.append(row);(d/'final-accept-status.json').write_text(json.dumps(status,indent=2)+'\n');print(row,flush=True)
  assert r.returncode==0,row
summary={};selection={}
for short,limit in [('pruned',500),('full',800)]:
 result={}
 for variant in configs:
  reports=[json.loads((d/('accept-'+variant+'-r'+str(i))/short/'report.json').read_text())for i in [1,2]]
  times=[x for a in reports for x in a['wall_ms']]
  result[variant]={'samples':len(times),'mean_ms':float(np.mean(times)),'p95_ms':float(np.percentile(times,95)),
                   'max_ms':max(times),'relative_peak':max(a['reference_relative_peak']for a in reports)}
  assert len(times)==30 and result[variant]['relative_peak']==0,result
 chosen=min(['B','C','D'],key=lambda k:result[k]['mean_ms'])
 assert result[chosen]['p95_ms']<limit and result[chosen]['max_ms']<limit,result
 selection[short]=json.loads((d/configs[chosen]).read_text())[short]
 if short=='full':selection[short].pop('VLA_PI05_PRUNE_MEAN',None)
 summary[short]={'variants':result,'chosen':chosen,'speedup_percent':100*(1-result[chosen]['mean_ms']/result['A']['mean_ms'])}
(d/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
(repo/'scripts/pi05_musa_profiles.json').write_text(json.dumps(selection,indent=2)+'\n')
(d/'final-acceptance.json').write_text(json.dumps(summary,indent=2)+'\n')
print('ACCEPTANCE',json.dumps(summary),flush=True)
