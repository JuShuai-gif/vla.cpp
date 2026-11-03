from pathlib import Path
import os,sys,json,subprocess
d=Path(__file__).resolve().parent;repo=d.parents[1]
candidate=json.loads((d/'candidate.json').read_text())
base=dict(os.environ,CONFIG_SELECTION='candidate.json',CHECK_TAG='candidate-',VLA_DEVICE='0',VLA_N_THREADS='4')
jobs=[]
for short,name in [('pruned','pi05_cube_merge_cksize50_step3_0929'),('full','cup_cube_joint_061_a800_50k_20261022')]:
 for bucket in [False,True]:
  mode='candidate-'+short+('-bucket'if bucket else '')
  e=dict(base,**candidate[short]);e['CHECK_NUMERIC']='1'if short=='full'else'0'
  if bucket:e['VLA_PI05_LANG_BUCKET']='16'
  jobs.append((mode,[sys.executable,str(d/'check_inputs.py'),mode,name],e))
 e=dict(base,**candidate[short]);e['CHECK_NUMERIC']='1'if short=='full'else'0'
 jobs.append(('candidate-lifecycle-'+short,[sys.executable,str(d/'check_lifecycle.py'),name,'candidate'],e))
jobs.extend([
 ('candidate-full-entry',[sys.executable,str(d/'check_full_entry.py')],base),
 ('candidate-entry-reuse',[sys.executable,str(d/'check_entry_reuse.py')],base),
 ('candidate-profile',[sys.executable,str(d/'profile_io.py'),'candidate'],dict(base,PROFILE_SELECTED='1')),
 ('candidate-summarize',[sys.executable,str(d/'summarize_io.py'),'candidate'],base)])
rows=[]
for name,cmd,env in jobs:
 print(name,flush=True)
 (d/(name+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items()if k.startswith(('VLA_','GGML_','CONFIG_','CHECK_','PROFILE_'))}},indent=2)+'\n')
 with (d/(name+'.log')).open('w')as f:
  p=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'name':name,'returncode':p.returncode})
 (d/'candidate-validation-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)
