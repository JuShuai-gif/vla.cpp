from pathlib import Path
import os,subprocess,sys,json
repo=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
base=dict(os.environ,CHECK_NUMERIC='1',VLA_DEVICE='0',VLA_N_THREADS='4',VLA_MUSA_VISION_SDPA='1',VLA_PI05_VISION_QKV='1',VLA_MUSA_VISION_EPILOGUE='all')
selected=json.loads((d/'selection.json').read_text())
jobs=[]
for short,name in [('pruned','pi05_cube_merge_cksize50_step3_0929'),('full','cup_cube_joint_061_a800_50k_20261022')]:
 for bucket in [False,True]:
  mode='final-'+short+('-bucket' if bucket else '')
  e=dict(base,**selected[short]);e['CHECK_NUMERIC']='1' if e.get('VLA_PI05_ACTION_LINEAR')=='1' else '0'
  if bucket:e['VLA_PI05_LANG_BUCKET']='16'
  jobs.append((mode,[sys.executable,str(d/'check_inputs.py'),mode,name],e))
for short,name in [('pruned','pi05_cube_merge_cksize50_step3_0929'),('full','cup_cube_joint_061_a800_50k_20261022')]:
 e=dict(base,**selected[short]);e['CHECK_NUMERIC']='1' if e.get('VLA_PI05_ACTION_LINEAR')=='1' else '0'
 jobs.append(('lifecycle-final-'+short,[sys.executable,str(d/'check_lifecycle.py'),name,'final'],e))
jobs.extend([
 ('full-entry',[sys.executable,str(d/'check_full_entry.py')],base),
 ('entry-exact',[sys.executable,str(d/'check_entry_reuse.py')],base),
 ('profile-final',[sys.executable,str(d/'profile_io.py'),'final'],dict(base,PROFILE_SELECTED='1')),
 ('summarize-final',[sys.executable,str(d/'summarize_io.py'),'final'],base),
 ('profile-baseline',[sys.executable,str(d/'profile_io.py'),'baseline'],base),
 ('summarize-baseline',[sys.executable,str(d/'summarize_io.py'),'baseline'],base),
])
rows=[]
for name,cmd,env in jobs:
 print('CHECK',name,flush=True)
 (d/(name+'-command.json')).write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith(('VLA_','GGML_','PROFILE_'))}},indent=2)+'\n')
 with (d/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'name':name,'returncode':p.returncode});(d/'validation-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
