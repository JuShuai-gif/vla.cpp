from pathlib import Path
import gzip,hashlib,json,subprocess
d=Path(__file__).resolve().parent;repo=d.parents[1]
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
packed=[]
for p in sorted(d.rglob('*')):
 if not p.is_file() or not (p.suffix in ('.log','.patch') or p.name in ('nodes.json','trace_musa_api_trace.json')):continue
 b=p.read_bytes();z=gzip.compress(b,mtime=0);assert gzip.decompress(z)==b
 target=p.with_name(p.name+'.gz');target.write_bytes(z)
 packed.append({'source':str(p.relative_to(d)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'gzip_sha256':digest(target)})
(d/'压缩记录清单.json').write_text(json.dumps(packed,ensure_ascii=False,indent=2)+'\n')
paths=list(d.rglob('*.npy'))+list(Path('/home/skysi/VLA/memory-reuse-20261102').rglob('*'))
old=d.parent/'2026-11-02_前缀与动作Attention优化'
for folder,name in [('changing-final','pi05_cube_merge_cksize50_step3_0929'),('changing-full-fused-all','cup_cube_joint_061_a800_50k_20261022')]:
 for suffix in ['','-bucket']:paths+=list((old/(folder+suffix)).glob(name+'-*.npy'))
external=[{'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(set(paths)) if p.is_file()]
(d/'外部产物清单.json').write_text(json.dumps(external,ensure_ascii=False,indent=2)+'\n')
r=json.loads((d/'final-r1/pruned/report.json').read_text());m=r['metadata']
patch=subprocess.check_output(['git','diff',m['source_commit'],'--','src','scripts','tests','CMakeLists.txt'],cwd=repo)
assert patch==(d/'final-r1/source.patch').read_bytes(),'Runtime source changed after measurements'
assert digest(repo/'build-musa-official/libvla.so')==m['library_sha256']
assert digest(repo/'build-musa-official/libvla_core.so')==m['core_sha256']
for p,h in m['backend_sha256'].items():assert digest(p)==h
for stage in ['baseline-r1','final-r1','final-r2','baseline-r2','compare-baseline-phases','compare-final-phases']:
 assert (d/stage/'source.patch').read_bytes()==patch
 for model in ['pruned','full']:
  mr=json.loads((d/stage/model/'report.json').read_text())['metadata']
  for key in ['source_commit','library_sha256','core_sha256','backend_sha256']:assert mr[key]==m[key],(stage,model,key)
checks=json.loads((d/'validation-summary.json').read_text());life=json.loads((d/'lifecycle-status.json').read_text())
assert len(checks)==13 and all(x['returncode']==0 for x in checks)
assert len(life)==14 and all(x['returncode']==0 for x in life)
ctest=(d/'ctest.log').read_text();assert '100% tests passed, 0 tests failed out of 24' in ctest and 'Skipped' not in ctest
exact=json.loads((d/'完整入口基线一致性.json').read_text());assert len(exact)==4 and all(x['max_abs']==0 for x in exact)
(d/'构建与验收.json').write_text(json.dumps({'runtime_source_matches_final_measurements':True,'ctest_passed':24,'validation':checks,'lifecycle':life,'complete_entry_exact':exact,'metadata':m},ensure_ascii=False,indent=2)+'\n')
print('Archived',len(packed),'raw files;',len(external),'external artifacts; source, runtime hashes and validation verified')
