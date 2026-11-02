from pathlib import Path
import gzip,hashlib,json,subprocess
d=Path(__file__).resolve().parent;repo=d.parents[1]
sha=lambda b:hashlib.sha256(b).hexdigest()
packed=[]
for p in sorted(d.rglob('*')):
 if not p.is_file() or not (p.suffix=='.log' or p.suffix=='.patch' or p.name=='nodes.json'):continue
 b=p.read_bytes();z=gzip.compress(b,mtime=0);assert gzip.decompress(z)==b
 p.with_name(p.name+'.gz').write_bytes(z)
 packed.append({'source':str(p.relative_to(d)),'bytes':len(b),'sha256':sha(b),'gzip_sha256':sha(z)})
(d/'压缩记录清单.json').write_text(json.dumps(packed,ensure_ascii=False,indent=2)+'\n')
external=[]
paths=list(d.rglob('*.npy'))+list(Path('/home/skysi/VLA/vision-epilogue-graphs-20261102').glob('*'))+list(Path('/home/skysi/VLA/vision-epilogue-graphs-20261102-noncontract').glob('*'))
old=d.parent/'2026-11-02_前缀与动作Attention优化'
for folder,name in [('changing-final','pi05_cube_merge_cksize50_step3_0929'),('changing-full-fused-all','cup_cube_joint_061_a800_50k_20261022')]:
 for suffix in ['', '-bucket']:
  paths+=list((old/(folder+suffix)).glob(name+'-*.npy'))
for p in sorted(set(paths)):
 if p.is_file():external.append({'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())})
(d/'外部产物清单.json').write_text(json.dumps(external,ensure_ascii=False,indent=2)+'\n')
r=json.loads((d/'final-r1/pruned/report.json').read_text())
patch=subprocess.check_output(['git','diff',r['metadata']['source_commit'],'--','src','scripts','tests','CMakeLists.txt'],cwd=repo)
assert patch==(d/'final-r1/source.patch').read_bytes(),'Runtime source changed after measurement'
m=r['metadata']
assert sha((repo/'build-musa-official/libvla.so').read_bytes())==m['library_sha256']
assert sha((repo/'build-musa-official/libvla_core.so').read_bytes())==m['core_sha256']
for path,h in m['backend_sha256'].items():assert sha(Path(path).read_bytes())==h
checks=json.loads((d/'validation-summary.json').read_text())
assert len(checks)==7 and all(x['returncode']==0 for x in checks)
assert '100% tests passed, 0 tests failed out of 20' in (d/'ctest.log').read_text()
(d/'构建与验收.json').write_text(json.dumps({'runtime_source_matches_final_measurements':True,'ctest_passed':20,'validation':checks,'metadata':m},ensure_ascii=False,indent=2)+'\n')
print('Archived',len(packed),'raw files;',len(external),'external artifacts; source and runtime hashes verified')
