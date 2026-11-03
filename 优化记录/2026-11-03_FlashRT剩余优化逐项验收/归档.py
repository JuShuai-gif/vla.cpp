from pathlib import Path
import json,gzip,hashlib,subprocess
d=Path(__file__).resolve().parent;r=d.parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,n in [('validation-summary.json',12),('random-status.json',4),('abc-status.json',6)]:
 rows=json.loads((d/name).read_text());assert len(rows)==n and all(x['returncode']==0 for x in rows)
s=(d/'ctest-linear.log').read_text();assert '100% tests passed, 0 tests failed out of 25' in s and 'Skipped' not in s
ref=json.loads((d/'baseline-r1/pruned/report.json').read_text())['metadata']
patch=subprocess.check_output(['git','diff',ref['source_commit'],'--','src','scripts','tests','CMakeLists.txt'],cwd=r)
assert hashlib.sha256(patch).hexdigest()==ref['source_diff_sha256']
verified=[]
stages=[x+'-r'+str(i) for x in ['baseline','exact','linear'] for i in [1,2]]+['io-baseline','io-final','final-baseline-phases','final-selected-phases']
for stage in stages:
 for model in ['pruned','full']:
  p=d/stage/model/'report.json';m=json.loads(p.read_text())['metadata']
  for k in ['source_diff_sha256','library_sha256','core_sha256','backend_sha256']:assert m[k]==ref[k],(p,k)
  verified.append(str(p.relative_to(d)))
assert len(verified)==20
assert sha(r/'build-musa-official/libvla.so')==ref['library_sha256']
assert sha(r/'build-musa-official/libvla_core.so')==ref['core_sha256']
for p,h in ref['backend_sha256'].items():assert sha(Path(p))==h
(d/'源码与构建校验.json').write_text(json.dumps({'metadata':ref,'verified_final_reports':verified,'note':'前期候选为不同逐步源码版本，各自保留source.patch和命令；这里只强制核对最终代码的六轮、profile及分段报告。'},ensure_ascii=False,indent=2)+'\n')
for p in list(d.rglob('*')):
 if p.is_file() and (p.suffix in ['.log','.patch'] or p.name in ['nodes.json','trace_musa_api_trace.json']):
  with p.open('rb') as src,gzip.GzipFile(filename=str(p)+'.gz',mode='wb',mtime=0) as dst:
   for b in iter(lambda:src.read(1024*1024),b''):dst.write(b)
external=list(d.rglob('*.npy'))
external += [p for p in Path('/home/skysi/VLA/flashrt-rest-20261103').rglob('*') if p.is_file()]
external += [p for p in [Path('/tmp/pi05-probe-remaining'),Path('/tmp/pi05-probe-remaining-tuned'),Path('/tmp/pi05-probe-mma32')] if p.exists()]
(d/'外部原始文件清单.json').write_text(json.dumps([{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(external)],ensure_ascii=False,indent=2)+'\n')
for p in d.glob('*.py'):compile(p.read_text(),str(p),'exec')
print('verified',len(verified),'final reports; archived',len(external),'external files')
