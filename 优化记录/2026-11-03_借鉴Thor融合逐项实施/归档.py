from pathlib import Path
import json,gzip,hashlib,subprocess
d=Path(__file__).resolve().parent;r=d.parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
checks=json.loads((d/'validation-summary.json').read_text())
assert len(checks)==12 and all(x['returncode']==0 for x in checks)
assert all(x['returncode']==0 for x in json.loads((d/'comparison-status.json').read_text()))
assert '100% tests passed, 0 tests failed out of 25' in (d/'ctest-02.log').read_text()
ref=json.loads((d/'control/pruned/report.json').read_text())['metadata']
patch=subprocess.check_output(['git','diff',ref['source_commit'],'--','src','scripts','tests','CMakeLists.txt'],cwd=r)
assert hashlib.sha256(patch).hexdigest()==ref['source_diff_sha256']
verified=[]
for p in d.glob('*/*/report.json'):
 if p.parts[-3]=='baseline':continue
 m=json.loads(p.read_text())['metadata']
 for k in ['source_diff_sha256','library_sha256','core_sha256','backend_sha256']:assert m[k]==ref[k],(p,k)
 verified.append(str(p.relative_to(d)))
assert sha(r/'build-musa-official/libvla.so')==ref['library_sha256']
assert sha(r/'build-musa-official/libvla_core.so')==ref['core_sha256']
for p,h in ref['backend_sha256'].items():assert sha(Path(p))==h
(d/'源码与构建校验.json').write_text(json.dumps({'baseline_commit':ref['source_commit'],'metadata':ref,'verified_reports':verified},indent=2)+'\n')
for p in list(d.rglob('*')):
 if p.is_file() and (p.suffix in ['.log','.patch'] or p.name in ['nodes.json','trace_musa_api_trace.json']):
  with p.open('rb') as src,gzip.GzipFile(filename=str(p)+'.gz',mode='wb',mtime=0) as dst:
   for b in iter(lambda:src.read(1024*1024),b''):dst.write(b)
raw=Path('/home/skysi/VLA/thor-fusion-20261103')
external=list(d.rglob('*.npy'))+[p for p in raw.rglob('*') if p.is_file()]
(d/'外部原始文件清单.json').write_text(json.dumps([{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(external)],ensure_ascii=False,indent=2)+'\n')
for p in d.glob('*.py'):compile(p.read_text(),str(p),'exec')
print('archived',len(verified),'reports;',len(external),'external files')
