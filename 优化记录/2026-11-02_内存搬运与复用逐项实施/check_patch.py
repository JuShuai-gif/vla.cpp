from pathlib import Path
import subprocess,tempfile,hashlib,json
r=Path('/home/skysi/code/vla.cpp');source=r/'build-musa-deps/llama';d=r/'优化记录/2026-11-02_内存搬运与复用逐项实施'
with tempfile.TemporaryDirectory(prefix='vla-musa-pristine-') as tmp:
 tmp=Path(tmp)
 archive=subprocess.check_output(['git','archive','HEAD','ggml'],cwd=source)
 subprocess.run(['tar','-x','-C',str(tmp)],input=archive,check=True)
 def apply():
  return subprocess.check_output(['python3',str(r/'scripts/patch_ggml_musa.py'),str(tmp)],text=True)
 log=apply()
 hashes=lambda:{str(p.relative_to(tmp)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp.rglob('*') if p.is_file()}
 first=hashes();secondlog=apply();second=hashes()
 assert first==second
 (d/'patch-idempotence.log').write_text(log+secondlog)
 (d/'patch-idempotence.json').write_text(json.dumps({'pristine_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),'applied_to_pristine':True,'second_apply_unchanged':True,'file_count':len(first)},indent=2)+'\n')
print('patch pristine+idempotence PASS')
