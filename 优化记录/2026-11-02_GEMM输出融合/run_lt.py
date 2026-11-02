from pathlib import Path
import subprocess,os,json
d=Path(__file__).resolve().parent;out=d/'lt';out.mkdir(exist_ok=True)
cmd=['/usr/local/musa/bin/mcc','-O3','-std=c++17','-x','musa','-mtgpu','--cuda-gpu-arch=mp_22','-ffp-contract=off','-I/usr/local/musa/include',str(d/'probe_lt_epilogue.cpp'),'-L/usr/local/musa/lib','-lmusart','-Wl,-rpath,/usr/local/musa/lib','-lmublas','-lmublasLt','-o','/tmp/pi05-probe-lt-epilogue']
with (out/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
rows=[]
for shape in [(1152,256,1152,1),(4304,256,1152,1),(1152,256,4304,1),(1152,256,1152,2),(4304,256,1152,2),(1152,256,4304,2)]:
 for gelu in [0,1]:
  name='x'.join(map(str,shape))+('-gelu' if gelu else '-bias')
  cmd=['/tmp/pi05-probe-lt-epilogue',*map(str,shape),str(gelu)]
  with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/musa/lib'),timeout=120)
  lines=(out/(name+'.log')).read_text()
  rows.append(dict(shape=shape,gelu=gelu,returncode=r.returncode,output=lines))
  print(name,r.returncode,lines[-600:],flush=True)
  (out/'summary.json').write_text(json.dumps(rows,indent=2))
