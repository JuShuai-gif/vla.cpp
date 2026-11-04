"""Serial feasibility screen. CPU references validate INT8 arithmetic, not BF16 policy accuracy."""
import json, os, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=Path(__file__).resolve().parent
if len(sys.argv)>1:
    folder=folder/sys.argv[1]
    folder.mkdir(exist_ok=True)
env=dict(os.environ,LD_LIBRARY_PATH="/usr/local/musa/lib")
cases=[("edge-k32-n1",32,64,1,32,32,1),
       ("edge-k96-n50",96,64,50,32,64,1),
       *[(f"prefix-gu-{tm}-{tn}-{nw}",2048,32768,320,tm,tn,nw)
         for tm,tn,nw in [(32,64,1),(32,64,2),(32,64,4),(64,64,1),(64,64,2),(64,64,4)]],
       ("prefix-down",16384,2048,320,64,64,2),
       ("expert-gu",1024,8192,50,64,64,2),
       ("vision-qkv",1152,3456,256,64,64,1)]
results=[]
for name,k,m,n,tm,tn,nw in cases:
    command=[str(root/("validation/quant/"+(sys.argv[2] if len(sys.argv)>2 else "bench-row-shared"))),*map(str,[k,m,n,20,tm,tn,nw])]
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
    (folder/(name+".txt")).write_text(result.stdout+result.stderr)
    rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    record=dict(name=name,command=command,returncode=result.returncode,rows=rows)
    results.append(record)
    (folder/"shared-screen.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(dict(name=name,returncode=result.returncode,summary=[r for r in rows if "path" in r])),flush=True)
    if result.returncode: raise SystemExit(result.returncode)
