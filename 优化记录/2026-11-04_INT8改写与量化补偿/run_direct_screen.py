import json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
env=dict(os.environ,LD_LIBRARY_PATH="/usr/local/musa/lib")
cases=[("prefix",2048,32768,320,tm,tn,nw) for tm,tn,nw in
       [(32,128,1),(32,160,1),(32,160,2),(32,320,1),(32,320,2),(32,320,4),(64,128,2),(64,160,1),(64,160,2),(64,320,1)]]
cases += [(name,k,m,n,tm,tn,nw) for name,k,m,n in
          [("down",16384,2048,320),("expert",1024,8192,50),("vision",1152,3456,256)]
          for tm,tn,nw in ([(32,160,1),(32,320,1),(64,128,2)] if n==320 else [(32,64,1),(64,64,2),(32,128,1)])]
results=[]
for name,k,m,n,tm,tn,nw in cases:
    command=[str(root/"validation/quant/bench-row-direct"),*map(str,(k,m,n,20,tm,tn,nw))]
    proc=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=120)
    key=f"{name}-{tm}-{tn}-{nw}"
    (folder/f"direct-{key}.txt").write_text(proc.stdout+proc.stderr)
    result=dict(case=key,command=command,returncode=proc.returncode,rows=[json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")])
    results.append(result)
    (folder/"direct-screen.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(dict(case=key,returncode=proc.returncode,rows=[row for row in result["rows"] if "path" in row])),flush=True)
    if proc.returncode:raise SystemExit(proc.returncode)
