import json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
env=dict(os.environ,LD_LIBRARY_PATH="/usr/local/musa/lib")
results=[]
for split in (1,2,4):
    binary=root/f"validation/quant/bench-int8-accum-{split}"
    command=["/usr/local/musa/bin/mcc","-x","musa","-O3","-fsigned-char","--offload-arch=mp_22","-res-usage",f"-DPREFETCH_STEPS={split}","-I/usr/local/musa/include",str(folder/"accumulator-prototype.cpp"),"-L/usr/local/musa/lib","-Wl,-rpath,/usr/local/musa/lib","-lmusart","-lmublas","-lmudnn","-o",str(binary)]
    with (folder/f"accum-{split}-build.txt").open("w") as log:proc=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:raise SystemExit(proc.returncode)
    for k,m,n,tm,tn,nw in [(96,128,50,32,64,1),*[(1024,8192,50,tm,tn,nw) for tm,tn,nw in [(32,64,1),(32,64,2),(64,64,1),(64,64,2),(128,32,1),(128,64,1)]]]:
        command=[str(binary),*map(str,(k,m,n,20,tm,tn,nw))]
        proc=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=120)
        key=f"accum-{split}-{k}-{tm}-{tn}-{nw}";(folder/f"{key}.txt").write_text(proc.stdout+proc.stderr)
        row=dict(case=key,command=command,returncode=proc.returncode,rows=[json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")])
        results.append(row);(folder/"accumulator-screen.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps(dict(case=key,returncode=proc.returncode,summary=[s for s in row["rows"] if "path" in s])),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)
