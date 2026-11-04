import json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
env=dict(os.environ,LD_LIBRARY_PATH="/usr/local/musa/lib")
results=[]
for threads in (32,64,128):
    binary=root/f"validation/quant/bench-row-direct-{threads}"
    command=["/usr/local/musa/bin/mcc","-x","musa","-O3","-fsigned-char","--offload-arch=mp_22",f"-DINT8_WARP={threads}","-I/usr/local/musa/include","tests/bench_musa_row_quant_direct.cpp","-L/usr/local/musa/lib","-Wl,-rpath,/usr/local/musa/lib","-lmusart","-lmublas","-lmudnn","-o",str(binary)]
    with (folder/f"threads-{threads}-build.txt").open("w") as log:proc=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:raise SystemExit(proc.returncode)
    for shape in ((96,64,50,32,64,1),(2048,32768,320,64,64,2)):
        k,m,n,tm,tn,nw=shape
        command=[str(binary),*map(str,(k,m,n,10,tm,tn,nw))]
        proc=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=120)
        (folder/f"threads-{threads}-{k}.txt").write_text(proc.stdout+proc.stderr)
        result=dict(threads=threads,shape=shape,command=command,returncode=proc.returncode,rows=[json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")])
        results.append(result);print(json.dumps(result),flush=True)
        (folder/"threads-screen.json").write_text(json.dumps(results,indent=2)+"\n")
        if proc.returncode:break
