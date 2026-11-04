"""Offline adapter smoke test; no robot or camera devices."""
import json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=Path(__file__).resolve().parent
env=dict(os.environ,PYTHONPATH="/home/skysi/code/robot_runtime-quant/src:"+str(root/"bindings/python")+":"+str(root/"scripts"),LD_LIBRARY_PATH="/usr/local/musa/lib")
results=[]
for variant in ("pruned","full"):
    command=["/home/skysi/code/robot_runtime/.venv/bin/python","-m","robot_runtime.replay",
             "--root","/home/skysi/code/robot_runtime",
             "--config",str(folder/"runtime-native-config.json"),"--profile",variant,
             "--observation",f"/home/skysi/code/robot_runtime/validation/vla_cpp/{variant}-observation.npz",
             "--samples","3","--warmup","3","--output",str(folder/f"runtime-{variant}-native-int8.json")]
    with (folder/f"runtime-{variant}-native-int8.log").open("w") as log:
        proc=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    result=dict(variant=variant,command=command,returncode=proc.returncode)
    results.append(result)
    (folder/"runtime-native-commands.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    if proc.returncode:raise SystemExit(proc.returncode)
