import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path.insert(0,"/home/skysi/code/robot_runtime-quant/src")
from robot_runtime.setup.runtime import HardwareLease
env=dict(os.environ,LD_LIBRARY_PATH="/usr/local/musa/lib")
results=[]
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
 for name,cmd in [
    ("native-v2-build",["cmake","--build","build-musa-quant","-j4","--target","test_musa_int8","bench_musa_native_int8","vla"]),
    ("native-v2-test",[str(root/"build-musa-quant/tests/test_musa_int8")]),
    ("graph-gemm-v2",[str(root/"build-musa-quant/tests/bench_musa_native_int8")])]:
    with (folder/f"{name}.txt").open("w") as log:proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    row=dict(name=name,command=cmd,returncode=proc.returncode)
    results.append(row);(folder/"native-v2-build-commands.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(row),flush=True)
    if proc.returncode:raise SystemExit(proc.returncode)
