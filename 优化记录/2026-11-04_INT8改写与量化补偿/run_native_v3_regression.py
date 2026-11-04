import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path.insert(0,"/home/skysi/code/robot_runtime-quant/src")
from robot_runtime.setup.runtime import HardwareLease
env=dict(os.environ,LD_LIBRARY_PATH="/usr/local/musa/lib",PYTHONPATH=str(root/"bindings/python")+":"+str(root/"scripts")+":"+str(root/"tests/py"))
commands=[
 ("native-v3-final-build",["cmake","--build","build-musa-quant","-j4"]),
 ("native-v3-ctest",["ctest","--test-dir","build-musa-quant","--output-on-failure","-j1"]),
 ("native-v3-python",[sys.executable,"-m","unittest","-v","test_policy_api","test_precision_bench","test_quantize_gguf","test_pi05_io","test_int8_compensation"])
]
results=[]
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
 for name,cmd in commands:
    with (folder/f"{name}.txt").open("w") as log:proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    row=dict(name=name,command=cmd,returncode=proc.returncode)
    results.append(row);(folder/"native-v3-regression-commands.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(row),flush=True)
    if proc.returncode:raise SystemExit(proc.returncode)
