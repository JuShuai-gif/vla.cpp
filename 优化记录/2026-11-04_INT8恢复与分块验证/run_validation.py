"""Run regression checks and offline integration sequentially."""
import json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=Path(__file__).resolve().parent
python="/home/skysi/code/robot_runtime/.venv/bin/python"
env=dict(os.environ,PYTHONPATH=str(root/"bindings/python")+":"+str(root/"scripts")+":"+str(root/"tests/py"),LD_LIBRARY_PATH="/usr/local/musa/lib")
jobs=[
("regression-build",["cmake","--build","build-musa-quant","-j4"]),
("regression-ctest",["ctest","--test-dir","build-musa-quant","--output-on-failure"]),
("regression-python",[python,"-m","unittest","-v","test_policy_api","test_precision_bench","test_quantize_gguf","test_pi05_io"]),
("runtime-replay",[python,str(folder/"run_runtime_replay.py")])]
results=[]
for name,cmd in jobs:
    with (folder/f"{name}.txt").open("w") as log:
        proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    result=dict(name=name,command=cmd,returncode=proc.returncode)
    results.append(result)
    (folder/"validation-commands.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    if proc.returncode:raise SystemExit(proc.returncode)
