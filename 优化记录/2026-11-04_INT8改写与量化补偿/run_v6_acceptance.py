"""Run sealed acceptance only after frozen development ABBA succeeds."""
import json,subprocess,sys,os,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path.insert(0,"/home/skysi/code/robot_runtime-quant/src")
from robot_runtime.setup.runtime import HardwareLease
env=dict(os.environ,LD_LIBRARY_PATH="/usr/local/musa/lib")
for v in ("pruned","full"):
 j=json.loads((folder/f"v6-abba-{v}-summary.json").read_text())
 assert j["offline_accuracy_pass"] and j["development_performance_pass"] and all(j["action_repeat_exact"].values())
runs=[]
def run(name,command):
 begin=time.time()
 with (folder/(name+".txt")).open("w") as log:p=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 row=dict(name=name,command=command,returncode=p.returncode,elapsed_seconds=time.time()-begin)
 runs.append(row);(folder/"v6-acceptance-commands.json").write_text(json.dumps(runs,indent=2)+"\n");print(json.dumps(row),flush=True)
 if p.returncode:raise SystemExit(p.returncode)
run("v6-heldout-driver",[sys.executable,str(folder/"run_v6_heldout.py")])
held=json.loads((folder/"v6-heldout-summary.json").read_text())
assert len(held)==2 and all(x["offline_accuracy_pass"] and x["native_p50_reduction"]>0 and x["predict_p50_reduction"]>0 for x in held)
run("v6-runtime-driver",[sys.executable,str(folder/"run_v6_runtime.py")])
runtime=json.loads((folder/"v6-runtime-summary.json").read_text())
assert len(runtime)==2 and all(x["software_chain_performance_pass"] for x in runtime)
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
 run("native-v6-ctest-all",["ctest","--test-dir","build-musa-quant","--output-on-failure"])
print("All frozen V6 offline acceptance stages passed; final provenance/protected-state checks and archive remain.",flush=True)
