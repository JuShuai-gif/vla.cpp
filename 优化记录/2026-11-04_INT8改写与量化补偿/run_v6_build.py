import os,sys,subprocess,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path.insert(0,"/home/skysi/code/robot_runtime-quant/src")
from robot_runtime.setup.runtime import HardwareLease
python="/home/skysi/code/robot_runtime/.venv/bin/python"
env=dict(os.environ,PYTHONPATH=str(root/"scripts")+":"+str(root/"bindings/python")+":"+str(root/"tests/py"),LD_LIBRARY_PATH="/usr/local/musa/lib")
commands=[]
def run(label,cmd,timeout=1800):
 begin=time.time()
 with (folder/(label+".txt")).open("w") as log:p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 row=dict(label=label,command=cmd,returncode=p.returncode,seconds=time.time()-begin)
 commands.append(row);(folder/"native-v6-build-commands.json").write_text(json.dumps(commands,indent=2)+"\n")
 print(json.dumps(row),flush=True)
 if p.returncode:raise SystemExit(p.returncode)
run("native-v6-build",["cmake","--build","build-musa-quant","-j","4"])
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
 run("native-v6-ctest",["ctest","--test-dir","build-musa-quant","-R","musa_native_int8","--output-on-failure"])
run("native-v6-python",[python,"-m","unittest","test_policy_api","test_precision_bench","test_quantize_gguf","test_pi05_io","test_int8_compensation"])
for v in ("pruned","full"):
 run("native-v6-export-"+v,[python,"scripts/export_int8_parameters.py","--config",str(folder/(v+"-main-native-policy.json")),"--importance",str(root/("validation/quant/calibration/"+v+"-120/importance.json")),"--output-dir",str(root/("validation/quant/native-int8/"+v+"-int8-main-smooth")),"--alpha","0.75","--prefix","--prefix-alpha","0.5"])
run("native-v6-models",[python,str(folder/"run_native_models_v6.py")],2400)
