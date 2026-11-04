import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path.insert(0,"/home/skysi/code/robot_runtime-quant/src")
from robot_runtime.setup.runtime import HardwareLease
runtime="/home/skysi/code/robot_runtime/.venv/bin/python"
framework="/home/skysi/.conda/envs/lerobot_musa312/bin/python"
env=dict(os.environ,PYTHONPATH=str(root/"bindings/python")+":"+str(root/"scripts"),LD_LIBRARY_PATH="/usr/local/musa/lib")
results=[]
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
 for variant in ("pruned","full"):
    old=json.loads((root/f"优化记录/2026-11-04_低精度实现与验收/{variant}-bf16-reference-30.json").read_text())
    inputs=root/f"validation/quant/bf16-inputs-{variant}-development-90"
    commands=[
      (variant+"-freeze-development-90",[runtime,"scripts/prepare_pi05_bf16_inputs.py","--config",str(folder/f"{variant}-native-policy.json"),
        "--corpus","validation/quant/data/real-v1/manifest.json","--samples","90","--split","development","--output-dir",str(inputs)]),
      (variant+"-bf16-reference-90",[framework,"scripts/reference_pi05_bf16.py","--checkpoint",old["checkpoint"],
        "--inputs",str(inputs/"manifest.json"),"--samples","90","--output",str(folder/f"{variant}-bf16-reference-90.json")])]
    for name,cmd in commands:
      with (folder/f"{name}.log").open("w") as log:proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
      row=dict(name=name,command=cmd,returncode=proc.returncode);results.append(row)
      (folder/"development-reference-commands.json").write_text(json.dumps(results,indent=2)+"\n");print(json.dumps(row),flush=True)
      if proc.returncode:raise SystemExit(proc.returncode)
