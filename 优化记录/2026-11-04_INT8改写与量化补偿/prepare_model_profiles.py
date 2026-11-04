import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
env=dict(os.environ,PYTHONPATH=str(root/"scripts")+":"+str(root/"bindings/python"),OPENBLAS_NUM_THREADS="4")
python="/home/skysi/code/robot_runtime/.venv/bin/python"
results=[]
for variant in ("pruned","full"):
    config=root/f"优化记录/2026-11-04_INT8恢复与分块验证/{variant}-load-policy.json"
    cfg=json.loads(config.read_text());cfg["precision_profiles"]={}
    for name,opts in (("int8-expert-plain",[]),("int8-expert-smooth",["--alpha","0.75"]),("int8-expert-compensated",["--alpha","0.75","--outliers","32"])):
        directory=root/f"validation/quant/native-int8/{variant}-{name}"
        cmd=[python,"scripts/export_int8_parameters.py","--config",str(config),"--importance",str(root/f"validation/quant/calibration/{variant}-120/importance.json"),"--output-dir",str(directory),*opts]
        proc=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True)
        results.append(dict(command=cmd,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
        (folder/"parameter-export.json").write_text(json.dumps(results,indent=2)+"\n")
        if proc.returncode:print(proc.stderr);raise SystemExit(proc.returncode)
        cfg["precision_profiles"][name]=dict(quant_route="native_int8",int8_parameters=str(directory/"manifest.json"),status="experimental")
        print(proc.stdout,flush=True)
    (folder/f"{variant}-native-policy.json").write_text(json.dumps(cfg,indent=2)+"\n")
