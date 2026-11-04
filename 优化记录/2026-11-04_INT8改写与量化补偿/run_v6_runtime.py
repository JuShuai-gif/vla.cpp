"""Production replay entry point using isolated library, no robot/camera devices."""
import json,os,subprocess,time,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
held=json.loads((folder/"v6-heldout-summary.json").read_text())
assert len(held)==2 and all(x["offline_accuracy_pass"] for x in held)
env=dict(os.environ,PYTHONPATH="/home/skysi/code/robot_runtime-quant/src:"+str(root/"bindings/python")+":"+str(root/"scripts"),LD_LIBRARY_PATH="/usr/local/musa/lib")
frozen=json.loads((folder/"candidate-v6-freeze.json").read_text())
cfg=json.loads((folder/"runtime-v6-config.json").read_text())
assert all(x is None for group in cfg["devices"].values() for x in group.values())
assert all(p["execute_robot"] is False and p["dry_run"] is True for p in cfg["profiles"].values())
results=[];summaries=[]
for variant in ("pruned","full"):
 reports={}
 for precision,pair in (("baseline",0),("int8-main-smooth",0),("int8-main-smooth",1),("baseline",1)):
  for name,expected in frozen["hashes"].items():
   if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError("Frozen artifact changed: "+name)
  output=folder/f"v6-runtime-{variant}-{precision}-{pair}.json"
  if output.exists():raise ValueError("Refusing to overwrite "+str(output))
  command=["/home/skysi/code/robot_runtime/.venv/bin/python","-m","robot_runtime.replay","--root","/home/skysi/code/robot_runtime","--config",str(folder/"runtime-v6-config.json"),"--profile",variant,"--precision",precision,"--observation",f"/home/skysi/code/robot_runtime/validation/vla_cpp/{variant}-observation.npz","--samples","30","--warmup","3","--output",str(output)]
  begin=time.time()
  with output.with_suffix(".log").open("w") as log:p=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  row=dict(variant=variant,precision=precision,pair=pair,command=command,returncode=p.returncode,elapsed_seconds=time.time()-begin)
  if p.returncode==0:
   report=json.loads(output.read_text());reports[(precision,pair)]=report
   assert report["network_requests"]==0 and report["action_shape"]==[50,7]
   assert report["config"]["profiles"][variant]["execute_robot"] is False
   row["predict_ms"]=report["predict_ms"]
  results.append(row);(folder/"v6-runtime-commands.json").write_text(json.dumps(results,indent=2)+"\n");print(json.dumps(row),flush=True)
  if p.returncode:raise SystemExit(p.returncode)
 gains=[1-reports[("int8-main-smooth",i)]["predict_ms"]["p50"]/reports[("baseline",i)]["predict_ms"]["p50"] for i in (0,1)]
 summary=dict(variant=variant,pairs=2,samples_per_process=30,order="A0 B0 B1 A1",pair_predict_p50_reductions=gains,software_chain_performance_pass=all(x>0 for x in gains),finite_action_shape=[50,7],robot_camera_network_io=False)
 summaries.append(summary);(folder/"v6-runtime-summary.json").write_text(json.dumps(summaries,indent=2)+"\n");print(json.dumps(summary),flush=True)
