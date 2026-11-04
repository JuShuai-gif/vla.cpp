"""Paired policy screen of actual native INT8, no robot or camera access."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path[:0]=[str(root/"scripts"),str(root/"bindings/python"),"/home/skysi/code/robot_runtime-quant/src"]
from bench_pi05_precision import compare
from robot_runtime.setup.runtime import HardwareLease
python="/home/skysi/code/robot_runtime/.venv/bin/python"
env=dict(os.environ,PYTHONPATH=str(root/"scripts")+":"+str(root/"bindings/python"),LD_LIBRARY_PATH="/usr/local/musa/lib")
results=[]
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
 for variant in ("pruned","full"):
    order=["baseline","int8-expert-smooth"]
    for precision in order:
        output=folder/f"v2-{variant}-{precision}.json"
        cmd=[python,"scripts/bench_pi05_precision.py","--config",str(folder/f"{variant}-native-policy.json"),"--corpus","validation/quant/data/real-v1/manifest.json","--samples","30","--warmup","3","--precision",precision,"--output",str(output)]
        begin=time.time()
        with output.with_suffix(".log").open("w") as log:proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        row=dict(variant=variant,precision=precision,command=cmd,returncode=proc.returncode,elapsed_seconds=time.time()-begin)
        if proc.returncode==0:
            report=json.loads(output.read_text())
            comp=compare(root/f"优化记录/2026-11-04_低精度实现与验收/{variant}-bf16-reference-30.json",output,folder/f"v2-{variant}-baseline.json")
            output.with_name(output.stem+"-comparison.json").write_text(json.dumps(comp,indent=2)+"\n")
            row.update(native_ms=report["native_ms"],predict_ms=report["predict_ms"],load_seconds=report["load_seconds"],
                       offline_accuracy_pass=comp["offline_accuracy_pass"],minimum_action_cosine=comp["cosine"]["minimum_action_vector"],native_p50_reduction=comp["native_p50_reduction"])
        results.append(row);(folder/"native-model-v2-screen.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps(row),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)
