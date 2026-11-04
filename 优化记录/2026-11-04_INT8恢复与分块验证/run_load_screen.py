"""Paired development screen; never accesses robot/camera devices."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=Path(__file__).resolve().parent
sys.path[:0]=[str(root/"scripts"),str(root/"bindings/python"),"/home/skysi/code/robot_runtime-quant/src"]
from bench_pi05_precision import compare
from robot_runtime.setup.runtime import HardwareLease
python="/home/skysi/code/robot_runtime/.venv/bin/python"
env=dict(os.environ,PYTHONPATH=str(root/"bindings/python")+":"+str(root/"scripts"),LD_LIBRARY_PATH="/usr/local/musa/lib")
results=[]
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
    for variant in ("pruned","full"):
        order=("baseline","q8-prefix-mlp-blas","q8-prefix-mlp-load-f16") if variant=="pruned" else ("baseline","q8-prefix-mlp-load-f16","q8-prefix-mlp-blas")
        for precision in order:
            output=folder/f"{variant}-{precision}.json"
            cmd=[python,"scripts/bench_pi05_precision.py","--config",str(folder/f"{variant}-load-policy.json"),"--corpus","validation/quant/data/real-v1/manifest.json","--samples","30","--warmup","3","--precision",precision,"--output",str(output)]
            begin=time.time()
            with output.with_suffix(".log").open("w") as log:
                proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            result=dict(variant=variant,precision=precision,command=cmd,returncode=proc.returncode,elapsed_seconds=time.time()-begin)
            if proc.returncode==0:
                report=json.loads(output.read_text())
                result.update(native_ms=report["native_ms"],predict_ms=report["predict_ms"],load_seconds=report["load_seconds"])
                reference=root/f"优化记录/2026-11-04_低精度实现与验收/{variant}-bf16-reference-30.json"
                comparison=compare(reference,output,folder/f"{variant}-baseline.json")
                output.with_name(output.stem+"-comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
                result.update(offline_accuracy_pass=comparison["offline_accuracy_pass"],minimum_action_cosine=comparison["cosine"]["minimum_action_vector"],native_p50_reduction=comparison["native_p50_reduction"])
            results.append(result)
            (folder/"load-screen.json").write_text(json.dumps(results,indent=2)+"\n")
            print(json.dumps(result),flush=True)
            if proc.returncode:raise SystemExit(proc.returncode)
