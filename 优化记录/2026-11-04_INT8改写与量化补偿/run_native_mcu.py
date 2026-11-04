import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path.insert(0,"/home/skysi/code/robot_runtime-quant/src")
from robot_runtime.setup.runtime import HardwareLease
env=dict(os.environ,LD_LIBRARY_PATH="/usr/local/musa/lib",PYTHONPATH=str(root/"bindings/python")+":"+str(root/"scripts"),
         GGML_CUDA_DISABLE_GRAPHS="1")
out=root/"validation/quant/native-int8-mcu";out.mkdir(exist_ok=False)
results=[]
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
 for precision in ("baseline","int8-expert-smooth"):
    name="v3-mcu-full-"+precision
    cmd=["/opt/moore-threads/moore-perf-compute/1.2.2/target-linux-aarch64/mcu","--dump-mode","normal",
         "--kernel-name","regex:(small_gemm_kernel|packed_norm_kernel|packed_geglu_kernel|row_wmma|int8_native.*pack|vla_r2_begin|vla_r2_end)",
         "-o",str(out/name),sys.executable,str(folder/"profile_native.py"),"full",precision]
    with (folder/f"{name}.log").open("w") as log:proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    raw=(folder/f"{name}.log").read_text()
    row=dict(name=name,command=cmd,returncode=proc.returncode,
             valid_kernel_report="Duration" in raw and "==ERROR==" not in raw and "No kernels were profiled" not in raw)
    results.append(row);(folder/"native-mcu-commands.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(row),flush=True)
    if proc.returncode or not row["valid_kernel_report"]:raise SystemExit(1)
