"""Fit scaling on calibration inputs, then replay native INT8 on development."""
import ctypes,json,os,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path[:0]=[str(root/"scripts"),str(root/"bindings/python"),"/home/skysi/code/robot_runtime-quant/src"]
import gguf
from pi05_gguf import GGUFReader
from calibrate_pi05 import original_names
from int8_compensation import prepare,simulate,error_metrics
from quantize_gguf import file_digest
from robot_runtime.setup.runtime import HardwareLease
variant=sys.argv[1]
manifest_path=folder/f"{variant}-input-manifest.json"
manifest=json.loads(manifest_path.read_text())
assert file_digest(manifest["model"])==manifest["model_sha256"]
reader=GGUFReader(manifest["model"])
tensors={t.name:t for t in reader.tensors}
stats=json.loads((root/f"validation/quant/calibration/{variant}-120/importance.json").read_text())
assert stats["split"]=="calibration" and stats["source_sha256"]==manifest["model_sha256"]
lib=ctypes.CDLL(str(root/"validation/quant/libint8_replay.so"))
ptr=ctypes.c_void_p
lib.int8_replay_create.argtypes=[ptr]*6+[ctypes.c_int]*4;lib.int8_replay_create.restype=ptr
lib.int8_replay_run.argtypes=[ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr]
lib.int8_replay_run.restype=ctypes.c_int
lib.int8_replay_destroy.argtypes=[ptr]
def address(a):return ptr(a.ctypes.data)
def subset(p,indices):
    return dict(p,codes=np.ascontiguousarray(p["codes"][indices]),scales=np.ascontiguousarray(p["scales"][indices]),extra=np.ascontiguousarray(p["extra"][indices]))
def run(handle,x,m,mode,runs=5):
    out=np.empty((len(x),m),np.float32);times=np.empty(runs,np.float32)
    assert lib.int8_replay_run(handle,address(x),len(x),runs,mode,address(out),address(times))==0
    assert np.isfinite(out).all()
    return out,times.tolist()
results=[]
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
 for name in sorted({c["weight"] for obs in manifest["observations"] for c in obs["captures"]}):
    inputs={"calibration":[],"development":[]}
    for obs in manifest["observations"]:
        for cap in obs["captures"]:
            if cap["weight"]!=name:continue
            assert file_digest(cap["file"])==cap["sha256"]
            x=np.fromfile(cap["file"],dtype="<f4",offset=16).reshape(cap["shape"])
            inputs[obs["split"]].append((cap,np.ascontiguousarray(x)))
    source_names=original_names(name)
    w=np.concatenate([gguf.quants.dequantize(tensors[n].data,tensors[n].tensor_type).astype(np.float16).astype(np.float32) for n in source_names],axis=0)
    m,k=w.shape
    assert k==inputs["calibration"][0][1].shape[1]
    amax=np.asarray(stats["weights"][source_names[0]]["amax"],np.float32)
    mean_abs=sum(np.sum(np.abs(x),axis=0,dtype=np.float64) for _,x in inputs["calibration"])/sum(len(x) for _,x in inputs["calibration"])
    mean_abs=mean_abs.astype(np.float32)
    fit_x=np.concatenate([x[np.linspace(0,len(x)-1,min(16,len(x)),dtype=int)] for _,x in inputs["calibration"]])
    output_indices=np.linspace(0,m-1,min(512,m),dtype=int)
    ref=fit_x@w[output_indices].T
    candidates=[dict(method="smoothquant",alpha=None,outlier_count=0)]+[
        dict(method=method,alpha=alpha,outlier_count=0) for method in ("smoothquant","awq") for alpha in (.25,.5,.75)]
    fits=[]
    def build(config):return prepare(w,amax,activation_mean=mean_abs,**config)
    def fit(config):
        p=build(config);metrics=error_metrics(ref,simulate(fit_x,subset(p,output_indices)))
        item=dict(config=config,calibration_metrics=metrics);fits.append(item);return item
    for config in candidates:fit(config)
    best=min(fits,key=lambda r:r["calibration_metrics"]["relative_l2"])["config"]
    for count in (16,32):fit(dict(best,outlier_count=count))
    compensated=min(fits[-2:],key=lambda r:r["calibration_metrics"]["relative_l2"])["config"]
    chosen=[("plain",candidates[0]),("scaled",best),("compensated",compensated)]
    record=dict(variant=variant,weight=name,shape=[m,k],calibration_fit_rows=len(fit_x),calibration_output_channels=len(output_indices),fits=fits,development=[])
    for label,config in chosen:
        p=build(config);original=np.ascontiguousarray(w,dtype=np.float16)
        handle=lib.int8_replay_create(*[address(p[key]) for key in ("codes","scales","inverse","extra","indices")],address(original),k,m,max(len(x) for _,x in inputs["development"]),p["outlier_count"])
        assert handle
        observations=[]
        try:
          for cap,x in inputs["development"]:
            baseline,tbase=run(handle,x,m,1)
            output,times=run(handle,x,m,0)
            mathematical=simulate(x,subset(p,output_indices))
            check=error_metrics(mathematical,output[:,output_indices])
            if check["relative_l2"]>5e-5:raise AssertionError(("native arithmetic mismatch",name,label,check))
            observations.append(dict(input_file=cap["file"],input_sha256=cap["sha256"],occurrence=cap["occurrence"],shape=list(x.shape),native_vs_math=check,vs_f16=error_metrics(baseline,output),native_ms=times,f16_ms=tbase))
        finally:lib.int8_replay_destroy(handle)
        result=dict(label=label,config=config,outlier_indices=p["indices"].tolist(),observations=observations,
          native_p50_ms=float(np.median([v for o in observations for v in o["native_ms"]])),
          f16_p50_ms=float(np.median([v for o in observations for v in o["f16_ms"]])),
          minimum_token_cosine=min(o["vs_f16"]["minimum_token_cosine"] for o in observations),
          mean_relative_l2=float(np.mean([o["vs_f16"]["relative_l2"] for o in observations])))
        record["development"].append(result)
        print(json.dumps(dict(variant=variant,weight=name,**{key:result[key] for key in ("label","config","native_p50_ms","f16_p50_ms","minimum_token_cosine","mean_relative_l2")})),flush=True)
        (folder/f"{variant}-compensation-partial.json").write_text(json.dumps(results+[record],indent=2)+"\n")
    results.append(record)
    (folder/f"{variant}-compensation.json").write_text(json.dumps(dict(variant=variant,source_sha256=manifest["model_sha256"],capture_manifest_sha256=file_digest(manifest_path),scope="operator output accuracy, not BF16 policy-action acceptance",weights=results),indent=2)+"\n")
