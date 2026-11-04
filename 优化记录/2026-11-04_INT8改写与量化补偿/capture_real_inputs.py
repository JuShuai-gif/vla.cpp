"""Capture selected real GEMM inputs; diagnostic, never benchmark timings."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path[:0]=[str(root/"bindings/python"),str(root/"scripts"),"/home/skysi/code/robot_runtime-quant/src"]
from vla_cpp import load_policy
from robot_runtime.setup.runtime import HardwareLease
from quantize_gguf import file_digest
variant=sys.argv[1]
corpus_path=root/"validation/quant/data/real-v1/manifest.json"
corpus=json.loads(corpus_path.read_text())
output=root/f"validation/quant/int8-compensation-inputs/{variant}"
output.mkdir(parents=True,exist_ok=False);raw=output/"raw";raw.mkdir()
cfg=json.loads((root/f"优化记录/2026-11-04_INT8恢复与分块验证/{variant}-load-policy.json").read_text())
cfg.pop("precision_profiles",None);cfg["default_precision"]="baseline"
profiles=json.loads(Path(cfg["optimization_profile"]).read_text())
selected=["vlm.blk.0.ffn_gateup.weight","vlm.blk.9.ffn_gateup.weight","vlm.blk.17.ffn_down.weight","aex.blk.0.ffn_gateup.weight"]
for values in profiles.values():
    values["VLA_PI05_CALIBRATION_DIR"]=str(raw)
    values["VLA_PI05_CALIBRATION_INPUTS"]=",".join(selected)
(output/"optimization.json").write_text(json.dumps(profiles,indent=2)+"\n")
cfg["optimization_profile"]=str(output/"optimization.json")
(output/"policy.json").write_text(json.dumps(cfg,indent=2)+"\n")
records=[]
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
  with load_policy(output/"policy.json") as policy:
    for split in ("calibration","development"):
      for row in [r for r in corpus["rows"] if r["split"]==split][:4]:
        old=set(raw.glob("*.jsonl"));path=corpus_path.parent/row["file"]
        assert file_digest(path)==row["sha256"]
        with np.load(path,allow_pickle=False) as d:
          policy.predict({f"observation.images.{key}":np.array(d[key]) for key in ("top","wrist")},state=np.array(d["state"]),task=row["task"])
        captures=[]
        for trace in sorted(set(raw.glob("*.jsonl"))-old):
          for line in trace.read_text().splitlines():
            stat=json.loads(line)
            if "input_file" in stat:
              p=Path(stat["input_file"])
              shape=np.fromfile(p,dtype="<u8",count=2)
              data=np.fromfile(p,dtype="<f4",offset=16).reshape(shape)
              assert data.shape==(stat["rows"],stat["channels"]) and np.isfinite(data).all()
              captures.append(dict(weight=stat["weight"],occurrence=stat["occurrence"],file=str(p),sha256=file_digest(p),shape=shape.tolist()))
        assert captures
        records.append(dict(split=split,observation=row["file"],episode=row["episode"],sha256=row["sha256"],captures=captures))
        (output/"manifest.json").write_text(json.dumps(dict(variant=variant,model=cfg["model"],model_sha256=file_digest(cfg["model"]) if len(records)==1 else model_sha,selected=selected,observations=records),indent=2)+"\n")
        if len(records)==1:model_sha=file_digest(cfg["model"])
        print(json.dumps(dict(variant=variant,split=split,observations=len(records),captures=len(captures))),flush=True)
(folder/f"{variant}-input-manifest.json").write_bytes((output/"manifest.json").read_bytes())
