import ctypes,hashlib,json,sys
from pathlib import Path
import numpy as np
from vla_cpp.policy import load_policy
folder=Path(__file__).resolve().parent;root=folder.parents[1]
variant,precision=sys.argv[1:3]
corpus=root/"validation/quant/data/real-v1/manifest.json"
row=next(r for r in json.loads(corpus.read_text())["rows"] if r["split"]=="development")
path=corpus.parent/row["file"];assert hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"]
with np.load(path,allow_pickle=False) as data:
 images={f"observation.images.{k}":np.array(data[k]) for k in ("top","wrist")}
 state=np.array(data["state"])
marker=ctypes.CDLL("/home/skysi/VLA/four-deep-20261103/profile_markers.so")
marker.r2_begin.restype=ctypes.c_int;marker.r2_end.restype=ctypes.c_int
with load_policy(folder/f"{variant}-mcu-policy.json",precision=precision) as model:
 import os
 assert os.environ.get("GGML_CUDA_DISABLE_GRAPHS")=="1"
 model.warmup(images,state=state,task=row["task"],runs=1)
 assert marker.r2_begin()==0
 actions=model.predict(images,state=state,task=row["task"])
 assert marker.r2_end()==0
 assert np.isfinite(actions).all()
 print(json.dumps(dict(variant=variant,precision=precision,input_sha256=row["sha256"],
                      scope="Graph OFF single warmed observation; profiler-perturbed diagnostic only")),flush=True)
