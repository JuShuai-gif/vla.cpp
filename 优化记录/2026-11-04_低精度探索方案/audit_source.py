"""Read-only quantization scope/capability audit; no model conversion or GPU work."""
from pathlib import Path
import collections, hashlib, json, subprocess, sys
import numpy as np
import gguf

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/"bindings/python"), str(REPO/"scripts")]
from vla_cpp.pi05_gguf import GGUFReader
import quantize_gguf as converter

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    report = {"scope": "Static metadata and tiny synthetic CPU pack tests only; not model accuracy/performance validation",
              "code_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
              "dependency_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO/"build-musa-deps/llama",text=True).strip(),
              "source_sha256": {}, "writer_checks": {}, "models": {}}
    paths = ["scripts/quantize_gguf.py","src/loader.cpp","src/models/pi05.cpp","src/modules/gemma_expert.h",
             "src/act_dtype.h","bindings/python/vla_cpp/policy.py","scripts/pi05_musa_profiles.json",
             "build-musa-deps/llama/ggml/src/ggml-cuda/mmq.cu",
             "build-musa-deps/llama/ggml/src/ggml-cuda/common.cuh",
             "build-musa-deps/llama/ggml/src/ggml-cuda/ggml-cuda.cu"]
    report["source_sha256"]={p:sha(REPO/p) for p in paths}
    for name in converter.QK:
        kind=getattr(gguf.GGMLQuantizationType,name)
        try:
            a=np.linspace(-1,1,32,dtype=np.float32).reshape(1,32)
            packed=gguf.quants.quantize(a,kind)
            restored=gguf.quants.dequantize(packed,kind)
            assert restored.shape==a.shape and np.isfinite(restored).all()
            report["writer_checks"][name]={"pack_unpack":True,"block_values":32,"block_bytes":packed.nbytes}
        except Exception as exc:
            report["writer_checks"][name]={"pack_unpack":False,"error":repr(exc)}
    for variant,folder in [("pruned","pi05_cube_merge_cksize50_step3_0929"),("full","cup_cube_joint_061_a800_50k_20261022")]:
        path=Path("/home/skysi/VLA/models")/folder/"gguf/model.gguf"
        reader=GGUFReader(str(path))
        selected=[t for t in reader.tensors if converter.eligible(t.name,t.shape,"Q8_0",converter.SKIP+converter.SKIP_VISION)
                  and t.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.BF16)]
        prefix=[t for t in selected if t.name.startswith("vlm.blk.")]
        mlp=[t for t in prefix if ".ffn_" in t.name]
        total=sum(t.data.nbytes for t in reader.tensors)
        plans={}
        for label,tensors in [("current_default",selected),("strict_prefix",prefix),("prefix_mlp",mlp)]:
            original=sum(t.data.nbytes for t in tensors)
            sizes={}
            for kind in ("Q8_0","Q4_0"):
                block,packed_bytes=gguf.GGML_QUANT_SIZES[getattr(gguf.GGMLQuantizationType,kind)]
                size=total-original+sum(int(np.prod(t.shape))//block*packed_bytes for t in tensors)
                sizes[kind]={"estimated_total_tensor_bytes":size,"tensor_payload_reduction":1-size/total}
            plans[label]={"tensor_count":len(tensors),"source_tensor_bytes":original,"estimates":sizes}
        report["models"][variant]={"model":str(path),"file_bytes":path.stat().st_size,
            "model_sha256":None,"hash_note":"Full model SHA256 must be recorded before experiments; this audit reads metadata only",
            "total_tensor_bytes":total,"tensor_count":len(reader.tensors),"plans":plans,
            "unexpected_non_prefix":[t.name for t in selected if not t.name.startswith("vlm.blk.")],
            "default_selected_tensors":[{"name":t.name,"ggml_shape":list(map(int,t.shape)),"source_type":t.tensor_type.name} for t in selected]}
    destination=Path(__file__).with_name("source-audit.json")
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:{"plans":v["plans"],"unexpected_non_prefix":v["unexpected_non_prefix"]} for k,v in report["models"].items()},indent=2))
    print("Saved",destination)

if __name__=="__main__":main()
