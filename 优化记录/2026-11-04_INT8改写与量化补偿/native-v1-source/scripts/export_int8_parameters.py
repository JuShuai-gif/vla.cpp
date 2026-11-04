"""Export calibration-bound native INT8 expert GateUp parameters, never weights."""
import argparse,json,struct
from pathlib import Path
import gguf
import numpy as np
from pi05_gguf import GGUFReader
from int8_compensation import prepare
from quantize_gguf import file_digest

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",type=Path,required=True);ap.add_argument("--importance",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--alpha",type=float)
    ap.add_argument("--outliers",type=int,default=0)
    args=ap.parse_args();cfg=json.loads(args.config.read_text())
    model=(args.config.resolve().parent/Path(cfg["model"]).expanduser()).resolve()
    calibration=json.loads(args.importance.read_text());source_sha=file_digest(model)
    if calibration.get("split")!="calibration" or calibration.get("source_sha256")!=source_sha:
        raise ValueError("calibration must belong to the selected original checkpoint")
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    reader=GGUFReader(model);tensors={t.name:t for t in reader.tensors}
    manifest=dict(schema_version=1,mode="expert_gateup",split="calibration",source_sha256=source_sha,
       calibration_sha256=file_digest(args.importance),calibration_path=str(args.importance.resolve()),
       scaling="plain" if args.alpha is None else "smoothquant",alpha=args.alpha,outlier_count=args.outliers,layers={})
    for i in range(reader.fields["pi05.n_layers"].contents()):
        name=f"aex.blk.{i}.ffn_gateup"
        sources=[f"aex.blk.{i}.ffn_{part}.weight" for part in ("gate","up")]
        w=np.concatenate([gguf.quants.dequantize(tensors[n].data,tensors[n].tensor_type).astype(np.float16).astype(np.float32) for n in sources])
        a=np.asarray(calibration["weights"][sources[0]]["amax"],np.float32)
        p=prepare(w,a,args.alpha,args.outliers)
        data=struct.pack("<III",0x31503849,w.shape[1],args.outliers)+p["inverse"].astype("<f4").tobytes()+p["indices"].astype("<i4").tobytes()
        file=root/(name+".i8p");file.write_bytes(data)
        manifest["layers"][name]=dict(sha256=file_digest(file),shape=list(w.shape),outlier_indices=p["indices"].tolist())
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(directory=str(root),layers=len(manifest["layers"]),alpha=args.alpha,outliers=args.outliers)),flush=True)
if __name__=="__main__":main()
