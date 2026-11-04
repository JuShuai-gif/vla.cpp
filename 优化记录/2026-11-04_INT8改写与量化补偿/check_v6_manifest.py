import json,sys,shutil,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path[:0]=[str(root/"bindings/python"),str(root/"scripts")]
from vla_cpp import inspect_policy
from pi05_gguf import GGUFReader
results=[]
for v in ("pruned","full"):
 original=folder/(v+"-main-native-policy.json")
 cfg=json.loads(original.read_text())
 manifest=Path(cfg["precision_profiles"]["int8-main-smooth"]["int8_parameters"])
 m=json.loads(manifest.read_text())
 spec=inspect_policy(original,precision="int8-main-smooth")
 assert len(m["layers"])==35
 reader=GGUFReader(cfg["model"])
 total=sum(int(t.n_elements) for t in reader.tensors if t.name.endswith(".weight"))
 quantized=sum(row["shape"][0]*row["shape"][1] for row in m["layers"].values())
 row=dict(variant=v,layers=len(m["layers"]),weight_elements=total,int8_weight_elements=quantized,fraction=quantized/total,invalid_cases={})
 with tempfile.TemporaryDirectory(prefix="int8-manifest-check-",dir=root/"validation/quant") as temp:
  tmp=Path(temp)
  for p in manifest.parent.glob("*.i8p"):shutil.copy2(p,tmp/p.name)
  cfg["precision_profiles"]["int8-main-smooth"]["int8_parameters"]=str(tmp/"manifest.json")
  config=tmp/"policy.json";config.write_text(json.dumps(cfg))
  for case in ("wrong_mode","missing_prefix","unused_layer","corrupt_hash"):
   bad=json.loads(json.dumps(m))
   if case=="wrong_mode":bad["mode"]="expert_gateup"
   if case=="missing_prefix":bad["layers"].pop("vlm.blk.0.ffn_gateup")
   if case=="unused_layer":bad["layers"]["vlm.blk.17.ffn_gateup"]=bad["layers"]["vlm.blk.0.ffn_gateup"]
   if case=="corrupt_hash":bad["layers"]["vlm.blk.0.ffn_gateup"]["sha256"]="0"*64
   (tmp/"manifest.json").write_text(json.dumps(bad))
   try:inspect_policy(config,precision="int8-main-smooth")
   except ValueError as e:row["invalid_cases"][case]=str(e)
   else:raise AssertionError(case+" was accepted")
 results.append(row)
(folder/"native-v6-manifest-check.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps(results),flush=True)
