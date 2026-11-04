"""One sealed heldout evaluation of frozen V6; no robot/camera access."""
import json,os,subprocess,sys,time,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path[:0]=[str(root/"scripts"),str(root/"bindings/python"),"/home/skysi/code/robot_runtime-quant/src"]
from bench_pi05_precision import compare
from robot_runtime.setup.runtime import HardwareLease
runtime="/home/skysi/code/robot_runtime/.venv/bin/python"
framework="/home/skysi/.conda/envs/lerobot_musa312/bin/python"
env=dict(os.environ,PYTHONPATH=str(root/"bindings/python")+":"+str(root/"scripts"),LD_LIBRARY_PATH="/usr/local/musa/lib")
for v in ("pruned","full"):
 dev=json.loads((folder/f"v6-abba-{v}-summary.json").read_text())
 assert dev["offline_accuracy_pass"] and dev["development_performance_pass"] and all(dev["action_repeat_exact"].values()),v
frozen=json.loads((folder/"candidate-v6-freeze.json").read_text())
def verify():
 for name,expected in frozen["hashes"].items():
  if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError("Frozen artifact changed: "+name)
verify()
corpus=json.loads((root/"validation/quant/data/real-v1/manifest.json").read_text())
held=[r for r in corpus["rows"] if r["split"]=="heldout"]
assert len(held)==396
sets={split:{r["episode"] for r in corpus["rows"] if r["split"]==split} for split in ("calibration","development","heldout")}
assert not(sets["heldout"] & (sets["calibration"]|sets["development"]))
seal=folder/"v6-heldout-seal.json"
with seal.open("x") as f:json.dump(dict(candidate_commit=json.loads((folder/"candidate-v6-commit.json").read_text())["commit"],samples=396,episodes=len(sets["heldout"]),source_corpus_sha256=hashlib.sha256((root/"validation/quant/data/real-v1/manifest.json").read_bytes()).hexdigest(),time_unix=time.time()),f,indent=2)
results=[]
def run(name,cmd):
 verify();begin=time.time()
 with (folder/f"{name}.log").open("w") as log:p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 row=dict(name=name,command=cmd,returncode=p.returncode,elapsed_seconds=time.time()-begin)
 results.append(row);(folder/"v6-heldout-commands.json").write_text(json.dumps(results,indent=2)+"\n")
 print(json.dumps(row),flush=True)
 if p.returncode:raise SystemExit(p.returncode)
summaries=[]
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
 for v in ("pruned","full"):
  old=json.loads((folder/f"{v}-bf16-reference-90.json").read_text())
  inputs=root/f"validation/quant/bf16-inputs-{v}-heldout-396"
  run(f"v6-heldout-{v}-freeze",[runtime,"scripts/prepare_pi05_bf16_inputs.py","--config",str(folder/f"{v}-main-native-policy.json"),"--corpus","validation/quant/data/real-v1/manifest.json","--samples","396","--split","heldout","--output-dir",str(inputs)])
  reference=folder/f"{v}-bf16-reference-heldout-396.json"
  run(f"v6-heldout-{v}-bf16",[framework,"scripts/reference_pi05_bf16.py","--checkpoint",old["checkpoint"],"--inputs",str(inputs/"manifest.json"),"--samples","396","--output",str(reference)])
  ref=json.loads(reference.read_text())
  assert ref["source_sha256"]==old["source_sha256"]
  assert ref["num_steps"]==old["num_steps"]
  for precision in ("baseline","int8-main-smooth"):
   run(f"v6-heldout-{v}-{precision}",[runtime,"scripts/bench_pi05_precision.py","--config",str(folder/f"{v}-main-native-policy.json"),"--corpus","validation/quant/data/real-v1/manifest.json","--samples","396","--split","heldout","--warmup","3","--precision",precision,"--output",str(folder/f"v6-heldout-{v}-{precision}.json")])
  comp=compare(reference,folder/f"v6-heldout-{v}-int8-main-smooth.json",folder/f"v6-heldout-{v}-baseline.json")
  (folder/f"v6-heldout-{v}-comparison.json").write_text(json.dumps(comp,indent=2)+"\n")
  baseline_comp=compare(reference,folder/f"v6-heldout-{v}-baseline.json",folder/f"v6-heldout-{v}-baseline.json")
  (folder/f"v6-heldout-{v}-baseline-comparison.json").write_text(json.dumps(baseline_comp,indent=2)+"\n")
  row=dict(variant=v,samples=396,action_vectors=396*50,offline_accuracy_pass=comp["offline_accuracy_pass"],minimum_action_cosine=comp["cosine"]["minimum_action_vector"],native_p50_reduction=comp["native_p50_reduction"],predict_p50_reduction=comp["end_to_end_p50_reduction"],source_checkpoint_sha_matches_development=True)
  summaries.append(row);(folder/"v6-heldout-summary.json").write_text(json.dumps(summaries,indent=2)+"\n");print(json.dumps(row),flush=True)
  if not row["offline_accuracy_pass"]:raise SystemExit("Heldout accuracy failed; do not tune on heldout.")
