import json,subprocess,sys
from pathlib import Path
folder=Path(__file__).resolve().parent;root=folder.parents[1]
results=[]
for script in ("run_native_v5_regression.py","run_development_reference.py"):
 with (folder/(script.removesuffix(".py")+"-queue.log")).open("w") as log:
  proc=subprocess.run([sys.executable,str(folder/script)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 row=dict(script=script,returncode=proc.returncode);results.append(row)
 (folder/"v5-expansion-queue.json").write_text(json.dumps(results,indent=2)+"\n")
 print(json.dumps(row),flush=True)
 if proc.returncode:raise SystemExit(proc.returncode)
