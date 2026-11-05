import os, subprocess, sys, time
from pathlib import Path
a=Path(__file__).resolve().parent
root=Path("/home/skysi/code/robot_runtime")
env=dict(os.environ,PYTHONPATH=str(root/"src")+":"+str(root/"tests/python"))
for profile in ("full","pruned"):
    for schedule in ("before","candidate"):
        tag=f"{profile}-{schedule}"
        print(f"START {tag}",flush=True)
        start=time.monotonic()
        with (a/f"{tag}.log").open("w") as log:
            proc=subprocess.run([sys.executable,str(a/"check_continuity.py"),"--profile",profile,"--schedule",schedule,"--model","native","--seconds","30","--output",str(a/f"{tag}.json")],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(f"END {tag}: exit={proc.returncode}, seconds={time.monotonic()-start:.1f}",flush=True)
        if proc.returncode:
            print((a/f"{tag}.log").read_text()[-4000:],flush=True)
            raise SystemExit(proc.returncode)
