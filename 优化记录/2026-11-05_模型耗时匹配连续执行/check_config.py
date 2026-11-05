"""Check exactly the requested profiles and actual launcher forwarding; no discovery."""
from pathlib import Path
import contextlib
import io
import json
import shlex
from unittest.mock import patch
from robot_runtime.setup import config,launcher
ROOT=Path("/home/skysi/code/robot_runtime")
ARCHIVE=Path(__file__).resolve().parent
backup=Path(json.loads((ARCHIVE/"before-state.json").read_text())["backup"])
old=json.loads((backup/"configs/vla_piper_live.json").read_text())
new=json.loads((ARCHIVE/"candidate-live.json").read_text())
changes=json.loads((ARCHIVE/"parameter-rationale.json").read_text())
restored=json.loads(json.dumps(new))
for variant in ("pruned","full"):
    for key,value in changes[variant]["settings"].items():
        assert new["profiles"][variant][key]==value
        restored["profiles"][variant][key]=old["profiles"][variant][key]
assert restored==old, "Unrelated configuration changed"
assert json.loads((ROOT/"configs/vla_piper_live.json").read_text())==new, "Candidate not deployed"
config.validate_config(new,require_bound=True)
arms={"single":{"interface":"memory-can"}}
cameras={role:{"device":f"/dev/memory-{role}"} for role in config.camera_roles(new)}
rows=[]
for variant in ("pruned","full"):
    config.validate_assets(ROOT,new,variant)
    out=io.StringIO()
    with patch.object(launcher,"resolve_devices",return_value=(arms,cameras)), \
         patch("robot_runtime.setup.devices.camera_capture_path",side_effect=lambda row,identity:row["device"]), \
         contextlib.redirect_stdout(out):
        assert launcher.main(["--config",str(ROOT/"configs/vla_piper_live.json"),"--profile",variant,"--print-command"])==0
    argv=shlex.split(out.getvalue().strip())
    for key,value in changes[variant]["settings"].items():
        text=json.dumps(value) if isinstance(value,bool) else str(value)
        assert f"--{key}={text}" in argv,(key,argv)
    assert not any(a.startswith("--precision=") for a in argv)
    rows.append({"profile":variant,"settings_forwarded":changes[variant]["settings"],"precision_unchanged":True})
print(json.dumps({"profiles":rows,"unrelated_configuration_preserved":True,"no_hardware_discovery":True},indent=2))
