"""Verify INT8 isolation and actual launcher arguments with hardware discovery mocked."""
from pathlib import Path
import contextlib, hashlib, io, json, shlex
from unittest.mock import patch
from robot_runtime.setup import config, launcher
ROOT=Path("/home/skysi/code/robot_runtime")
ARCHIVE=Path(__file__).resolve().parent
old=json.loads((ARCHIVE/"baseline-live.json").read_text())
new=json.loads((ARCHIVE/"candidate-live.json").read_text())
changes=json.loads((ARCHIVE/"parameter-rationale.json").read_text())
restored=json.loads(json.dumps(new))
for variant in ("full","pruned"):
    assert new["profiles"][variant]["precision"]=="int8-main-smooth"
    assert new["profiles"][variant]["diagnostics"] is True
    assert new["profiles"][variant]["replan_threshold"]==changes[variant]["new_threshold"]
    for key in ("precision","diagnostics","replan_threshold"):
        if key in old["profiles"][variant]:
            restored["profiles"][variant][key]=old["profiles"][variant][key]
        else:
            restored["profiles"][variant].pop(key)
restored["last_checks"]=old["last_checks"]
assert restored==old, "Unrelated config changed"
config.validate_config(new,require_bound=True)
arms={"single":{"interface":"memory-can"}}
cameras={role:{"device":f"/dev/memory-{role}"} for role in config.camera_roles(new)}
def command(file,variant,precision=None):
    out=io.StringIO()
    args=["--config",str(file),"--profile",variant,"--print-command"]
    if precision:args+=["--precision",precision]
    with patch.object(launcher,"resolve_devices",return_value=(arms,cameras)), \
         patch("robot_runtime.setup.devices.camera_capture_path",side_effect=lambda row,identity:row["device"]), \
         contextlib.redirect_stdout(out):
        assert launcher.main(args)==0
    return shlex.split(out.getvalue().strip())
rows=[]
for variant in ("full","pruned"):
    config.validate_assets(ROOT,new,variant)
    for precision in (None,"baseline"):
        before=command(ARCHIVE/"baseline-live.json",variant,precision)
        after=command(ROOT/"configs/vla_piper_live.json",variant,precision)
        assert before==after, "Original command changed"
    candidate=command(ARCHIVE/"candidate-live.json",variant)
    assert candidate==command(ARCHIVE/"candidate-live.json",variant,"int8-main-smooth")
    deployed=ROOT/"configs/vla_piper_live_int8.json"
    if deployed.exists():
        assert deployed.read_bytes()==(ARCHIVE/"candidate-live.json").read_bytes()
        assert candidate==command(deployed,variant,"int8-main-smooth")
    for key in ("precision","diagnostics","replan_threshold"):
        value=new["profiles"][variant][key]
        text=json.dumps(value) if isinstance(value,bool) else str(value)
        assert f"--{key}={text}" in candidate,(key,candidate)
    rows.append({"profile":variant,"threshold":changes[variant]["new_threshold"],"baseline_command_unchanged":True,"int8_default_and_explicit_match":True})
manifest=json.loads((ARCHIVE/"before-state.json").read_text())
for file,expected in manifest["protected_files"].items():
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected,file
report={"profiles":rows,"protected_files_unchanged":len(manifest["protected_files"]),"unrelated_configuration_preserved":True,"hardware_discovery_mocked":True}
(ARCHIVE/"config-validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(report,ensure_ascii=False))
