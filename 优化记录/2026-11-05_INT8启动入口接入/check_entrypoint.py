"""Static production command/policy integration checks; device discovery is mocked."""
import contextlib
import copy
import io
import json
import shlex
from pathlib import Path
from unittest.mock import patch
from robot_runtime.setup import config, launcher
from vla_cpp.policy import _config
ROOT=Path("/home/skysi/code/robot_runtime")
LIVE=ROOT/"configs/vla_piper_live.json"
before=LIVE.read_bytes()
cfg=config.load_config(LIVE)
arms={"single":{"interface":"can-mock"}}
cameras={role:{"device":f"/dev/mock-{role}"} for role in config.camera_roles(cfg)}
checks=[]
for variant in ("pruned","full"):
    policy=ROOT/cfg["profiles"][variant]["policy_config"]
    for precision in (None,"baseline","int8-main-smooth"):
        out=io.StringIO()
        args=["--config",str(LIVE),"--profile",variant,"--print-command"]
        if precision:args+=["--precision",precision]
        with patch.object(launcher,"resolve_devices",return_value=(arms,cameras)), \
             patch("robot_runtime.setup.devices.camera_capture_path",side_effect=lambda row,identity:row["device"]), \
             contextlib.redirect_stdout(out):
            assert launcher.main(args)==0
        argv=shlex.split(out.getvalue().strip())
        if precision:
            assert f"--precision={precision}" in argv, argv
        else:
            assert not any(a.startswith("--precision=") for a in argv)
        selected=_config(policy,precision=precision)
        expected="vla.cpp-quant/build-musa-quant" if precision=="int8-main-smooth" else "vla.cpp/build-musa-official"
        assert expected in str(selected["library"]), selected["library"]
        checks.append({"variant":variant,"precision":precision or "default","library":str(selected["library"]),"argv_precision_forwarded":True})
    bad=copy.deepcopy(cfg)
    bad["profiles"][variant]["precision"]="nonexistent-precision"
    try:
        config.validate_assets(ROOT,bad,variant)
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown precision was accepted")
assert LIVE.read_bytes()==before
print(json.dumps({"checks":checks,"live_config_unchanged":True,"device_discovery_mocked":True},indent=2))
