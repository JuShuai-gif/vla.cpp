"""Replay saved RGB/state through the runtime bridge; never open robot devices."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from robot_runtime.backends import load_backend
from robot_runtime.replay import observation
from robot_runtime.setup import config, runs
from robot_runtime.setup.runtime import HardwareLease

parser = argparse.ArgumentParser()
parser.add_argument("profile", choices=("full", "pruned"))
args = parser.parse_args()
root = Path("/home/skysi/code/robot_runtime")
folder = Path(__file__).resolve().parent
cfg = config.load_config(root / "configs/vla_piper_live.json")
params = cfg["profiles"][args.profile]
source = root / f"validation/vla_cpp/{args.profile}-observation.npz"
images, state = observation(source)
rows = []
with HardwareLease(root, "model-replay"):
    model = load_backend(cfg["policy_backend"]).create_replay_model(root, cfg, args.profile)
    try:
        inputs = [images[key] for key in model.mapping.camera_order]
        model.warmup(inputs, state=state, task=params["task"], runs=3)
        for number in range(1, 4):
            start = time.monotonic()
            actions = model.predict(inputs, state=state, task=params["task"])
            end = time.monotonic()
            timing = dict(model.last_predict_timings_ms)
            assert actions.shape == (50, 7) and np.isfinite(actions).all()
            assert all(np.isfinite(timing[key]) and timing[key] >= 0 for key in
                       ("native_call_ms", "preprocess_ms", "postprocess_ms"))
            elapsed = (end - start) * 1000
            assert elapsed >= sum(timing.values()) - .01
            runs.prediction_metrics({}, timing, capture_start=start, capture_end=start,
                                    predict_ms=elapsed, feedback_timestamp=time.time(),
                                    completed_at=end, console_label=f"vla_cpp predict #{number}")
            rows.append(dict(prediction=number, predict_ms=elapsed, **timing))
        report = dict(profile=args.profile, scope="saved observations; no camera or robot I/O",
                      warmup=3, samples=3, platform=platform.platform(),
                      observation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      resolved=model.resolved_parameters, action_shape=list(actions.shape), timings=rows)
    finally:
        model.close()
(folder / f"{args.profile}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"profile": args.profile, "mean_native_call_ms": np.mean([r["native_call_ms"] for r in rows]),
                  "mean_predict_ms": np.mean([r["predict_ms"] for r in rows])}))
