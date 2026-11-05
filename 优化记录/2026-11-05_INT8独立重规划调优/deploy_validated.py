"""Deploy only the independent INT8 config after checking archived test results."""
import hashlib, json, os
from pathlib import Path
ROOT=Path("/home/skysi/code/robot_runtime")
A=Path(__file__).resolve().parent
summary={}
for profile in ("full","pruned"):
    before=json.loads((A/f"{profile}-before.json").read_text())
    candidate=json.loads((A/f"{profile}-candidate.json").read_text())
    budget=json.loads((A/f"{profile}-budget.json").read_text())
    for report in (before,candidate,budget):
        assert report["queue_empty_after_start"]==0
        assert report["gaps_over_two_periods"]==0
        assert min(m["pending_before"] for m in report["merges"][1:])>=5
        assert report["sdk_calls_are_simulated"] and not report["physical_motion_verified"]
    for report in (candidate,budget):
        diag=report["diagnostic_events"][-1][1]["summary"]
        assert diag["written"]>0 and diag["dropped"]==0 and diag["error"] is None
        assert not diag["writer_alive"] and not diag["truncated"]
    old=before["steady_plan_interval_ms"]["p50"]
    new=candidate["steady_plan_interval_ms"]["p50"]
    assert new<old
    summary[profile]={
        "old_plan_interval_p50_ms":old,
        "new_plan_interval_p50_ms":new,
        "plan_interval_reduction_pct":100*(1-new/old),
        "old_predict_p50_ms":before["prediction_ms"]["p50"],
        "new_predict_p50_ms":candidate["prediction_ms"]["p50"],
        "new_dispatch_interval_ms":candidate["dispatch_interval_ms"],
        "native_min_overlap_steps":min(m["pending_before"] for m in candidate["merges"][1:]),
        "budget_min_overlap_steps":min(m["pending_before"] for m in budget["merges"][1:]),
        "new_steady_plan_intervals":candidate["steady_plan_interval_ms"]["count"],
        "queue_empty_after_start":0,
    }
manifest=json.loads((A/"before-state.json").read_text())
for file,expected in manifest["protected_files"].items():
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected,file
destination=ROOT/"configs/vla_piper_live_int8.json"
assert not destination.exists(), "Refusing to overwrite an existing independent config"
data=(A/"candidate-live.json").read_bytes()
with destination.open("xb") as target:
    target.write(data)
os.chmod(destination,(ROOT/"configs/vla_piper_live.json").stat().st_mode & 0o777)
assert destination.read_bytes()==data
summary["deployment"]={
    "config":str(destination),
    "sha256":hashlib.sha256(data).hexdigest(),
    "protected_files_unchanged":len(manifest["protected_files"]),
    "physical_motion_verified":False,
}
(A/"acceptance.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False,indent=2))
