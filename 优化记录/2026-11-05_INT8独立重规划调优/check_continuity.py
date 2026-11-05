"""Actual inference-loop/queue test with saved images, memory SDK and no hardware."""
import argparse
import contextlib
import json
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import test_rknn_diagnostics_integration as helpers
from robot_runtime.setup import runs
from robot_runtime.setup.runtime import HardwareLease
from robot_runtime.vla_cpp_backend import VlaCppBackend

ROOT=Path("/home/skysi/code/robot_runtime")
ARCHIVE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=("full","pruned"),required=True)
    parser.add_argument("--schedule",choices=("before","candidate"),required=True)
    parser.add_argument("--model",choices=("native","budget"),default="native")
    parser.add_argument("--seconds",type=float,default=30)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    backup=Path(json.loads((ARCHIVE/"before-state.json").read_text())["backup"])
    file=(ARCHIVE/"baseline-live.json" if args.schedule=="before" else ARCHIVE/"candidate-live.json")
    config=json.loads(file.read_text())
    params=dict(config["profiles"][args.profile])
    assert params.get("calibration_file") is None
    assert config["robot_layout"]=="single" and config["policy_backend"]=="vla_cpp"
    with np.load(ROOT/f"validation/vla_cpp/{args.profile}-observation.npz",allow_pickle=False) as data:
        state=data["state"].copy()
        images={key:data[key].copy() for key in data.files if key!="state"}
    budget=json.loads((ARCHIVE/"parameter-rationale.json").read_text())[args.profile]["budget_ms"]/1000
    ns=helpers.runtime_namespace()
    sends=[];merges=[];events=[];predictions=[];empty=[]
    class Camera:
        config=SimpleNamespace(color_mode="rgb")
        def __init__(self,role):self.role=role
        def connect(self):pass
        def disconnect(self):pass
        def async_read(self):return images[self.role].copy()
    class SDK(helpers.FakeSDK):
        def JointCtrl(self,*values):
            time.sleep(.004)  # simulate SDK submission latency, without CAN
            sends.append(time.monotonic())
            super().JointCtrl(*values)
    class BudgetModel:
        chunk_size=50
        mapping=SimpleNamespace(camera_order=("top","wrist"))
        last_predict_timings_ms={}
        resolved_parameters={}
        def predict(self,*args,**kwargs):
            time.sleep(budget)
            return np.tile(state,(50,1))
        def close(self):pass
    queue_type=ns["ActionQueue"]
    class MeasuredQueue(queue_type):
        def merge(self,*pos,**kw):
            pending=self.qsize()
            result=super().merge(*pos,**kw)
            merges.append({"at":time.monotonic(),"pending_before":pending,"pending_after":self.qsize(),"delay_steps":pos[2]})
            return result
        def get(self):
            result=super().get()
            if result is None and sends:empty.append(time.monotonic())
            return result
    plugin=VlaCppBackend()
    original_factory=plugin.create_model
    def factory(cfg):
        model=original_factory(cfg) if args.model=="native" else BudgetModel()
        predict=model.predict
        def measured(*a,**kw):
            start=time.monotonic()
            result=predict(*a,**kw)
            predictions.append((time.monotonic()-start)*1000)
            return result
        model.predict=measured
        return model
    plugin.create_model=factory
    sdk=SDK()
    ns.update(mp=helpers.FakeMP(),load_backend=lambda name:plugin,ActionQueue=MeasuredQueue,
              make_cameras_from_configs=lambda cfg:{role:Camera(role) for role in images},
              busy_wait=lambda duration:time.sleep(max(0,duration)))
    def read_state(*args,return_timestamp=False,**kwargs):
        return (state.copy(),time.time()) if return_timestamp else state.copy()
    ns["_read_state"]=read_state
    params.update(duration=args.seconds,precision="int8-main-smooth",diagnostic_dir=str(args.output.with_suffix("")),camera_save_interval_s=0,action_log_interval=0,
                  warmup_steps=3 if args.model=="native" else 1,execute_robot=True,dry_run=False)
    cfg=ns["InferenceConfig"](policy_backend="vla_cpp",
        robot=SimpleNamespace(type="single_piper",ctrl_mode="joint",use_degrees=True,
                              port="MEMORY-SDK-NO-CAN",cameras={role:None for role in images}),**params)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with HardwareLease(ROOT,"offline-continuity-mock-devices"), \
         patch.dict(sys.modules,{"piper_sdk":SimpleNamespace(C_PiperInterface=lambda *a,**kw:sdk)}), \
         patch("socket.socket",side_effect=AssertionError("Hardware/network sockets forbidden")), \
         patch.object(runs,"stopping",return_value=False), \
         patch.object(runs,"event",side_effect=lambda kind,**data:events.append((kind,data))):
        ns["inference"](cfg)
    intervals=np.diff(sends)*1000
    assert len(intervals)>50 and len(merges)>=3,(len(intervals),len(merges))
    def dist(values):
        values=np.asarray(values)
        return {"count":len(values),"p50":float(np.percentile(values,50)),"p99":float(np.percentile(values,99)),"max":float(values.max())}
    report={"profile":args.profile,"schedule":args.schedule,"model":args.model,
            "scope":"Actual production inference/dispatch function and queue; saved static observations; memory SDK and thread-based fake I/O process; real GPU only in native mode. No real cameras/CAN/robot.",
            "parameters":params,"dispatch_interval_ms":dist(intervals),"prediction_ms":dist(predictions),
            "queue_empty_after_start":len(empty),"merges":merges,
            "plan_interval_ms":dist(np.diff([m["at"] for m in merges])*1000),
            "diagnostic_events":[(k,v) for k,v in events if "diagnostic" in k],
            "steady_plan_interval_ms":dist(np.diff([m["at"] for m in merges])[1:]*1000),
            "gaps_over_two_periods":int(np.sum(intervals>2*cfg.execution_period*1000)),
            "gaps_over_100ms":int(np.sum(intervals>100)),
            "physical_motion_verified":False,"sdk_calls_are_simulated":True}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("parameters","merges","scope")},ensure_ascii=False))
    if params.get("diagnostics"):
        summary=report["diagnostic_events"][-1][1]["summary"]
        assert summary["written"]>0 and summary["dropped"]==0 and summary["error"] is None
        assert not summary["writer_alive"] and not summary["truncated"]
    assert not empty, "Queue starvation"
    assert report["gaps_over_two_periods"]==0, "Dispatch cadence exceeded two periods"
    assert min(row["pending_before"] for row in merges[1:])>=5, "Blend overlap budget was not retained"

if __name__=="__main__":main()
