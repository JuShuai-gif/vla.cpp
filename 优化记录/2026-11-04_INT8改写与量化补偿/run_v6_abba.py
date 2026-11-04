"""Frozen V6: two fresh-process pairs in A-B-B-A order on all development observations."""
import json,os,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];folder=Path(__file__).resolve().parent
sys.path[:0]=[str(root/"scripts"),str(root/"bindings/python"),"/home/skysi/code/robot_runtime-quant/src"]
from bench_pi05_precision import compare
from robot_runtime.setup.runtime import HardwareLease
env=dict(os.environ,PYTHONPATH=str(root/"scripts")+":"+str(root/"bindings/python"),LD_LIBRARY_PATH="/usr/local/musa/lib")
results=[]
def episode_bootstrap(before,after,episodes):
    labels=list(dict.fromkeys(episodes));indexes=[np.flatnonzero(np.asarray(episodes)==e) for e in labels]
    rng=np.random.default_rng(173);ratios=[]
    for _ in range(5000):
        ids=np.concatenate([indexes[i]for i in rng.integers(0,len(indexes),len(indexes))])
        ratios.append(1-float(after[:,ids].mean())/float(before[:,ids].mean()))
    return dict(mean_reduction=1-float(after.mean())/float(before.mean()),
                episode_bootstrap_95_percent_interval=np.quantile(ratios,[.025,.975]).tolist(),
                episodes=len(labels),resamples=5000,
                scope="paired observations clustered by episode; two opposite-order process pairs")
with HardwareLease(Path("/home/skysi/code/robot_runtime"),"model-replay"):
 for variant in ("pruned","full"):
    reports={}
    for precision,pair in (("baseline",0),("int8-main-smooth",0),("int8-main-smooth",1),("baseline",1)):
        output=folder/f"v6-abba-{variant}-{precision}-{pair}.json"
        cmd=[sys.executable,"scripts/bench_pi05_precision.py","--config",str(folder/f"{variant}-main-native-policy.json"),
             "--corpus","validation/quant/data/real-v1/manifest.json","--samples","90","--warmup","3",
             "--precision",precision,"--output",str(output)]
        frozen=json.loads((folder/"candidate-v6-freeze.json").read_text())
        for name,expected in frozen["hashes"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError("Frozen artifact changed: "+name)
        begin=time.time()
        with output.with_suffix(".log").open("w") as log:proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        row=dict(variant=variant,precision=precision,pair=pair,command=cmd,returncode=proc.returncode,elapsed_seconds=time.time()-begin)
        if proc.returncode==0:
            report=json.loads(output.read_text());reports[(precision,pair)]=(output,report)
            row.update(native_ms=report["native_ms"],predict_ms=report["predict_ms"])
        results.append(row);(folder/"v6-abba-commands.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps(row),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)
    comparisons=[]
    for pair in (0,1):
        baseline,_=reports[("baseline",pair)];candidate,_=reports[("int8-main-smooth",pair)]
        comp=compare(folder/f"{variant}-bf16-reference-90.json",candidate,baseline)
        (folder/f"v6-abba-{variant}-comparison-{pair}.json").write_text(json.dumps(comp,indent=2)+"\n")
        comparisons.append(comp)
    exact={}
    for precision in ("baseline","int8-main-smooth"):
        x=np.load(reports[(precision,0)][0].with_suffix(".actions.npz"),allow_pickle=False)["actions"]
        y=np.load(reports[(precision,1)][0].with_suffix(".actions.npz"),allow_pickle=False)["actions"]
        exact[precision]=bool(np.array_equal(x,y))
    stats={}
    episodes=reports[("baseline",0)][1]["episodes"]
    for key in ("native","predict"):
        before=np.array([reports[("baseline",i)][1][f"raw_{key}_ms"]for i in (0,1)])
        after=np.array([reports[("int8-main-smooth",i)][1][f"raw_{key}_ms"]for i in (0,1)])
        assert before.shape==after.shape==(2,90) and np.isfinite(before).all() and np.isfinite(after).all()
        stats[key]=episode_bootstrap(before,after,episodes)
        stats[key]["pooled_p50_reduction"]=1-float(np.median(after))/float(np.median(before))
    report=dict(variant=variant,unique_observations=90,pairs=2,order="A0 B0 B1 A1",action_repeat_exact=exact,
                offline_accuracy_pass=all(x["offline_accuracy_pass"] for x in comparisons),
                minimum_action_cosine=min(x["cosine"]["minimum_action_vector"]for x in comparisons),
                pair_native_p50_reductions=[x["native_p50_reduction"] for x in comparisons],
                pair_predict_p50_reductions=[x["end_to_end_p50_reduction"]for x in comparisons],stats=stats,
                criteria="Each reverse-order pair has positive native/predict P50 gain; episode-bootstrap lower bound for mean gains >0. This is the current improvement goal, not the historical 20% target.")
    report["development_performance_pass"]=all(v>0 for v in report["pair_native_p50_reductions"]+report["pair_predict_p50_reductions"]) and all(x["episode_bootstrap_95_percent_interval"][0]>0 for x in stats.values())
    (folder/f"v6-abba-{variant}-summary.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
