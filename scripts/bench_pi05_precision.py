#!/usr/bin/env python3
"""Resident public-API benchmark with paired RGB/state/prompt/noise and full actions.

Run one precision per fresh process. Outputs are screening evidence unless the
caller completes independent heldout ABBA batches and physical/closed-loop gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from pi05_gguf import GGUFReader  # installs the repository binding path
from vla_cpp import load_policy
from vla_cpp.policy import prepare_prompt, _config
from quantize_gguf import file_digest


def distribution(values):
    x = np.asarray(values)
    return dict(p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)), min=float(x.min()), max=float(x.max()), mean=float(x.mean()))



def cosine_quality(reference, candidate, threshold=0.999):
    """Gate each seven-dimensional action against BF16; padding is never scored."""
    x=np.asarray(reference,dtype=np.float64)[:,:,:7]
    y=np.asarray(candidate,dtype=np.float64)[:,:,:7]
    if x.shape != y.shape or x.ndim != 3 or x.shape[2] != 7 or not x.size:
        raise ValueError("Expected paired nonempty physical action chunks")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite actions are a zero-tolerance failure")
    def cosine(a,b,axis):
        na=np.linalg.norm(a,axis=axis);nb=np.linalg.norm(b,axis=axis)
        denom=na*nb
        out=np.divide(np.sum(a*b,axis=axis),denom,out=np.zeros_like(denom),where=denom>0)
        return np.clip(np.where((na==0)&(nb==0),1.,out),-1.,1.)
    global_value=float(cosine(x.reshape(-1),y.reshape(-1),0))
    chunks=cosine(x.reshape(len(x),-1),y.reshape(len(y),-1),1)
    dimensions=cosine(x,y,1)
    action_vectors=cosine(x,y,2)
    passed=bool(np.all(action_vectors>=threshold))
    return dict(threshold=threshold,global_cosine=global_value,chunk_cosine=chunks.tolist(),
                action_vector_cosine=action_vectors.tolist(),minimum_action_vector=float(action_vectors.min()),
                failed_action_vectors=np.argwhere(action_vectors<threshold).tolist(),
                per_chunk_dimension_cosine=dimensions.tolist(),minimum_chunk=float(chunks.min()),
                minimum_by_dimension=dimensions.min(axis=0).tolist(),minimum=float(min(global_value,chunks.min(),dimensions.min())),
                failed_chunks=np.flatnonzero(chunks<threshold).tolist(),
                failed_chunk_dimensions=np.argwhere(dimensions<threshold).tolist(),
                offline_cosine_pass=bool(passed),zero_norm_rule="both zero: 1; only one zero: 0",
                scope="acceptance: each 7D action at each of 50 times; global/chunk/per-dimension scores are diagnostics",
                limitation="Direction similarity does not bound amplitude error or establish closed-loop success.")

def compare(reference, candidate, performance_reference=None):
    a = json.loads(Path(reference).read_text())
    b = json.loads(Path(candidate).read_text())
    for key in ("input_sha256", "noise_sha256", "task_sha256", "episodes"):
        if a[key] != b[key]:
            raise ValueError(f"Unpaired runs: {key}")
    tokens_paired = "token_sha256" in a and "token_sha256" in b
    if tokens_paired and a["token_sha256"] != b["token_sha256"]:
        raise ValueError("Unpaired runs: tokens")
    x = np.load(Path(reference).with_suffix(".actions.npz"), allow_pickle=False)["actions"]
    y = np.load(Path(candidate).with_suffix(".actions.npz"), allow_pickle=False)["actions"]
    if x.shape != y.shape or x.shape[1:] != (50, 32):
        raise ValueError("Expected paired full 50x32 action chunks")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite actions are a zero-tolerance failure")
    error = y.astype(np.float64) - x
    real = error[:, :, :7]
    scales = np.asarray(a["action_half_range"], np.float64)
    if scales.shape != (7,) or not np.array_equal(scales, b["action_half_range"]):
        raise ValueError("Action normalization differs")
    normalized = real / np.where(scales > 0, scales, 1)
    def metrics(z):
        return dict(mae=np.mean(abs(z), axis=(0, 1)).tolist(),
                    rmse=np.sqrt(np.mean(z*z, axis=(0, 1))).tolist(),
                    bias=z.mean(axis=(0, 1)).tolist(), max_abs=np.max(abs(z), axis=(0, 1)).tolist(),
                    p95_abs=np.percentile(abs(z), 95, axis=(0, 1)).tolist(),
                    p99_abs=np.percentile(abs(z), 99, axis=(0, 1)).tolist())
    performance = json.loads(Path(performance_reference).read_text()) if performance_reference else a
    for key in ("input_sha256", "noise_sha256", "task_sha256", "episodes"):
        if performance[key] != b[key]:
            raise ValueError(f"Unpaired performance baseline: {key}")
    if "token_sha256" in performance and performance["token_sha256"] != b.get("token_sha256"):
        raise ValueError("Unpaired performance baseline: tokens")
    def reduction(metric, percentile):
        if metric not in performance:
            return None
        return 1-b[metric][percentile]/performance[metric][percentile]
    cosine=cosine_quality(x,y)
    bf16_reference=a.get("reference_kind")=="source-framework-bf16"
    return dict(reference=str(reference), candidate=str(candidate), samples=len(x),
        accuracy_reference_kind=a.get("reference_kind","floating-runtime-diagnostic"),
        performance_reference=str(performance_reference or reference),
        offline_accuracy_pass=bool(bf16_reference and tokens_paired and cosine["offline_cosine_pass"]),
        units=a["units"], tokens_paired=tokens_paired, cosine=cosine, physical=metrics(real), normalized=metrics(normalized),
        control_boundaries="unassessed: verified units and limits required",
        gripper_decisions="unassessed: verified control rule required",
        prefix8=metrics(real[:, :8]), tail8=metrics(real[:, -8:]),
        step_max_abs=np.max(abs(real), axis=(0, 2)).tolist(),
        action_increment_error=metrics(np.diff(real, axis=1)),
        padded_max_abs=float(np.max(abs(error[:, :, 7:]))),
        native_p50_reduction=reduction("native_ms","p50"),
        native_p95_reduction=reduction("native_ms","p95"),
        end_to_end_p50_reduction=reduction("predict_ms","p50"),
        end_to_end_p95_reduction=reduction("predict_ms","p95"),
        conclusion="Screening only; physical error budgets, stable heldout batches and closed-loop validation are separate.")


def run(args):
    if args.output.exists() or args.output.with_suffix(".actions.npz").exists():
        raise ValueError("Refusing to overwrite an experiment")
    manifest = json.loads(args.corpus.read_text())
    rows = [r for r in manifest["rows"] if r["split"] == args.split][args.offset:args.offset+args.samples]
    if len(rows) != args.samples or args.samples < 1 or args.warmup < 3:
        raise ValueError("Insufficient unique observations or fewer than three warmup runs")
    observations = []
    for row in rows:
        path = args.corpus.parent / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Corpus observation checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            observations.append(({f"observation.images.{k}": np.array(data[k]) for k in ("top", "wrist")},
                                 np.array(data["state"])))
    started = time.perf_counter()
    with load_policy(args.config, precision=args.precision) as model:
        loaded = time.perf_counter()
        tokens = [np.asarray(prepare_prompt(model._reader,row["task"],state)[1],np.int32)
                  for row, (_,state) in zip(rows,observations)]
        resolved = _config(args.config,args.precision)
        library = Path(resolved["library"])
        artifacts = [args.config,library,library.parent/"libvla_core.so",library.parent/"bin/libggml-musa.so",Path(__file__)]
        artifact_hashes = {str(p.resolve()):file_digest(p) for p in artifacts if p.is_file()}
        untracked = subprocess.check_output(["git","ls-files","--others","--exclude-standard","-z"],text=True).split("\0")
        untracked_hashes = {p:file_digest(p) for p in untracked if p.startswith(("src/","scripts/","tests/","bindings/")) and Path(p).is_file()}
        warm_shapes = {}
        for row, (images, state), tok in zip(rows, observations,tokens):
            length = len(tok)
            warm_shapes.setdefault(length, (row, images, state))
        for row, images, state in warm_shapes.values():
            model.warmup(images, state=state, task=row["task"], runs=args.warmup)
        raw_predict = model._native.predict
        capture = {}
        def record(*a, **kw):
            result = raw_predict(*a, **kw)
            capture["actions"] = result
            capture["noise"] = kw["noise"]
            return result
        model._native.predict = record
        times, native, preprocess, actions, noise_hashes = [], [], [], [], []
        for row, (images, state) in zip(rows, observations):
            t = time.perf_counter()
            model.predict(images, state=state, task=row["task"])
            times.append((time.perf_counter()-t)*1000)
            native.append(model.last_predict_timings_ms["native_call_ms"])
            preprocess.append(model.last_predict_timings_ms["preprocess_ms"])
            actions.append(capture["actions"])
            noise_hashes.append(hashlib.sha256(capture["noise"].tobytes()).hexdigest())
        tensors = {t.name: t for t in model._reader.tensors}
        ranges = (tensors["action_q99"].data.reshape(-1) - tensors["action_q01"].data.reshape(-1)) / 2
        report = dict(schema_version=1, precision=model.spec.precision,
            resolved_parameters=model.resolved_parameters, samples=len(rows), split=args.split,
            offset=args.offset, warmup=args.warmup, warmup_language_lengths=sorted(warm_shapes), load_seconds=loaded-started,
            input_sha256=[r["sha256"] for r in rows], noise_sha256=noise_hashes,
            token_sha256=[hashlib.sha256(t.tobytes()).hexdigest() for t in tokens],
            language_lengths=[len(t) for t in tokens], artifact_sha256=artifact_hashes,
            untracked_code_sha256=untracked_hashes,
            task_sha256=[hashlib.sha256(r["task"].encode()).hexdigest() for r in rows],
            episodes=[r["episode"] for r in rows], action_half_range=ranges[:7].tolist(),
            units=dict(joints=model.spec.joint_unit, gripper=model.spec.gripper_unit,
                       confirmed=model.spec.contract_verified),
            predict_ms=distribution(times), native_ms=distribution(native), preprocess_ms=distribution(preprocess),
            raw_predict_ms=times, raw_native_ms=native, raw_preprocess_ms=preprocess,
            environment={k:v for k,v in os.environ.items() if k.startswith(("VLA_", "GGML_"))},
            platform=platform.platform(), cpu=platform.processor(),
            code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            dirty_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"])).hexdigest(),
            scope="resident public API, no camera acquisition, robot I/O or closed-loop evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".actions.npz"), actions=np.asarray(actions))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:report[k] for k in ("precision", "samples", "predict_ms", "native_ms")}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--precision", default="baseline")
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--split", default="development", choices=("development", "heldout", "calibration"))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    ap.add_argument("--performance-reference",type=Path,help="Separate F16 timing baseline when --compare uses BF16 actions")
    args = ap.parse_args()
    if args.compare:
        if args.output.exists():
            raise ValueError("Refusing to overwrite comparison")
        result = compare(*args.compare, performance_reference=args.performance_reference)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    else:
        if args.config is None or args.corpus is None or args.offset < 0:
            ap.error("benchmark requires --config, --corpus and nonnegative --offset")
        run(args)


if __name__ == "__main__":
    main()
