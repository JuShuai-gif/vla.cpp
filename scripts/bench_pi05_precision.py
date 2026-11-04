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
from vla_cpp.policy import prepare_prompt


def distribution(values):
    x = np.asarray(values)
    return dict(p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)), min=float(x.min()), max=float(x.max()), mean=float(x.mean()))


def compare(reference, candidate):
    a = json.loads(Path(reference).read_text())
    b = json.loads(Path(candidate).read_text())
    for key in ("input_sha256", "noise_sha256", "task_sha256", "episodes"):
        if a[key] != b[key]:
            raise ValueError(f"Unpaired runs: {key}")
    x = np.load(Path(reference).with_suffix(".actions.npz"))["actions"]
    y = np.load(Path(candidate).with_suffix(".actions.npz"))["actions"]
    if x.shape != y.shape or x.shape[1:] != (50, 32):
        raise ValueError("Expected paired full 50x32 action chunks")
    error = y.astype(np.float64) - x
    real = error[:, :, :7]
    scales = np.asarray(a["action_half_range"], np.float64)
    if scales.shape != (7,) or not np.array_equal(scales, b["action_half_range"]):
        raise ValueError("Action normalization differs")
    normalized = real / np.where(scales > 0, scales, 1)
    def metrics(z):
        return dict(rmse=np.sqrt(np.mean(z*z, axis=(0, 1))).tolist(),
                    bias=z.mean(axis=(0, 1)).tolist(), max_abs=np.max(abs(z), axis=(0, 1)).tolist(),
                    p95_abs=np.percentile(abs(z), 95, axis=(0, 1)).tolist(),
                    p99_abs=np.percentile(abs(z), 99, axis=(0, 1)).tolist())
    return dict(reference=str(reference), candidate=str(candidate), samples=len(x),
        units=a["units"], physical=metrics(real), normalized=metrics(normalized),
        prefix8=metrics(real[:, :8]), tail8=metrics(real[:, -8:]),
        step_max_abs=np.max(abs(real), axis=(0, 2)).tolist(),
        action_increment_error=metrics(np.diff(real, axis=1)),
        padded_max_abs=float(np.max(abs(error[:, :, 7:]))),
        native_p50_reduction=1-b["native_ms"]["p50"]/a["native_ms"]["p50"],
        native_p95_reduction=1-b["native_ms"]["p95"]/a["native_ms"]["p95"],
        end_to_end_p50_reduction=1-b["predict_ms"]["p50"]/a["predict_ms"]["p50"],
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
        warm_shapes = {}
        for row, (images, state) in zip(rows, observations):
            length = len(prepare_prompt(model._reader, row["task"], state)[1])
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
    args = ap.parse_args()
    if args.compare:
        if args.output.exists():
            raise ValueError("Refusing to overwrite comparison")
        result = compare(*args.compare)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    else:
        if args.config is None or args.corpus is None or args.offset < 0:
            ap.error("benchmark requires --config, --corpus and nonnegative --offset")
        run(args)


if __name__ == "__main__":
    main()
