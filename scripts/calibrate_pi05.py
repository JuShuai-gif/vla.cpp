#!/usr/bin/env python3
# Copyright 2026 VinRobotics
# SPDX-License-Identifier: Apache-2.0
"""Collect real per-channel GEMM input statistics on the calibration split only.

Retaining intermediate tensors changes allocation and timing. These runs cannot
be used as performance evidence. Raw per-phase/per-occurrence reports are kept.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pi05_gguf import GGUFReader
from quantize_gguf import file_digest
from vla_cpp import load_policy


def original_names(name):
    name = name.removesuffix(" (view)")
    for suffix, replacements in ((".attn_qkv.weight", ("attn_q", "attn_k", "attn_v")),
                                  (".attn_kv.weight", ("attn_k", "attn_v")),
                                  (".ffn_gateup.weight", ("ffn_gate", "ffn_up"))):
        if name.endswith(suffix):
            return [name[:-len(suffix)] + "." + x + ".weight" for x in replacements]
    return [name]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--capture-inputs", help="comma-separated exact resident weight names; diagnostic F32 input dumps")
    args = ap.parse_args()
    if args.samples < 1:
        ap.error("samples must be positive")
    cfg = json.loads(args.config.read_text())
    for key in ("model", "library", "optimization_profile"):
        cfg[key] = str((args.config.resolve().parent / Path(cfg[key]).expanduser()).resolve())
    source_sha = file_digest(cfg["model"])
    corpus = json.loads(args.corpus.read_text())
    rows = [r for r in corpus["rows"] if r["split"] == "calibration"][:args.samples]
    if len(rows) != args.samples:
        ap.error("not enough calibration observations")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw = root / "raw"
    raw.mkdir()
    profiles = json.loads(Path(cfg["optimization_profile"]).read_text())
    for settings in profiles.values():
        settings["VLA_PI05_CALIBRATION_DIR"] = str(raw)
        if args.capture_inputs:
            settings["VLA_PI05_CALIBRATION_INPUTS"] = args.capture_inputs
    (root / "optimization.json").write_text(json.dumps(profiles, indent=2) + "\n")
    cfg["optimization_profile"] = str(root / "optimization.json")
    cfg["default_precision"] = "baseline"
    cfg.pop("precision_profiles", None)
    (root / "policy.json").write_text(json.dumps(cfg, indent=2) + "\n")
    aggregate, observations, unsupported, actions = {}, [], [], []
    with load_policy(root / "policy.json", precision="baseline") as model:
        source_names = {t.name for t in model._reader.tensors}
        for row in rows:
            path = args.corpus.parent / row["file"]
            if file_digest(path) != row["sha256"]:
                raise ValueError("Observation SHA mismatch")
            old = set(raw.iterdir())
            with np.load(path, allow_pickle=False) as data:
                images = {f"observation.images.{k}": np.array(data[k]) for k in ("top", "wrist")}
                action = model.predict(images, state=np.array(data["state"]), task=row["task"])
            actions.append(np.array(action))
            files = sorted(set(raw.iterdir()) - old)
            if len(files) != 3:
                raise ValueError("Expected vision, prefix and denoise traces; native library may lack calibration support")
            stats_hashes = {}
            for path in files:
                stats_hashes[path.name] = file_digest(path)
                for line in path.read_text().splitlines():
                    stat = json.loads(line)
                    if stat["status"] != "captured":
                        unsupported.append(dict(observation=row["file"], **stat))
                        continue
                    for name in original_names(stat["weight"]):
                        if name not in source_names:
                            raise ValueError(f"No source tensor for captured weight {name}")
                        maximum = np.asarray(stat["amax"], np.float64)
                        squares = np.asarray(stat["sum_squares"], np.float64)
                        if name not in aggregate:
                            aggregate[name] = dict(amax=maximum, sum_squares=squares, rows=stat["rows"])
                        else:
                            a = aggregate[name]
                            a["amax"] = np.maximum(a["amax"], maximum)
                            a["sum_squares"] += squares
                            a["rows"] += stat["rows"]
            observations.append(dict(file=row["file"], sha256=row["sha256"], episode=row["episode"],
                                     traces=stats_hashes, action_sha256=hashlib.sha256(np.asarray(action).tobytes()).hexdigest()))
            print(f"calibrated {len(observations)}/{len(rows)}", flush=True)
    np.savez_compressed(root / "actions.npz", actions=np.asarray(actions))
    report = dict(schema_version=1, split="calibration", source_sha256=source_sha,
                  corpus_sha256=file_digest(args.corpus), observations=observations, unsupported=unsupported,
                  weights={name: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()}
                           for name, stats in aggregate.items()},
                  scope="per-channel statistics only; no performance or deployment-quality verdict")
    (root / "importance.json").write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
