#!/usr/bin/env python3
# Copyright 2026 VinRobotics
# SPDX-License-Identifier: Apache-2.0
"""Run development screens in fresh processes and retain failed-process evidence.

A JSON report written before a native shutdown crash is not a successful run.
This tool does not replace independent heldout ABBA or physical/closed-loop gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from bench_pi05_precision import compare


def execute(command, report, log):
    started = time.time()
    with Path(log).open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    status = "completed" if result.returncode == 0 and Path(report).is_file() else "process_failed"
    return dict(status=status, returncode=result.returncode, report=str(report), log=str(log),
                elapsed_seconds=time.time()-started)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--precisions", nargs="+", required=True)
    ap.add_argument("--accuracy-reference",type=Path,help="Paired source-framework BF16 action report")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.samples < 1 or args.offset < 0 or "baseline" in args.precisions or len(set(args.precisions)) != len(args.precisions):
        ap.error("positive sample count, nonnegative offset and unique candidate names required; baseline runs automatically")
    if args.accuracy_reference:
        accuracy=json.loads(args.accuracy_reference.read_text())
        if accuracy.get("reference_kind")!="source-framework-bf16":
            ap.error("--accuracy-reference must come from the original-checkpoint BF16 forward runner")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).with_name("bench_pi05_precision.py")
    runs = []
    manifest = dict(schema_version=1, scope="development screening", runs=runs,
                    accuracy_reference=str(args.accuracy_reference) if args.accuracy_reference else None, physical_quality="unassessed", closed_loop="unassessed", qualified_acceleration=False)
    baseline = None
    for precision in ["baseline", *args.precisions]:
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", precision) + "-" + hashlib.sha256(precision.encode()).hexdigest()[:8]
        report, log = root / (name + ".json"), root / (name + ".log")
        command = [sys.executable, str(runner), "--config", str(args.config.resolve()), "--corpus", str(args.corpus.resolve()),
                   "--precision", precision, "--split", "development", "--samples", str(args.samples),
                   "--offset", str(args.offset), "--output", str(report)]
        entry = execute(command, report, log)
        entry["precision"] = precision
        if entry["status"] == "completed":
            if precision == "baseline":
                baseline = report
            elif baseline:
                try:
                    result = compare(args.accuracy_reference or baseline, report, performance_reference=baseline)
                    comparison = root / (name + "-comparison.json")
                    comparison.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                    entry["comparison"] = str(comparison)
                    entry["cosine_diagnostic_pass"] = result["cosine"]["offline_cosine_pass"]
                    entry["offline_accuracy_pass"] = result["offline_accuracy_pass"]
                    entry["performance_screen_pass"] = bool(
                        result["native_p50_reduction"] >= .20 and result["native_p95_reduction"] >= .15
                        and result["end_to_end_p50_reduction"] >= .15 and result["end_to_end_p95_reduction"] >= 0)
                except (ValueError, KeyError) as exc:
                    entry.update(status="comparison_failed", error=str(exc))
        runs.append(entry)
        (root / "suite.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print(precision, entry["status"], entry.get("performance_screen_pass", "unassessed"), flush=True)
        if precision=="baseline" and entry["status"]!="completed":
            manifest["unrun_precisions"]=args.precisions
            manifest["stopped_reason"]="Baseline process failed; no valid paired performance reference"
            (root / "suite.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
            break
    return int(any(r["status"] != "completed" for r in runs))


if __name__ == "__main__":
    raise SystemExit(main())
