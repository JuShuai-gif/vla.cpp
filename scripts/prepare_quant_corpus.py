#!/usr/bin/env python3
"""Freeze an episode-disjoint pi05 replay corpus from a local LeRobot v3 dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def frame_at(path, timestamp):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int(timestamp / stream.time_base)), stream=stream, backward=True)
        best, best_distance = None, float("inf")
        for frame in container.decode(stream):
            current = float(frame.pts * stream.time_base)
            distance = abs(current - timestamp)
            if distance < best_distance:
                best, best_distance = frame, distance
            if current >= timestamp:
                break
        if best is None or best_distance > 0.018:
            raise ValueError(f"Cannot align frame within half a 30Hz interval: {path} {timestamp} {best_distance}")
        return best.to_ndarray(format="rgb24"), best_distance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory to freeze the dataset split")
    info = json.loads((args.dataset / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != 30:
        raise ValueError("This extractor requires LeRobot v3 at 30Hz")
    episodes = []
    for file in sorted((args.dataset / "meta/episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(file).to_pylist())
    rng = np.random.default_rng(20261104)
    order = rng.permutation(len(episodes))
    if len(order) < 100:
        raise ValueError("Need at least 100 episodes for calibration/development/heldout split")
    # Split before reading observations. All samples of one episode stay together.
    split = {int(i): "calibration" if n < 40 else "development" if n < 70 else "heldout"
             for n, i in enumerate(order)}
    args.output.mkdir(parents=True)
    rows, tables = [], {}
    for i in sorted(range(len(episodes)), key=lambda i: (split[i] != "development", i)):
        episode = episodes[i]
        data_path = args.dataset / info["data_path"].format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"])
        if data_path not in tables:
            tables[data_path] = pq.read_table(data_path, columns=["observation.state", "episode_index", "frame_index", "timestamp"])
        data = tables[data_path]
        mask = np.asarray(data["episode_index"]) == episode["episode_index"]
        selected = data.filter(mask).to_pylist()
        if len(selected) != episode["length"]:
            raise ValueError("Episode length differs from data")
        for fraction in (0.1, 0.5, 0.9):
            row = selected[round((len(selected) - 1) * fraction)]
            frames, offsets = {}, {}
            for role in ("top", "wrist"):
                key = f"observation.images.{role}"
                prefix = f"videos/{key}/"
                path = args.dataset / info["video_path"].format(video_key=key,
                    chunk_index=episode[prefix + "chunk_index"], file_index=episode[prefix + "file_index"])
                timestamp = episode[prefix + "from_timestamp"] + row["timestamp"]
                frames[role], offsets[role] = frame_at(path, timestamp)
            name = f"ep{episode['episode_index']:05d}-f{row['frame_index']:05d}.npz"
            target = args.output / name
            state = np.asarray(row["observation.state"], np.float32)
            if state.shape != (7,) or not np.isfinite(state).all():
                raise ValueError("Invalid seven-dimensional state")
            np.savez(target, state=state, **frames)
            rows.append(dict(file=name, episode=int(episode["episode_index"]), frame=int(row["frame_index"]),
                split=split[i], task=episode["tasks"][0], alignment_seconds=offsets,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        if len(rows) % 30 == 0:
            print(f"prepared {len(rows)} observations", flush=True)
    manifest = dict(schema_version=1, dataset=str(args.dataset.resolve()), split_seed=20261104,
        limitation="Local real Piper data; exact training provenance for either checkpoint is unconfirmed. No closed-loop evidence.",
        rows=rows)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print({s: sum(r["split"] == s for r in rows) for s in set(split.values())})


if __name__ == "__main__":
    main()
