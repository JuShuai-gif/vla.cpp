#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pack floating GGUF matrices with explicit, fusion-compatible recipes.

Packing saves storage; speed and action accuracy require device measurements.
The source and existing outputs are never overwritten. A verified manifest is
written next to each model. --dry-run describes every decision without writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import gguf
import numpy as np

from gguf_common import copy_kv, copy_tensor
from pi05_gguf import GGUFReader

QT = gguf.GGMLQuantizationType
QK = {name: 32 for name in ("Q8_0", "Q4_0", "Q5_0", "Q4_1", "Q5_1")}
RECIPES = ("auto", "prefix-mlp", "prefix", "vision", "expert-mlp", "expert",
           "prefix-vision", "prefix-mixed", "all-linear", "map")
SKIP = ("token_embd", "tok_embd", "output.weight", "patch_embd", "norm", "pos",
        "embed", "cls", "action", "state", "aex.", "ah.", "act.", "octo.head",
        "dit", "adaln", "ada_", "time")
SKIP_VISION = ("vit", "vision", "vis.d.", "vis.s.", "octo.obs.")
LINEAR = re.compile(r"^(vlm|aex|vit)\.blk\.(\d+)\.(attn_[qkvo]|ffn_(?:gate|up|down)|fc[12])\.weight$")


def digest(data):
    return hashlib.sha256(memoryview(np.ascontiguousarray(data)).cast("B")).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def bf16_to_f32(data):
    return (np.ascontiguousarray(data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def to_f32(t):
    shape = tuple(int(x) for x in t.shape[::-1])
    if t.tensor_type in (QT.F32, QT.F16):
        return t.data.astype(np.float32, copy=False).reshape(shape)
    if t.tensor_type == QT.BF16:
        return bf16_to_f32(t.data).reshape(shape)
    raise ValueError(f"{t.name}: unsupported source {t.tensor_type.name}")


def eligible(name, shape, qtype, skip):
    return (len(shape) == 2 and name.endswith(".weight")
            and not any(s in name for s in skip) and int(shape[0]) % QK[qtype] == 0)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate tensor in map: {key}")
        result[key] = value
    return result


def make_plan(reader, recipe="auto", qtype="Q8_0", overrides=None, vision=False):
    arch = reader.fields["general.architecture"].contents()
    if f"{arch}.quant.method" in reader.fields:
        raise ValueError("FoldQuant sources cannot be repacked as ggml block types")
    for t in reader.tensors:
        if gguf.GGML_QUANT_SIZES[t.tensor_type][0] > 1:
            raise ValueError(f"already quantized source: {t.name} ({t.tensor_type.name})")
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError("map must be an object of exact tensor names to types or KEEP")
    names = {t.name for t in reader.tensors}
    if set(overrides) - names:
        raise ValueError(f"unknown tensors in map: {sorted(set(overrides) - names)}")
    if any(not isinstance(v, str) or v not in (*QK, "KEEP") for v in overrides.values()):
        raise ValueError("map values must be supported block types or KEEP")
    if recipe == "map" and not overrides:
        raise ValueError("map recipe requires a nonempty --map")
    if recipe == "auto" and arch == "pi05":
        recipe = "prefix-vision" if vision else "prefix"
    elif vision:
        if recipe != "auto":
            raise ValueError("--vision is only valid with --recipe auto")
    if recipe != "auto" and arch != "pi05":
        raise ValueError("named recipes currently require pi05")
    rows = []
    for t in reader.tensors:
        target, reason, group = "KEEP", "protected/unselected", None
        match = LINEAR.fullmatch(t.name)
        if match:
            tower, layer, op = match.groups()
            family = "qkv" if op in ("attn_q", "attn_k", "attn_v") else "gu" if op in ("ffn_gate", "ffn_up") else op
            group = f"{tower}.blk.{layer}.{family}"
            selected = ((tower == "vlm" and recipe in ("prefix", "prefix-vision", "prefix-mixed", "all-linear"))
                        or (tower == "vlm" and recipe == "prefix-mlp" and op.startswith("ffn_"))
                        or (tower == "vit" and recipe in ("vision", "prefix-vision", "all-linear"))
                        or (tower == "aex" and recipe in ("expert", "all-linear"))
                        or (tower == "aex" and recipe == "expert-mlp" and op.startswith("ffn_")))
            if selected:
                target = "Q4_0" if recipe == "prefix-mixed" and family == "gu" else qtype
                reason = "recipe"
        if recipe == "auto" and eligible(t.name, t.shape, qtype, SKIP if vision else SKIP + SKIP_VISION):
            target, reason = qtype, "legacy non-pi05 selection"
        if target != "KEEP" and (len(t.shape) != 2 or int(t.shape[0]) % QK[target]):
            target, reason = "KEEP", "row width is not divisible by block size"
        if t.name in overrides:
            target, reason = overrides[t.name], "explicit map"
        if target != "KEEP":
            if len(t.shape) != 2 or not t.name.endswith(".weight") or int(t.shape[0]) % QK[target]:
                raise ValueError(f"{t.name}: not a block-aligned 2D weight")
            if t.tensor_type not in (QT.F32, QT.F16, QT.BF16):
                raise ValueError(f"{t.name}: unsupported source {t.tensor_type.name}")
            # Maps refine precision within approved linear sites, never heads or embeddings.
            if arch == "pi05" and match is None:
                raise ValueError(f"{t.name}: protected pi05 tensor")
        size = int(t.data.nbytes)
        out_size = size if target == "KEEP" else int(np.prod(t.shape)) // 32 * gguf.GGML_QUANT_SIZES[QT[target]][1]
        rows.append(dict(name=t.name, shape=list(map(int, t.shape)), source_type=t.tensor_type.name,
                         target_type=t.tensor_type.name if target == "KEEP" else target,
                         quantize=target != "KEEP", fusion_group=group, reason=reason,
                         source_bytes=size, target_bytes=out_size))
    groups = {}
    for row in rows:
        if row["fusion_group"]:
            groups.setdefault(row["fusion_group"], set()).add(row["target_type"] if row["quantize"] else "float")
    bad = [g for g, types in groups.items() if len(types) > 1]
    if bad:
        raise ValueError(f"fusion groups must share precision: {bad}")
    if not any(r["quantize"] for r in rows):
        raise ValueError("selection would quantize zero tensors")
    return dict(schema_version=1, architecture=arch, recipe=recipe, default_type=qtype,
                quantized_tensors=sum(r["quantize"] for r in rows), tensors=rows,
                payload_bytes_before=sum(r["source_bytes"] for r in rows),
                payload_bytes_after=sum(r["target_bytes"] for r in rows))


def verify(reader, path, plan):
    result = GGUFReader(str(path))
    if len(result.tensors) != len(reader.tensors):
        raise ValueError("readback tensor count mismatch")
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key not in result.fields or result.fields[key].types != field.types:
            raise ValueError(f"readback metadata type mismatch: {key}")
        a, b = field.contents(), result.fields[key].contents()
        if not np.array_equal(a, b):
            raise ValueError(f"readback metadata mismatch: {key}")
    for row, t in zip(plan["tensors"], result.tensors):
        if (t.name != row["name"] or list(t.shape) != row["shape"]
                or t.tensor_type.name != row["target_type"] or digest(t.data) != row["output_sha256"]):
            raise ValueError(f"readback tensor mismatch: {row['name']}")


def convert(src, dst, recipe="auto", qtype="Q8_0", overrides=None, vision=False, dry_run=False):
    src, dst = Path(src).resolve(), Path(dst).resolve()
    manifest = Path(str(dst) + ".quant.json")
    if src == dst or dst.exists() or manifest.exists():
        raise ValueError("source, output and manifest must be distinct; outputs must not exist")
    reader = GGUFReader(str(src))
    plan = make_plan(reader, recipe, qtype, overrides, vision)
    plan.update(source=str(src), output=str(dst), status="planned" if dry_run else "converting")
    if dry_run:
        return plan
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".partial", dir=dst.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    manifest_tmp = Path(tmp_name + ".json")
    writer = None
    published = False
    try:
        plan["source_sha256"] = file_digest(src)
        writer = gguf.GGUFWriter(str(tmp), plan["architecture"], use_temp_file=True)
        copy_kv(reader, writer)
        for t, row in zip(reader.tensors, plan["tensors"]):
            row["source_sha256"] = digest(t.data)
            if row["quantize"]:
                values = to_f32(t)
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite source: {t.name}")
                packed = gguf.quants.quantize(values, QT[row["target_type"]])
                row["output_sha256"] = digest(packed)
                writer.add_tensor(t.name, packed, raw_dtype=QT[row["target_type"]])
                del values, packed
            else:
                row["output_sha256"] = row["source_sha256"]
                copy_tensor(writer, t)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        verify(reader, tmp, plan)
        # Detect a source modified while conversion was running.
        if file_digest(src) != plan["source_sha256"]:
            raise ValueError("source changed during conversion")
        plan.update(output_sha256=file_digest(tmp), status="verified")
        manifest_tmp.write_text(json.dumps(plan, indent=2) + "\n")
        # Exclusive publication: a concurrent writer cannot be overwritten.
        os.link(tmp, dst)
        published = True
        os.link(manifest_tmp, manifest)
    except BaseException:
        if published:
            dst.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--type", default="Q8_0", choices=sorted(QK))
    ap.add_argument("--recipe", default="auto", choices=RECIPES)
    ap.add_argument("--map", type=Path, help="JSON object: exact tensor name -> type or KEEP")
    ap.add_argument("--vision", action="store_true", help="legacy alias for auto + vision")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    overrides = json.loads(args.map.read_text(), object_pairs_hook=unique_keys) if args.map else None
    try:
        plan = convert(args.src, args.dst, args.recipe, args.type, overrides, args.vision, args.dry_run)
    except ValueError as exc:
        ap.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        print(json.dumps({k: v for k, v in plan.items() if k != "tensors"}, indent=2))


if __name__ == "__main__":
    main()
