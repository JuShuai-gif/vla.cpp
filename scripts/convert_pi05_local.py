#!/usr/bin/env python3
"""Convert a local LeRobot pi05 checkpoint, including its own stats/tokenizer.

Supports standard SigLIP and attention-pruned SigLIP checkpoints. Uses the
project's convert_pi05_to_gguf, then verifies every serialized tensor payload.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

import convert_pi05_to_gguf as converter


def checkpoint_dir(path):
    path = Path(path).resolve()
    if not (path / "model.safetensors").exists():
        path = path / "pretrained_model"
    if not (path / "model.safetensors").is_file():
        raise ValueError(f"No model.safetensors under {path}")
    return path


def processor_stats(ckpt, cfg):
    def state_file(filename, registry):
        pipeline = json.loads((ckpt / filename).read_text())
        matches = [s for s in pipeline["steps"] if s["registry_name"] == registry]
        if len(matches) != 1:
            raise ValueError(f"Expected one {registry} in {filename}")
        return ckpt / matches[0]["state_file"]

    pre = state_file("policy_preprocessor.json", "normalizer_processor")
    post = state_file("policy_postprocessor.json", "unnormalizer_processor")
    stats = {}
    with safe_open(pre, framework="np") as a, safe_open(post, framework="np") as b:
        for feat, dim in (("observation.state", cfg["input_features"]["observation.state"]["shape"][0]),
                          ("action", cfg["output_features"]["action"]["shape"][0])):
            stats[feat] = {}
            for key in ("mean", "std", "q01", "q99"):
                name = feat + "." + key
                value = a.get_tensor(name).reshape(-1)
                if value.size != dim or not np.isfinite(value).all():
                    raise ValueError(f"Invalid stats for {name}")
                if name in b.keys() and not np.array_equal(value, b.get_tensor(name).reshape(-1)):
                    raise ValueError(f"Pre/post stats differ for {name}")
                stats[feat][key] = value.tolist()
            if np.any(np.array(stats[feat]["q99"]) < np.array(stats[feat]["q01"])):
                raise ValueError(f"Invalid quantile interval for {feat}")
    return stats


def local_tokenizer(ckpt, requested):
    candidates = [Path(requested)] if requested else [ckpt / "tokenizer"]
    if not requested:
        pre = json.loads((ckpt / "policy_preprocessor.json").read_text())
        for step in pre["steps"]:
            if step["registry_name"] == "tokenizer_processor":
                p = Path(step["config"]["tokenizer_name"])
                candidates.append(p if p.is_absolute() else ckpt / p)
    for p in candidates:
        if p.is_dir():
            # Prefer the saved fast tokenizer, including its added-token contract.
            p = p / ("tokenizer.json" if (p / "tokenizer.json").exists() else "tokenizer.model")
        if p.is_file() and p.suffix in (".model", ".json"):
            return p.resolve()
    raise ValueError("Local training tokenizer not found; pass --tokenizer PATH")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tokenizer", type=Path)
    args = ap.parse_args()
    ckpt = checkpoint_dir(args.ckpt)
    out = args.out.resolve()
    if out.exists():
        raise ValueError(f"Refusing to overwrite {out}")
    cfg = json.loads((ckpt / "config.json").read_text())
    if cfg.get("type") != "pi05" or cfg.get("use_relative_actions", False) or cfg.get("empty_cameras", 0):
        raise ValueError("Expected pi05 with absolute actions and no empty camera slots")
    stats = processor_stats(ckpt, cfg)
    tok = local_tokenizer(ckpt, args.tokenizer)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    temporary = out.with_name(out.stem + ".partial.gguf")
    if temporary.exists():
        raise ValueError(f"Previous partial output exists: {temporary}")
    audit, used = {}, set()
    original_open = converter.safe_open

    class Source:
        def __init__(self, *a, **kw): self.sf = original_open(*a, **kw)
        def keys(self): return self.sf.keys()
        def get_slice(self, k): return self.sf.get_slice(k)
        def get_tensor(self, k):
            used.add(k)
            return self.sf.get_tensor(k)

    class Writer(gguf.GGUFWriter):
        def add_tensor(self, name, tensor, **kwargs):
            array = np.ascontiguousarray(tensor)
            audit[name] = {"shape": list(array.shape), "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
            return super().add_tensor(name, tensor, **kwargs)

    cameras = [k for k, v in cfg["input_features"].items() if v["type"] == "VISUAL"]

    def factory(path, arch):
        writer = Writer(str(path), arch)
        writer.add_string("pi05.architecture", arch)
        writer.add_array("pi05.camera_keys", cameras)
        if tok.suffix == ".model":
            writer.add_array("pi05.tokenizer.spm_model", tok.read_bytes())
        else:
            writer.add_string("pi05.tokenizer.hf_json", tok.read_text())
        for key in ("q01", "q99"):
            writer.add_array("pi05.state." + key, stats["observation.state"][key])
        return writer

    converter.safe_open = Source
    try:
        converter.convert(ckpt, temporary, writer_factory=factory, dataset_stats=stats_path)
    finally:
        converter.safe_open = original_open
    with safe_open(ckpt / "model.safetensors", framework="pt") as source:
        unused = sorted(set(source.keys()) - used)
        for name in unused:
            # The diffusion expert never uses its vocabulary output head.
            if name in ("model.paligemma_with_expert.gemma_expert.lm_head.weight",
                        "model.paligemma_with_expert.paligemma.lm_head.weight"):
                continue
            if name.endswith("paligemma.model.language_model.embed_tokens.weight"):
                import torch
                if torch.equal(source.get_tensor(name), source.get_tensor("model.paligemma_with_expert.paligemma.lm_head.weight")):
                    continue
            raise ValueError(f"Unconverted checkpoint tensor: {name}")
    reader = gguf.GGUFReader(str(temporary))
    if len(reader.tensors) != len(audit):
        raise ValueError("Tensor count mismatch")
    for tensor in reader.tensors:
        digest = hashlib.sha256(memoryview(np.ascontiguousarray(tensor.data)).cast("B")).hexdigest()
        if digest != audit[tensor.name]["sha256"]:
            raise ValueError(f"Tensor payload mismatch: {tensor.name}")
    layers = reader.fields["pi05.vit_layers"].contents()
    tokens = cfg.get("vision_token_prune", 0) or reader.fields["pi05.n_img_tokens"].contents()
    del reader
    temporary.replace(out)
    h = hashlib.sha256()
    with out.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    report = {"checkpoint": str(ckpt), "gguf": str(out), "sha256": h.hexdigest(),
              "tensor_count": len(audit), "tensor_payloads_verified": True, "unused_in_diffusion": unused,
              "vision_layers": layers, "tokens_per_image": tokens, "camera_order": cameras,
              "chunk_size": cfg["chunk_size"], "num_steps": cfg["num_inference_steps"], "tokenizer": str(tok), "tensors": audit}
    out.with_suffix(".conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    out.with_suffix(".gguf.sha256").write_text(h.hexdigest() + "  " + out.name + "\n")
    print(f"VERIFIED {len(audit)} tensor payloads; {layers} vision layers; {tokens} tokens/image; cameras={cameras}", flush=True)
    print(f"GGUF: {out}\nSHA256: {h.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
