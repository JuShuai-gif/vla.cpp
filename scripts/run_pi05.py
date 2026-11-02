#!/usr/bin/env python3
"""Run a standard or attention-pruned pi05 GGUF with its saved preprocessing.

Example: --model model.gguf --image top=top.jpg --image wrist=wrist.jpg
         --text 'pick up the cube' --state '0,0,0,0,0,0,0' --device 0
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import gguf
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bindings/python"))


def prepare_prompt(reader, text, state):
    tensors = {t.name: t for t in reader.tensors}
    q01 = np.asarray(tensors["state_q01"].data, dtype=np.float32).reshape(-1)
    q99 = np.asarray(tensors["state_q99"].data, dtype=np.float32).reshape(-1)
    state = np.asarray(state, dtype=np.float32)
    if state.shape != q01.shape or not np.isfinite(state).all():
        raise ValueError(f"Expected {len(q01)} finite raw state values")
    eps = reader.fields["pi05.norm_eps"].contents()
    denom = q99 - q01
    normed = 2 * (state - q01) / np.where(denom == 0, eps, denom) - 1
    bins = np.digitize(normed, np.linspace(-1, 1, 257)[:-1]) - 1
    text = text.strip().replace("_", " ").replace("\n", " ")
    prompt = f"Task: {text}, State: {' '.join(map(str, bins.tolist()))};\nAction: "
    length = int(reader.fields["pi05.tokenizer_max_length"].contents())
    if "pi05.tokenizer.hf_json" in reader.fields:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_str(reader.fields["pi05.tokenizer.hf_json"].contents())
        tok.no_padding()
        tok.enable_truncation(max_length=length)
        ids = tok.encode(prompt).ids
    elif "pi05.tokenizer.spm_model" in reader.fields:
        import sentencepiece as spm
        model = bytes(reader.fields["pi05.tokenizer.spm_model"].contents())
        tok = spm.SentencePieceProcessor(model_proto=model)
        ids = ([tok.bos_id()] + tok.encode(prompt))[:length]
    else:
        raise ValueError("GGUF has no local tokenizer; use convert_pi05_local.py")
    return prompt, ids


def prepare_image(path, size):
    # Match LeRobot's float32 bilinear resize (align_corners=False), centered zero pad.
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(4)
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    if (h, w) != (size, size):
        ratio = max(w / size, h / size)
        rh, rw = max(1, int(h / ratio)), max(1, int(w / ratio))
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None]
        tensor = F.interpolate(tensor, size=(rh, rw), mode="bilinear", align_corners=False).clamp(0, 1)
        top, left = (size - rh) // 2, (size - rw) // 2
        tensor = F.pad(tensor, (left, size - rw - left, top, size - rh - top))
        rgb = tensor[0].permute(1, 2, 0).numpy()
    return np.ascontiguousarray(rgb, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--image", action="append", required=True, metavar="CAMERA=PATH")
    ap.add_argument("--text", required=True)
    ap.add_argument("--state", required=True, help="Raw state values in checkpoint units, comma-separated")
    ap.add_argument("--device", default="0", help="MUSA device index, or cpu")
    ap.add_argument("--weight-dtype", choices=("f32", "f16", "bf16"), default="f32")
    ap.add_argument("--act-dtype", choices=("f32", "f16"), default="f32")
    ap.add_argument("--no-fuse", action="store_true", help="Disable MUSA expert fusion for A/B checks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    os.environ["VLA_DEVICE"] = args.device
    if args.no_fuse:
        os.environ["VLA_PI05_NO_FUSE"] = "1"
    os.environ.setdefault("VLA_N_THREADS", "4")
    os.environ.setdefault("VLA_LIBRARY", str(REPO / "build-musa-m1000/libvla.so"))
    import vla_cpp
    reader = gguf.GGUFReader(str(args.model))
    if reader.fields["general.architecture"].contents() != "pi05":
        raise ValueError("Expected a pi05 GGUF")
    cameras = reader.fields["pi05.camera_keys"].contents()
    provided = {}
    for item in args.image:
        key, path = item.split("=", 1)
        if not key.startswith("observation.images."): key = "observation.images." + key
        if key in provided: raise ValueError(f"Duplicate camera {key}")
        provided[key] = path
    if set(provided) != set(cameras):
        raise ValueError(f"Expected exactly these cameras: {cameras}")
    state = np.asarray([float(x) for x in args.state.split(",")], dtype=np.float32)
    prompt, tokens = prepare_prompt(reader, args.text, state)
    size = int(reader.fields["pi05.image_size"].contents())
    images = [prepare_image(provided[k], size) for k in cameras]
    del reader
    runtime = {"runtime": {"weight_dtype": args.weight_dtype, "act_dtype": args.act_dtype,
                           "mm_prec": "default" if args.act_dtype == "f16" else "f32", "flash_attn": False}}
    with tempfile.TemporaryDirectory(prefix="vla-pi05-") as temp:
        config = Path(temp) / "runtime.json"
        config.write_text(json.dumps(runtime))
        with vla_cpp.load(str(args.model), config_path=str(config)) as model:
            cfg = model.config
            padded = np.zeros(int(cfg.max_state_dim), dtype=np.float32)
            padded[:len(state)] = state
            noise = np.random.default_rng(args.seed).standard_normal(int(cfg.n_suffix * cfg.max_action_dim)).astype(np.float32)
            actions = model.predict(images, tokens=tokens, state=padded, noise=noise, pixel_format=vla_cpp.PIXEL_F32_RGB_01, timing=vla_cpp.TIMING_PHASE)
            if not np.isfinite(actions).all(): raise ValueError("Non-finite policy output")
            st = model.last_stats()
            result = {"model": str(args.model.resolve()), "device_requested": args.device,
                      "weight_dtype": args.weight_dtype, "camera_order": cameras,
                      "act_dtype": args.act_dtype,
                      "fusion_disabled": os.environ.get("VLA_PI05_NO_FUSE") == "1",
                      "tokens_per_image": int(cfg.n_img), "num_steps": int(cfg.num_steps),
                      "seed": args.seed, "prompt": prompt, "token_ids": tokens,
                      "denormalized": bool(cfg.denormalized),
                      "actions": actions[:, :cfg.real_action_dim].tolist(),
                      "timing_ms": {k: float(getattr(st, k)) for k in ("ms_total", "ms_vision", "ms_inference")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(result['actions'])} x {len(state)} actions to {args.output}; cameras={cameras}")


if __name__ == "__main__":
    main()
