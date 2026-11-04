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

from pi05_gguf import GGUFReader
from pi05_preprocess import prepare_image, prepare_images
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bindings/python"))


from vla_cpp.policy import prepare_prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--image", action="append", required=True, metavar="CAMERA=PATH")
    ap.add_argument("--text", required=True)
    ap.add_argument("--state", required=True, help="Raw state values in checkpoint units, comma-separated")
    ap.add_argument("--device", default="0", help="MUSA device index, or cpu")
    ap.add_argument("--weight-dtype", choices=("f32", "f16", "bf16"), default="f32")
    ap.add_argument("--act-dtype", choices=("f32", "f16"), default="f32")
    ap.add_argument("--flash-attn", action="store_true", help="Enable compiled FlashAttention kernels")
    ap.add_argument("--no-fuse", action="store_true", help="Disable MUSA expert fusion for A/B checks")
    ap.add_argument("--musa-optimized", action="store_true", help="Use the validated M1000 F16-weight/F32-activation/FA profile; detect pruned/full from GGUF")
    ap.add_argument("--warmup", type=int, default=0, help="Resident warmup predictions; use 2 to exercise Graph replay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.warmup < 0:
        ap.error("--warmup must be nonnegative")
    if args.musa_optimized and (args.device == "cpu" or args.no_fuse):
        ap.error("--musa-optimized requires a MUSA device and enabled fusion")
    os.environ["VLA_DEVICE"] = args.device
    if args.no_fuse:
        os.environ["VLA_PI05_NO_FUSE"] = "1"
    os.environ.setdefault("VLA_N_THREADS", "4")
    build = "build-musa-official" if args.musa_optimized else "build-musa-m1000"
    os.environ.setdefault("VLA_LIBRARY", str(REPO / build / "libvla.so"))
    reader = GGUFReader(str(args.model))
    if reader.fields["general.architecture"].contents() != "pi05":
        raise ValueError("Expected a pi05 GGUF")
    optimized_profile = None
    if args.musa_optimized:
        field = reader.fields.get("pi05.vision_token_prune")
        optimized_profile = "pruned" if field is not None and int(field.contents()) > 0 else "full"
        profiles = json.loads((REPO / "scripts/pi05_musa_profiles.json").read_text())
        inactive = {"VLA_PI05_K_ROPE_DIRECT", "VLA_PI05_PREFIX_GAMMA_FOLD", "VLA_PI05_PREFETCH_KIB",
                    "VLA_MUSA_GEGLU_FAST", "VLA_PI05_F16_PART", "VLA_MUSA_FLATTEN_GEMM",
                    "GGML_CUDA_DISABLE_GRAPHS", "VLA_PI05_NO_SEPARATE_GRAPHS",
                    "VLA_PI05_NO_VISION_BATCH", "VLA_PI05_NO_PROJ_FUSE", "VLA_PI05_NO_FUSE"}
        for key in set().union(*(x.keys() for x in profiles.values())) | inactive:
            os.environ.pop(key, None)
        os.environ.update(profiles[optimized_profile])
        args.weight_dtype, args.act_dtype, args.flash_attn = "f16", "f32", True
    import vla_cpp
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
    images = prepare_images([provided[k] for k in cameras], size)
    del reader
    runtime = {"runtime": {"weight_dtype": args.weight_dtype, "act_dtype": args.act_dtype,
                           "mm_prec": "default" if args.act_dtype == "f16" else "f32", "flash_attn": args.flash_attn}}
    with tempfile.TemporaryDirectory(prefix="vla-pi05-") as temp:
        config = Path(temp) / "runtime.json"
        config.write_text(json.dumps(runtime))
        with vla_cpp.load(str(args.model), config_path=str(config)) as model:
            cfg = model.config
            padded = np.zeros(int(cfg.max_state_dim), dtype=np.float32)
            padded[:len(state)] = state
            noise = np.random.default_rng(args.seed).standard_normal(int(cfg.n_suffix * cfg.max_action_dim)).astype(np.float32)
            first = None
            repeat_max_abs = 0.0
            for _ in range(args.warmup + 1):
                actions = model.predict(images, tokens=tokens, state=padded, noise=noise, pixel_format=vla_cpp.PIXEL_F32_RGB_01, timing=vla_cpp.TIMING_PHASE)
                if not np.isfinite(actions).all(): raise ValueError("Non-finite policy output")
                if first is None:
                    first = actions.copy()
                repeat_max_abs = max(repeat_max_abs, float(np.max(np.abs(actions-first))))
            st = model.last_stats()
            result = {"model": str(args.model.resolve()), "device_requested": args.device,
                      "weight_dtype": args.weight_dtype, "camera_order": cameras,
                      "act_dtype": args.act_dtype, "flash_attn": args.flash_attn,
                      "warmup": args.warmup, "repeat_max_abs": repeat_max_abs,
                      "fusion_disabled": os.environ.get("VLA_PI05_NO_FUSE") == "1",
                      "tokens_per_image": int(cfg.n_img), "num_steps": int(cfg.num_steps),
                      "seed": args.seed, "prompt": prompt, "token_ids": tokens,
                      "denormalized": bool(cfg.denormalized),
                      "actions": actions[:, :cfg.real_action_dim].tolist(),
                      "timing_ms": {k: float(getattr(st, k)) for k in ("ms_total", "ms_vision", "ms_inference")}}
    if optimized_profile is not None:
        result["musa_optimized_profile"] = optimized_profile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(result['actions'])} x {len(state)} actions to {args.output}; cameras={cameras}")


if __name__ == "__main__":
    main()
