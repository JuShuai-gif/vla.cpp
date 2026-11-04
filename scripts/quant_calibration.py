# Copyright 2026 VinRobotics
# SPDX-License-Identifier: Apache-2.0
"""Activation-weighted block scale search, preserving standard GGUF formats.

Uses diagonal input second moments. This is not full GPTQ or AWQ; it changes
neither activation arithmetic nor the packed Q4_0/Q8_0 runtime contract.
"""
import gguf
import numpy as np


def weighted_pack(values, kind, importance):
    if kind not in ("Q4_0", "Q8_0"):
        raise ValueError("Weighted scale search supports Q4_0 and Q8_0")
    values = np.asarray(values, dtype=np.float32)
    importance = np.asarray(importance, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] % 32 or importance.shape != (values.shape[1],)
            or not np.isfinite(values).all() or not np.isfinite(importance).all()
            or np.any(importance < 0) or not np.any(importance > 0)):
        raise ValueError("Invalid values or activation second moments")
    qt = getattr(gguf.GGMLQuantizationType, kind)
    packed = gguf.quants.quantize(values, qt)
    width = 18 if kind == "Q4_0" else 34
    blocks = packed.reshape(-1, width)
    x = values.reshape(-1, 32)
    # Global normalization keeps the objective finite and preserves its minimum.
    weights = (importance / importance.max()).reshape(-1, 32)
    before = after = 0.0
    changed = 0
    for start in range(0, len(x), 4096):
        stop = min(start + 4096, len(x))
        original = blocks[start:stop].copy()
        v = x[start:stop]
        h = weights[np.arange(start, stop) % len(weights)]
        decoded = gguf.quants.dequantize(original, qt).reshape(-1, 32)
        if not np.isfinite(decoded).all():
            raise ValueError("Nonfinite initial packed scales")
        best_error = np.sum(h * (v.astype(np.float64) - decoded)**2, axis=1)
        before += float(best_error.sum())
        best = original.copy()
        base = original[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
        for factor in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2):
            scale = base * factor
            for _ in range(3):
                # Score the actual FP16 scale and integer codes consumed by GGML.
                scale = scale.astype(np.float16).astype(np.float32)
                inv = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale != 0)
                if kind == "Q4_0":
                    code = np.clip(np.floor(v * inv[:, None] + 8.5), 0, 15).astype(np.uint8)
                    signed = code.astype(np.int16) - 8
                    encoded = code[:, :16] | (code[:, 16:] << 4)
                else:
                    signed = np.clip(np.rint(v * inv[:, None]), -127, 127).astype(np.int8)
                    encoded = signed.view(np.uint8)
                reconstructed = signed.astype(np.float32) * scale[:, None]
                error = np.sum(h * (v.astype(np.float64) - reconstructed)**2, axis=1)
                improve = error < best_error
                if np.any(improve):
                    candidate = np.concatenate((scale.astype(np.float16).view(np.uint8).reshape(-1, 2), encoded), axis=1)
                    best[improve] = candidate[improve]
                    best_error[improve] = error[improve]
                numerator = np.sum(h * v * signed, axis=1)
                denominator = np.sum(h * signed.astype(np.float64)**2, axis=1)
                scale = np.divide(numerator, denominator, out=scale.astype(np.float64), where=denominator > 0).astype(np.float32)
        changed += int(np.count_nonzero(np.any(best != original, axis=1)))
        blocks[start:stop] = best
        after += float(best_error.sum())
    assert after <= before * (1 + 1e-12)
    return packed, dict(method="activation-weighted-block-scale-search", objective="diagonal-input-second-moment",
                        normalized_weighted_sse_before=before, normalized_weighted_sse_after=after,
                        changed_blocks=changed, total_blocks=len(x))
