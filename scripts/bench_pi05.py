#!/usr/bin/env python3
"""Repeatable resident-model benchmark for full and vision-pruned pi05 GGUFs.

Uses the same synthetic inputs as validate_pi05.py. Loading and warmup are
excluded; all model cameras, action chunks and denoising steps are retained.
"""
import argparse
import ctypes
import hashlib
import platform
import subprocess
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model', type=Path, required=True)
ap.add_argument('--out-dir', type=Path, required=True)
ap.add_argument('--weight-dtype', choices=('f32', 'f16', 'bf16'), default='f32')
ap.add_argument('--act-dtype', choices=('f32', 'f16'), default='f32')
ap.add_argument('--flash-attn', action='store_true')
ap.add_argument('--split-phases', action='store_true', help='MUSA diagnostic: synchronize prefix/denoising separately')
ap.add_argument('--warmup', type=int, default=1)
ap.add_argument('--runs', type=int, default=5)
ap.add_argument('--reference', type=Path, help='Actions .npy from validate_pi05.py or this benchmark')
opt = ap.parse_args()
if opt.runs < 1 or opt.warmup < 0:
    ap.error('runs must be positive and warmup nonnegative')
if opt.split_phases:
    os.environ['VLA_PI05_SPLIT_PHASES'] = '1'
repo = Path(__file__).resolve().parents[1]
os.environ.setdefault('VLA_LIBRARY', str(repo/'build-musa-m1000/libvla.so'))
os.environ.setdefault('VLA_N_THREADS', '4')
os.environ.setdefault('VLA_DEVICE', '0')
sys.path.insert(0, str(repo/'bindings/python'))
import vla_cpp

# Do not record CPU fallback as an accelerator benchmark.
if os.environ['VLA_DEVICE'].lower() != 'cpu':
    count = vla_cpp._lib_handle().ggml_backend_cuda_get_device_count
    count.restype = ctypes.c_int
    device = int(os.environ['VLA_DEVICE'])
    if device < 0 or device >= count():
        raise RuntimeError('Requested MUSA device is unavailable; refusing CPU fallback benchmark')

def command_output(args):
    try:
        result = subprocess.run(args, cwd=repo, text=True, capture_output=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None

library = Path(os.environ['VLA_LIBRARY']).resolve()
metadata = {
    'source_commit': command_output(['git', 'rev-parse', 'HEAD']),
    'source_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=repo)).hexdigest(),
    'library': str(library),
    'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
    'core_sha256': hashlib.sha256((library.parent/'libvla_core.so').read_bytes()).hexdigest() if (library.parent/'libvla_core.so').exists() else None,
    'backend_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((library.parent/'bin' if (library.parent/'bin').exists() else library.parent).glob('libggml-*.so'))},
    'platform': platform.platform(),
    'power_profile': command_output(['powerprofilesctl', 'get']),
    'environment': {k: os.environ.get(k) for k in (
        'VLA_DEVICE', 'VLA_N_THREADS', 'VLA_PI05_NO_FUSE', 'VLA_PI05_NO_PROJ_FUSE',
        'VLA_MUSA_GEMMA_ATTN', 'VLA_MUSA_VISION_EPILOGUE', 'VLA_MUSA_NO_VECTOR_PACK', 'VLA_MUSA_VISION_SDPA', 'VLA_PI05_VISION_QKV',
        'VLA_PI05_INPUT_CACHE', 'VLA_MUSA_TRACE_GEMM', 'VLA_MUSA_FLATTEN_GEMM', 'VLA_MUSA_ATTN_DIRECT_OUT', 'VLA_MUSA_VISION_PACK', 'VLA_PI05_PIN_MAIN_INPUT', 'VLA_PI05_HOST_INPUT', 'VLA_PI05_ASYNC_VISION', 'VLA_PI05_VISION_DIRECT', 'VLA_MUSA_PROFILE_OPS', 'VLA_PI05_DUMP_GRAPH', 'VLA_PI05_DUMP_PRUNE',
        'GGML_CUDA_DISABLE_GRAPHS', 'GGML_MUSA_DISABLE_GRAPHS', 'VLA_PI05_SPLIT_PHASES',
        'VLA_PI05_HOST_PRUNE', 'VLA_PI05_NO_PRUNE_FLASH', 'VLA_PI05_NO_VISION_BATCH',
        'VLA_PI05_NO_SEPARATE_GRAPHS', 'VLA_PI05_KV_REUSE', 'VLA_PI05_NO_KV_REUSE', 'VLA_MUSA_F32_ATTN_TILE')},
}

opt.out_dir.mkdir(parents=True, exist_ok=True)
runtime = opt.out_dir/'runtime.json'
runtime.write_text(json.dumps({'runtime': {'weight_dtype': opt.weight_dtype,
    'act_dtype': opt.act_dtype, 'flash_attn': opt.flash_attn,
    'mm_prec': 'default' if opt.act_dtype == 'f16' else 'f32'}}))
with vla_cpp.load(str(opt.model), config_path=str(runtime)) as m:
    cfg = m.config
    # These converted checkpoints both have two 224x224 cameras.
    y, x, c = np.indices((224, 224, 3))
    images = [np.ascontiguousarray(((x+2*y+40*c+17*v)&255).astype(np.uint8)) for v in range(2)]
    state = np.zeros(cfg.max_state_dim, dtype=np.float32)
    state[:cfg.real_state_dim] = np.arange(1, cfg.real_state_dim+1)*0.01
    noise = (np.arange(cfg.n_suffix*cfg.max_action_dim, dtype=np.uint64)*2654435761%1000).astype(np.float32)*np.float32(.001)-np.float32(.5)
    rows = []
    first = None
    drift = 0.0
    for i in range(opt.warmup + opt.runs):
        start = time.perf_counter()
        actions = m.predict(images, tokens=[1,100,200,300,400,2], state=state,
                            noise=noise, timing=vla_cpp.TIMING_PHASE)
        wall = (time.perf_counter()-start)*1000
        if not np.isfinite(actions).all():
            raise RuntimeError('non-finite actions')
        if first is None:
            first = actions.copy()
        drift = max(drift, float(np.max(np.abs(actions-first))))
        st = m.last_stats()
        row = {k: float(getattr(st, k)) for k in ('ms_total', 'ms_vision', 'ms_inference')}
        row['wall_ms'] = wall
        print(json.dumps({'iteration': i, 'warmup': i < opt.warmup, **row}), flush=True)
        if i >= opt.warmup:
            rows.append(row)
    np.save(opt.out_dir/'actions.npy', actions)
    report = {'metadata': metadata, 'split_phases': opt.split_phases, 'flash_attn': opt.flash_attn, 'model': str(opt.model), 'dtype': opt.weight_dtype, 'act_dtype': opt.act_dtype, 'steps': cfg.num_steps,
              'synthetic_inputs': True, 'camera_count': len(images), 'text_token_count': 6,
              'chunk': cfg.n_suffix, 'vision_tokens_per_camera': cfg.n_img,
              'no_fuse': os.environ.get('VLA_PI05_NO_FUSE', '0'),
              'warmup': opt.warmup, 'runs': rows, 'repeat_max_abs': drift,
              'median_ms': {k: float(np.median([r[k] for r in rows])) for k in rows[0]},
              'p95_wall_ms': float(np.percentile([r['wall_ms'] for r in rows], 95))}
    if opt.reference:
        a = actions[:, :cfg.real_action_dim]
        ref = np.load(opt.reference)[:, :cfg.real_action_dim]
        d = a-ref
        report['error_vs_reference'] = {'max_abs': float(np.max(np.abs(d))),
            'rmse': float(np.sqrt(np.mean(d*d))),
            'relative_peak': float(np.max(np.abs(d))/max(float(np.max(np.abs(ref))), 1e-12))}
    (opt.out_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
