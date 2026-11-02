#!/usr/bin/env python3
"""Repeatable resident-model benchmark for full and vision-pruned pi05 GGUFs.

Uses the same synthetic inputs as validate_pi05.py. Loading and warmup are
excluded; all model cameras, action chunks and denoising steps are retained.
"""
import argparse
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
ap.add_argument('--warmup', type=int, default=1)
ap.add_argument('--runs', type=int, default=5)
ap.add_argument('--reference', type=Path, help='Actions .npy from validate_pi05.py or this benchmark')
opt = ap.parse_args()
if opt.runs < 1 or opt.warmup < 0:
    ap.error('runs must be positive and warmup nonnegative')
repo = Path(__file__).resolve().parents[1]
os.environ.setdefault('VLA_LIBRARY', str(repo/'build-musa-m1000/libvla.so'))
os.environ.setdefault('VLA_N_THREADS', '4')
os.environ.setdefault('VLA_DEVICE', '0')
sys.path.insert(0, str(repo/'bindings/python'))
import vla_cpp

opt.out_dir.mkdir(parents=True, exist_ok=True)
runtime = opt.out_dir/'runtime.json'
runtime.write_text(json.dumps({'runtime': {'weight_dtype': opt.weight_dtype,
    'act_dtype': opt.act_dtype, 'flash_attn': False,
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
    report = {'model': str(opt.model), 'dtype': opt.weight_dtype, 'act_dtype': opt.act_dtype, 'steps': cfg.num_steps,
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
