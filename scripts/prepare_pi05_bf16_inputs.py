#!/usr/bin/env python3
# Copyright 2026 VinRobotics
# SPDX-License-Identifier: Apache-2.0
"""Freeze public-API inputs for an independent checkpoint BF16 forward."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from pi05_gguf import GGUFReader
from vla_cpp.policy import _inspect,prepare_prompt
from vla_cpp.pi05_preprocess import prepare_rgb

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--corpus',type=Path,required=True)
    ap.add_argument('--samples',type=int,default=30)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--split',choices=('development','heldout'),default='development')
    args=ap.parse_args()
    cfg,spec,reader=_inspect(args.config,'baseline')
    corpus=json.loads(args.corpus.read_text())
    rows=[r for r in corpus['rows'] if r['split']==args.split][:args.samples]
    if args.samples<1 or len(rows)!=args.samples:raise ValueError('Insufficient selected-split observations')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(cfg['seed']);items=[]
    for i,row in enumerate(rows):
        path=args.corpus.parent/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Observation hash differs')
        with np.load(path,allow_pickle=False) as obs:
            state=np.asarray(obs['state'],np.float32)
            images=np.stack([prepare_rgb(obs[k.rsplit('.',1)[-1]],spec.image_size) for k in spec.camera_order])
        prompt,tokens=prepare_prompt(reader,row['task'],state)
        tokens=np.asarray(tokens,np.int32)
        noise=rng.standard_normal(50*32).astype(np.float32).reshape(50,32)
        file=args.output_dir/f'{i:04d}.npz'
        np.savez(file,images=images,tokens=tokens,noise=noise,state=state)
        items.append(dict(file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                          input_sha256=row['sha256'],task_sha256=hashlib.sha256(row['task'].encode()).hexdigest(),
                          token_sha256=digest(tokens),noise_sha256=digest(noise),episode=row['episode'],prompt=prompt))
    meta=dict(schema_version=1,source_model=cfg['model'],camera_order=list(spec.camera_order),num_steps=spec.num_steps,
              image_size=spec.image_size,split=args.split,rows=items,seed=cfg['seed'],
              preprocessing='production prepare_rgb F32 RGB [0,1], prepare_prompt; no native inference',
              source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output_dir/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(args.output_dir,len(items),flush=True)
if __name__=='__main__':main()
