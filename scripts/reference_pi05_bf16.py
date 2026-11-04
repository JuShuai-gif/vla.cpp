#!/usr/bin/env python3
# Copyright 2026 VinRobotics
# SPDX-License-Identifier: Apache-2.0
"""Original checkpoint BF16 forward using installed LeRobot; no quantized kernels.

The pruned variant uses its checkpoint layer count and attention/hard token
selection. Inputs are frozen by prepare_pi05_bf16_inputs.py, independently of
native inference. Source vision/norm FP32 exceptions remain as saved.
"""
import argparse,hashlib,json,math,types,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
import lerobot.policies.pi05.modeling_pi05 as pi
from lerobot.policies.pi05.configuration_pi05 import PI05Config

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--samples',type=int,default=3)
    ap.add_argument('--device',default='musa')
    a=ap.parse_args()
    if a.output.exists() or a.output.with_suffix('.actions.npz').exists():raise ValueError('Output already exists')
    cfg_path=a.checkpoint/'config.json';raw=json.loads(cfg_path.read_text())
    meta=json.loads(a.inputs.read_text());rows=meta['rows'][:a.samples]
    if len(rows)!=a.samples or a.samples<1:raise ValueError('Insufficient frozen inputs')
    if raw['dtype']!='bfloat16' or raw.get('use_relative_actions') or raw.get('rtc_config'):
        raise ValueError('Expected original BF16 absolute-action non-RTC checkpoint')
    if raw['num_inference_steps']!=meta['num_steps']:raise ValueError('Denoise steps differ')
    if [k for k,v in raw['input_features'].items() if v['type']=='VISUAL']!=meta['camera_order']:
        raise ValueError('Camera order differs')
    torch.set_num_threads(4)
    fields=('paligemma_variant','action_expert_variant','chunk_size','n_action_steps','max_state_dim',
            'max_action_dim','num_inference_steps','min_period','max_period','image_resolution')
    cfg=PI05Config(**{k:raw[k] for k in fields},dtype='bfloat16',device=a.device,compile_model=False)
    state=load_file(str(a.checkpoint/'model.safetensors'),device='cpu')
    if any(not k.startswith('model.') for k in state):raise ValueError('Unknown checkpoint namespace')
    state={k.removeprefix('model.'):v for k,v in state.items()}
    prefix='paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.'
    layers=len({int(k[len(prefix):].split('.')[0]) for k in state if k.startswith(prefix)})
    if layers<1:raise ValueError('No vision layers')
    # The local generic LeRobot constructor defaults to 27 vision layers.
    # Instantiate the actual checkpoint structure before strict state loading.
    constructor=pi.PaliGemmaForConditionalGenerationWithPiGemma
    def checkpoint_vision(config):
        config.vision_config.num_hidden_layers=layers
        return constructor(config)
    pi.PaliGemmaForConditionalGenerationWithPiGemma=checkpoint_vision
    try:
        with torch.device('meta'):model=pi.PI05Pytorch(cfg)
    finally:pi.PaliGemmaForConditionalGenerationWithPiGemma=constructor
    embedding='paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
    head='paligemma_with_expert.paligemma.lm_head.weight'
    aliases=[]
    for dst,src in ((embedding,head),(head,embedding)):
        if dst not in state and src in state:state[dst]=state[src];aliases.append([dst,src])
    model.load_state_dict(state,strict=True,assign=True)
    del state
    # Nonpersistent position/rope buffers are absent from safetensors. Rebuild
    # their small owning modules on CPU with the original HF initialization.
    rebuilt_buffers=[]
    for name,module in model.named_modules():
        pending={k:v for k,v in module.named_buffers(recurse=False) if v.is_meta}
        if not pending:continue
        if not hasattr(module,"config"):raise ValueError(f"Unknown meta-buffer owner {name}")
        fresh=type(module)(module.config)
        replacements=dict(fresh.named_buffers(recurse=False))
        for key,old in pending.items():
            value=replacements[key]
            if value.is_meta or value.shape!=old.shape:raise ValueError(f"Buffer reconstruction differs: {name}.{key}")
            # The outer constructor casts rotary buffers with the BF16 model.
            value=value.to(dtype=old.dtype)
            module.register_buffer(key,value,persistent=key not in module._non_persistent_buffers_set)
            if getattr(module,"original_inv_freq",None) is old:module.original_inv_freq=value
            rebuilt_buffers.append(name+"."+key)
        del fresh
    if any(p.is_meta for p in model.parameters()):raise ValueError("Unloaded meta parameters")
    print("Strict checkpoint loaded; initialized buffers",rebuilt_buffers,flush=True)
    parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters()))
    if parameter_dtypes.get('torch.bfloat16',0)==0:raise ValueError('No BF16 weights loaded')
    keep=int(raw.get('vision_token_prune',0))
    prune_ids=[]
    if keep:
        if raw.get('vision_token_score')!='attention' or raw.get('vision_token_sampling')!='hard':
            raise ValueError('Unsupported pruning contract')
        vlm=model.paligemma_with_expert.paligemma.model
        vlm.vision_tower.config._attn_implementation='eager'
        scores={}
        def capture_attention(module,args,out):
            if out[1] is None:raise ValueError('Missing eager vision attention weights')
            scores['last']=out[1]
        vlm.vision_tower.vision_model.encoder.layers[-1].self_attn.register_forward_hook(capture_attention)
        def embed_pruned(self,image):
            out=vlm.vision_tower(image.float(),return_dict=True)
            score=scores.pop('last').mean(1).mean(1)
            ids=score.topk(keep,dim=-1,sorted=False).indices.sort(-1).values
            selected=out.last_hidden_state.gather(1,ids[:,:,None].expand(-1,-1,out.last_hidden_state.shape[-1]))
            prune_ids.append(ids.cpu().numpy())
            return vlm.multi_modal_projector(selected)
        model.paligemma_with_expert.embed_image=types.MethodType(embed_pruned,model.paligemma_with_expert)
    model.eval().to(a.device)
    observed={}
    def linear_hook(name):
        def hook(module,args,out):
            observed[name]={'input':str(args[0].dtype),'weight':str(module.weight.dtype),'output':str(out.dtype)}
        return hook
    handles=[m.register_forward_hook(linear_hook(name)) for name,m in model.named_modules() if isinstance(m,torch.nn.Linear)]
    post=json.loads((a.checkpoint/'policy_postprocessor.json').read_text())
    step=[s for s in post['steps'] if s['registry_name']=='unnormalizer_processor']
    if len(step)!=1:raise ValueError('Missing original action unnormalizer')
    with safe_open(a.checkpoint/step[0]['state_file'],framework='np') as f:
        q01=f.get_tensor('action.q01').reshape(-1);q99=f.get_tensor('action.q99').reshape(-1)
    if q01.shape!=(7,) or q99.shape!=(7,):raise ValueError('Expected seven action dimensions')
    results=[];timings=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            path=a.inputs.parent/row['file']
            if sha(path)!=row['sha256']:raise ValueError('Frozen input checksum mismatch')
            with np.load(path,allow_pickle=False) as f:
                images=torch.from_numpy(f['images'].copy()).permute(0,3,1,2).to(a.device)*2-1
                tokens=torch.from_numpy(f['tokens'].astype(np.int64))[None].to(a.device)
                noise=torch.from_numpy(f['noise'].copy())[None].to(a.device)
            started=time.perf_counter()
            normalized=model.sample_actions([im[None] for im in images],
                [torch.ones(1,dtype=torch.bool,device=a.device) for _ in images],tokens,
                torch.ones_like(tokens,dtype=torch.bool),noise=noise)
            value=normalized.float().cpu().numpy()[0]
            timings.append((time.perf_counter()-started)*1000)
            if value.shape!=(50,32) or not np.isfinite(value).all():raise ValueError('Invalid BF16 actions')
            action=np.zeros_like(value)
            action[:,:7]=(value[:,:7]+1)*(q99-q01)*np.float32(.5)+q01
            results.append(action)
            print(f'{i+1}/{len(rows)} BF16 actions {timings[-1]:.1f} ms',flush=True)
    for h in handles:h.remove()
    sources=[Path(__file__),Path(pi.__file__),cfg_path,a.checkpoint/'model.safetensors',a.inputs,
             a.checkpoint/'policy_postprocessor.json',a.checkpoint/step[0]['state_file']]
    import lerobot.policies.pi_gemma as pg
    sources.append(Path(pg.__file__))
    report=dict(schema_version=1,reference_kind='source-framework-bf16',precision='original-checkpoint-bf16',
        source_model=meta['source_model'],checkpoint=str(a.checkpoint.resolve()),source_sha256={str(p):sha(p) for p in sources},
        torch_version=torch.__version__,parameter_dtypes=parameter_dtypes,linear_dtypes=observed,
        vision_layers=layers,vision_token_prune=keep,aliases=aliases,rebuilt_buffers=rebuilt_buffers,num_steps=cfg.num_inference_steps,
        input_sha256=[r['input_sha256'] for r in rows],noise_sha256=[r['noise_sha256'] for r in rows],
        token_sha256=[r['token_sha256'] for r in rows],task_sha256=[r['task_sha256'] for r in rows],
        episodes=[r['episode'] for r in rows],action_half_range=((q99-q01)*.5).tolist(),
        units={'joints':'unconfirmed','gripper':'unconfirmed','confirmed':False},
        reference_forward_ms=timings,scope='accuracy reference only; not a performance baseline',
        pruning_adaptation='checkpoint layer count; final attention mean over heads/queries, top-k then spatial sort' if keep else None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.output.with_suffix('.actions.npz'),actions=np.stack(results))
    if prune_ids:np.savez(a.output.with_suffix('.prune.npz'),ids=np.stack(prune_ids))
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
