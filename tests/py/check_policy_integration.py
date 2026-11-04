"""Manual real-GGUF check against a saved run_pi05.py action reference; no robot I/O."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import numpy as np
from PIL import Image
from vla_cpp import load_policy, inspect_policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--robot-config',type=Path,help='Also exercise the robot_runtime public backend bridge')
    args=parser.parse_args()
    spec=inspect_policy(args.config)
    base=Path(spec.checkpoint).parent
    state=np.asarray(json.loads((base/'model.stats.json').read_text())['observation.state']['mean'],np.float32)
    frames={role:np.asarray(Image.open(base/f'validation/{role}-480x640.png').convert('RGB'))
            for role in ('top','wrist')}
    reference=json.loads(args.reference.read_text())
    assert reference['seed']==0 and reference['weight_dtype']=='f16' and reference['act_dtype']=='f32'
    assert reference['model']==spec.checkpoint
    expected=np.asarray(reference['actions'],np.float32)
    if args.robot_config:
        from robot_runtime.backends import load_backend
        from robot_runtime.setup.config import load_config
        root=args.robot_config.resolve().parents[1]
        cfg=load_config(args.robot_config)
        params=cfg['profiles'][spec.variant]
        assert (root/params['policy_config']).resolve()==args.config.resolve()
        assert params['task']=='pick up the cube'
        model=load_backend('vla_cpp').create_replay_model(root,cfg,spec.variant)
        images=frames
    else:
        model=load_policy(args.config,task='pick up the cube')
        images={key:frames[key.rsplit('.',1)[-1]] for key in spec.camera_order}
    with closing(model):
        model.warmup(images,state=state,runs=3)
        actions=model.predict(images,state=state,task='pick up the cube')
        np.testing.assert_allclose(actions,expected,rtol=0,atol=1e-5)
        first_timings=dict(model.last_predict_timings_ms)
        second=model.predict(images,state=state,task='pick up the cube')
        assert not np.array_equal(actions,second),'Predictions reused the same noise'
        assert actions.shape==(50,7) and np.isfinite(actions).all()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez(args.output_dir/f'{spec.variant}-observation.npz',state=state,**frames)
    np.save(args.output_dir/f'{spec.variant}-api-actions.npy',actions)
    report=dict(model=spec.checkpoint,variant=spec.variant,camera_order=spec.camera_order,
        caller='robot_runtime' if args.robot_config else 'vla_cpp',
        action_shape=list(actions.shape),reference=str(args.reference),
        max_abs_reference_error=float(np.max(np.abs(actions-expected))),
        fresh_noise=True,warmup_restored_rng=True,first_predict_timings_ms=first_timings,
        timing_scope='single observation integration check; CPU wall times, not a performance benchmark',
        scope='saved synthetic RGB images and checkpoint state mean; no hardware or closed-loop task validation')
    (args.output_dir/f'{spec.variant}-api.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
