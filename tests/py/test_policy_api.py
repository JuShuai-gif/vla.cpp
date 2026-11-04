"""Policy lifecycle, validation and stochastic scheduling without native hardware."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import vla_cpp
from vla_cpp import policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        for name in ('model.gguf','libvla.so'):(root/name).touch()
        self.config=dict(schema_version=1,model='model.gguf',library='libvla.so',device='cpu',
            source_contract=dict(action_names=['joint'],action_space='absolute_joint',joint_unit='degrees',
                                 gripper_unit='mm',verified=False))
        self.path=root/'policy.json'; self.path.write_text(json.dumps(self.config))

    def test_config_paths_and_invalid_options(self):
        cfg=policy._config(self.path)
        self.assertEqual(cfg['model'],str(self.path.parent/'model.gguf'))
        for invalid in ({'seed':True},{'threads':0},{'runtime':{'act_dtype':'int8'}},
                        {'typo':1},{'source_contract':{}},{'device':'-1'}):
            self.path.write_text(json.dumps(dict(self.config,**invalid)))
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):policy.inspect_policy(self.path)

    def test_native_session_rng_warmup_and_cleanup(self):
        cfg=policy._config(self.path)
        spec=policy.PolicySpec(cfg['model'],('wrist','top'),7,7,50,
            tuple([f'joint_{i}.pos' for i in range(1,7)]+['gripper.pos']),
            'absolute_joint','degrees','mm',False,4,3,'pruned')
        native=Mock()
        native.config=SimpleNamespace(real_state_dim=7,real_action_dim=7,n_suffix=50,num_steps=3,
                                      max_state_dim=32,max_action_dim=32,denormalized=True)
        def predict(images,**kwargs):return kwargs['noise'].reshape(50,32).copy()
        native.predict.side_effect=predict
        fake_lib=SimpleNamespace(vla_model_load=Mock(return_value=1))
        with patch.object(policy,'_inspect',return_value=(cfg,spec,object())), \
             patch.object(policy,'prepare_prompt',return_value=('prompt',[1,2])), \
             patch.object(policy,'_process_settings',None), \
             patch('ctypes.CDLL',return_value=fake_lib), \
             patch('vla_cpp._ffi.load_library',return_value=fake_lib), \
             patch.object(vla_cpp,'Model',return_value=native):
            images={name:np.zeros((4,4,3),np.uint8) for name in spec.camera_order}
            with vla_cpp.load_policy(self.path,task='test') as model:
                with self.assertRaisesRegex(RuntimeError,'one resident'):vla_cpp.load_policy(self.path)
                model.warmup(images,state=[0]*7,runs=3)
                first=model.predict(images,state=[0]*7)
                expected=np.random.default_rng(0).standard_normal(50*32).astype(np.float32).reshape(50,32)[:,:7]
                np.testing.assert_array_equal(first,expected)
                self.assertFalse(np.array_equal(first,model.predict(images,state=[0]*7)))
                self.assertEqual(first.shape,(50,7))
                with self.assertRaises(ValueError):model.predict({'top':images['top']},state=[0]*7)
                with self.assertRaises(ValueError):model.predict(dict(images,wrist=np.zeros((4,4,3),np.float32)),state=[0]*7)
                with self.assertRaises(ValueError):model.predict(images,state=[0]*7,task='')
            native.close.assert_called_once()
            model.close()
            with self.assertRaisesRegex(RuntimeError,'closed'):model.predict(images,state=[0]*7)

    def test_load_failure_releases_session(self):
        cfg=policy._config(self.path)
        spec=SimpleNamespace(variant='full')
        with patch.object(policy,'_inspect',return_value=(cfg,spec,object())), \
             patch.object(policy,'_process_settings',None), patch('ctypes.CDLL',side_effect=OSError('broken library')):
            for _ in range(2):
                with self.assertRaisesRegex(OSError,'broken library'):vla_cpp.load_policy(self.path)


if __name__=='__main__':unittest.main()
