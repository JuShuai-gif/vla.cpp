"""Policy lifecycle, validation and stochastic scheduling without native hardware."""
import json
import os
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

    def test_named_precision_resolution_and_no_silent_fallback(self):
        (self.path.parent/'profiles.json').write_text('{}')
        cfg=dict(self.config, device='0', optimization_profile='profiles.json',
                 precision_profiles={'fp16-activations': {'runtime': {'act_dtype': 'f16'}},
                                     'q8': {'model': 'candidate.gguf', 'manifest': 'candidate.quant.json'}},
                 default_precision='fp16-activations')
        self.path.write_text(json.dumps(cfg))
        selected=policy._config(self.path)
        self.assertEqual(selected['runtime']['act_dtype'],'f16')
        self.assertEqual(selected['precision_status'],'experimental')
        self.assertEqual(policy._config(self.path,'baseline')['runtime']['act_dtype'],'f32')
        with self.assertRaisesRegex(ValueError,'Unknown precision'):
            policy._config(self.path,'fp4-typo')
        with self.assertRaisesRegex(ValueError,'Missing model'):
            policy._config(self.path,'q8')
        cfg['precision_profiles']['baseline']={}
        self.path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'reserved'):
            policy._config(self.path)

    def test_precision_library_override_keeps_original_baseline(self):
        alt=self.path.parent/"int8.so";alt.touch()
        optimization=self.path.parent/"int8-profiles.json";optimization.write_text("{}")
        cfg=dict(self.config,device="0",precision_profiles={"int8-test":{"library":"int8.so","optimization_profile":"int8-profiles.json"}})
        self.path.write_text(json.dumps(cfg))
        selected=policy._config(self.path,"int8-test")
        self.assertEqual(selected["library"],str(alt))
        self.assertEqual(selected["optimization_profile"],str(optimization))
        self.assertEqual(policy._config(self.path)["library"],str(self.path.parent/"libvla.so"))
        alt.unlink()
        with self.assertRaisesRegex(ValueError,"Missing library"):policy._config(self.path,"int8-test")
        self.assertEqual(policy._config(self.path,"baseline")["library"],str(self.path.parent/"libvla.so"))

    def test_quant_route_is_explicit_and_validated(self):
        cfg=dict(self.config,precision_profiles={"q8-blas":{"quant_route":"blas"}})
        self.path.write_text(json.dumps(cfg))
        self.assertEqual(policy._config(self.path,"q8-blas")["quant_route"],"blas")
        self.assertEqual(policy._config(self.path,"baseline")["quant_route"],"auto")
        cfg["precision_profiles"]["q8-blas"]["quant_route"]="typo"
        self.path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,"quant_route"):
            policy._config(self.path,"q8-blas")

    def test_load_f16_route_requires_f16_and_stays_opt_in(self):
        cfg=dict(self.config,precision_profiles={"q8-cached":{"quant_route":"load_f16"}})
        self.path.write_text(json.dumps(cfg))
        self.assertEqual(policy._config(self.path,"q8-cached")["quant_route"],"load_f16")
        self.assertEqual(policy._config(self.path,"baseline")["quant_route"],"auto")
        cfg["precision_profiles"]["q8-cached"]["runtime"]={"weight_dtype":"f32"}
        self.path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,"F16 resident"):
            policy._config(self.path,"q8-cached")

    def test_native_int8_requires_explicit_parameters_and_float_baseline(self):
        cfg=dict(self.config,precision_profiles={"native":{"quant_route":"native_int8"}})
        self.path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,"calibration manifest"):policy._config(self.path,"native")
        (self.path.parent/"int8.json").write_text("{}")
        cfg["precision_profiles"]["native"]["int8_parameters"]="int8.json"
        self.path.write_text(json.dumps(cfg))
        self.assertEqual(policy._config(self.path,"native")["quant_route"],"native_int8")
        cfg["precision_profiles"]["native"]["runtime"]={"act_dtype":"f16"}
        self.path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,"F32 activations"):policy._config(self.path,"native")

    def test_prompt_cache_reuses_metadata_but_not_state_or_task(self):
        def field(value):
            return SimpleNamespace(contents=Mock(return_value=value))
        reader=SimpleNamespace(tensors=[
            SimpleNamespace(name='state_q01',data=np.zeros(2,np.float32)),
            SimpleNamespace(name='state_q99',data=np.ones(2,np.float32))],
            fields={'pi05.norm_eps':field(1e-8),'pi05.tokenizer_max_length':field(64),
                    'pi05.tokenizer.hf_json':field('embedded-tokenizer')})
        tokenizer=SimpleNamespace(encode=lambda text:SimpleNamespace(ids=list(text.encode())))
        with patch.object(policy,'_tokenizer',return_value=tokenizer):
            a=policy.prepare_prompt(reader,'pick_up',[0.1,0.2])
            b=policy.prepare_prompt(reader,'put_down',[0.8,0.9])
            c=policy.prepare_prompt(reader,'pick_up',[0.1,0.2])
        self.assertEqual(a,c)
        self.assertNotEqual(a,b)
        self.assertIn('Task: pick up, State: 25 51;',a[0])
        reader.fields['pi05.tokenizer.hf_json'].contents.assert_called_once()
        with self.assertRaises(ValueError):policy.prepare_prompt(reader,'pick',[float('nan'),0])

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
            with patch.dict(os.environ, {"VLA_PI05_INT8_PREFIX":"1","VLA_PI05_INT8_PARAMETERS":"/unselected","VLA_LOAD_QUANT_F16":"1","GGML_MUSA_QUANT_ROUTE":"blas","GGML_CUDA_CUBLAS_COMPUTE_TYPE":"f16","GGML_MUSA_GRAPH_RECREATE":"1"}), vla_cpp.load_policy(self.path,task='test') as model:
                self.assertEqual(os.environ["GGML_MUSA_QUANT_ROUTE"],"auto")
                self.assertNotIn("VLA_LOAD_QUANT_F16",os.environ)
                self.assertNotIn("VLA_PI05_INT8_PARAMETERS",os.environ)
                self.assertNotIn("VLA_PI05_INT8_PREFIX",os.environ)
                self.assertNotIn("GGML_CUDA_CUBLAS_COMPUTE_TYPE",os.environ)
                self.assertNotIn("GGML_MUSA_GRAPH_RECREATE",os.environ)
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
