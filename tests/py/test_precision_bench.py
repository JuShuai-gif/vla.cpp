"""Paired report checks must reject changed inputs and nonfinite actions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from bench_pi05_precision import compare, cosine_quality

class PrecisionReportTests(unittest.TestCase):
    def test_cosine_gates_zero_norm_and_local_failures(self):
        x=np.ones((2,50,32),np.float32)
        self.assertTrue(cosine_quality(x,x)["offline_cosine_pass"])
        self.assertTrue(cosine_quality(x,x*2)["offline_cosine_pass"])
        zero=np.zeros_like(x)
        self.assertEqual(cosine_quality(zero,zero)["minimum"],1.)
        self.assertEqual(cosine_quality(zero,x)["minimum"],0.)
        y=x.copy();y[1,:,6]*=-1
        result=cosine_quality(x,y)
        self.assertFalse(result["offline_cosine_pass"])
        self.assertEqual(result["failed_chunk_dimensions"],[[1,6]])
        tiny=x.copy();tiny[:,:,6]=1e-4
        flipped=tiny.copy();flipped[:,:,6]*=-1
        self.assertTrue(cosine_quality(tiny,flipped)["offline_cosine_pass"])
        self.assertLess(cosine_quality(tiny,flipped)["minimum_by_dimension"][6],0)
        # One bad instant must fail even when the entire chunk passes.
        y=x.copy();y[1,49,6]*=-1
        result=cosine_quality(x,y)
        self.assertFalse(result["offline_cosine_pass"])
        self.assertEqual(result["failed_action_vectors"],[[1,49]])
        self.assertGreater(result["minimum_chunk"],0.99)
        # A small final-time perturbation crosses 0.999 but keeps chunk >0.999.
        y=x.copy();y[1,49,6]+=0.2
        result=cosine_quality(x,y)
        self.assertFalse(result["offline_cosine_pass"])
        self.assertEqual(result["failed_action_vectors"],[[1,49]])
        self.assertGreater(result["minimum_chunk"],0.999)
        y=x.copy();y[:,:,7:]=1e5
        self.assertTrue(cosine_quality(x,y)["offline_cosine_pass"])

    def test_report_written_before_process_failure_is_rejected(self):
        from run_precision_suite import execute
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=root/"report.json"
            code="from pathlib import Path;import sys;Path(sys.argv[1]).write_text('{}');sys.exit(6)"
            result=execute([sys.executable,"-c",code,str(report)],report,root/"failed.log")
            self.assertTrue(report.exists());self.assertEqual(result["status"],"process_failed")
            self.assertEqual(result["returncode"],6)
            result=execute([sys.executable,"-c","pass"],root/"missing.json",root/"missing.log")
            self.assertEqual(result["status"],"process_failed")

    def test_pairing_metrics_and_zero_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/"a.json";b=root/"b.json"
            report=dict(input_sha256=["i"],noise_sha256=["n"],task_sha256=["t"],episodes=[7],token_sha256=["tok"],
                        action_half_range=[1.0]*7,units={"confirmed":False},native_ms={"p50":100.0,"p95":120.0},predict_ms={"p50":110.0,"p95":130.0})
            a.write_text(json.dumps(report));b.write_text(json.dumps(report))
            x=np.zeros((1,50,32),np.float32);y=x.copy();y[:,:,:7]=0.125
            np.savez(a.with_suffix(".actions.npz"),actions=x);np.savez(b.with_suffix(".actions.npz"),actions=y)
            result=compare(a,b)
            self.assertTrue(result["tokens_paired"]);self.assertEqual(result["physical"]["mae"],[0.125]*7)
            self.assertEqual(result["action_increment_error"]["max_abs"],[0.0]*7)
            self.assertEqual(result["end_to_end_p95_reduction"],0)
            bf16=root/"bf16.json"
            ref=dict(report,reference_kind="source-framework-bf16")
            ref.pop("native_ms");ref.pop("predict_ms")
            bf16.write_text(json.dumps(ref))
            np.savez(bf16.with_suffix(".actions.npz"),actions=y)
            separate=compare(bf16,b,performance_reference=a)
            self.assertTrue(separate["offline_accuracy_pass"])
            self.assertEqual(separate["native_p50_reduction"],0.)
            self.assertIsNone(compare(bf16,b)["native_p50_reduction"])
            altered=dict(report);altered["episodes"]=[9];a.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,"Unpaired performance"):
                compare(bf16,b,performance_reference=a)
            a.write_text(json.dumps(report))
            for key in ("input_sha256","noise_sha256","token_sha256"):
                altered=dict(report);altered[key]=["changed"];b.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,"Unpaired"):compare(a,b)
            b.write_text(json.dumps(report));y[0,49,6]=np.nan;np.savez(b.with_suffix(".actions.npz"),actions=y)
            with self.assertRaisesRegex(ValueError,"Nonfinite"):compare(a,b)

if __name__=="__main__":unittest.main()
