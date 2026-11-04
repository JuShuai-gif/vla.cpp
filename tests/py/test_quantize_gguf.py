"""Small files exercise the real GGUF writer/reader, not mocked tensor packing."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import gguf
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from quantize_gguf import convert, GGUFReader, make_plan, to_f32, unique_keys

QT = gguf.GGMLQuantizationType


def source(path, dtype=QT.F32, nonfinite=False):
    w = gguf.GGUFWriter(str(path), "pi05")
    w.add_string("pi05.note", "metadata survives 量化")
    w.add_array("pi05.values", [1.25, 2.5])
    rng = np.random.default_rng(17)
    for name in [f"vlm.blk.0.{op}.weight" for op in
                 ("attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate", "ffn_up", "ffn_down")] + [
                     "mm.proj.weight", "action_out_proj.weight", "vit.blk.0.fc1.weight", "vit.blk.0.fc2.weight"]:
        x = rng.normal(size=(16, 48 if name == "vit.blk.0.fc2.weight" else 64)).astype(np.float32)
        if nonfinite:
            x[0, 0] = np.nan
        if dtype == QT.F16:
            x = x.astype(np.float16)
        elif dtype == QT.BF16:
            x = (x.view(np.uint32) >> 16).astype(np.uint16)
        w.add_tensor(name, x, raw_dtype=dtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class QuantizeTest(unittest.TestCase):
    def test_roundtrip_and_error_bound_all_source_dtypes(self):
        for dtype in (QT.F32, QT.F16, QT.BF16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as root:
                src, dst = Path(root)/"src.gguf", Path(root)/"out.gguf"
                source(src, dtype)
                plan = convert(src, dst, "prefix-mlp")
                self.assertEqual(plan["quantized_tensors"], 3)
                original = {t.name: t for t in GGUFReader(str(src)).tensors}
                for t in GGUFReader(str(dst)).tensors:
                    a = original[t.name]
                    if "ffn_" in t.name:
                        self.assertEqual(t.tensor_type, QT.Q8_0)
                        value = to_f32(a)
                        decoded = gguf.quants.dequantize(t.data, t.tensor_type)
                        # Nearest rounding error plus FP16 scale error.
                        bound = np.max(np.abs(value)) / 254 + 0.002
                        self.assertLessEqual(float(np.max(np.abs(decoded - value))), bound)
                    else:
                        np.testing.assert_array_equal(a.data, t.data)
                        self.assertEqual(a.tensor_type, t.tensor_type)
                self.assertEqual(json.loads(Path(str(dst)+".quant.json").read_text())["status"], "verified")
                with self.assertRaisesRegex(ValueError, "must not exist"):
                    convert(src, dst)
                with self.assertRaisesRegex(ValueError, "already quantized"):
                    convert(dst, Path(root)/"again.gguf")

    def test_recipes_maps_and_block_alignment(self):
        with tempfile.TemporaryDirectory() as root:
            src = Path(root)/"src.gguf"
            source(src)
            r = GGUFReader(str(src))
            self.assertEqual(make_plan(r)["quantized_tensors"], 7)
            p = make_plan(r, "vision")
            self.assertEqual(p["quantized_tensors"], 1)
            self.assertIn("not divisible", p["tensors"][-1]["reason"])
            for overrides, message in [
                ({"vlm.blk.0.attn_q.weight": "Q4_0"}, "fusion groups"),
                ({"mm.proj.weight": "Q8_0"}, "protected"),
                ({"absent": "Q8_0"}, "unknown"),
                ({"vit.blk.0.fc2.weight": "Q8_0"}, "block-aligned"),
                ({"vlm.blk.0.attn_q.weight": []}, "map values")]:
                with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                    make_plan(r, "map", overrides=overrides)
            with self.assertRaisesRegex(ValueError, "zero tensors"):
                make_plan(r, "expert")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                json.loads('{"same":"Q8_0","same":"Q4_0"}', object_pairs_hook=unique_keys)
            dst = Path(root)/"dry.gguf"
            self.assertEqual(convert(src, dst, dry_run=True)["status"], "planned")
            self.assertFalse(dst.exists())
            with self.assertRaisesRegex(ValueError, "distinct"):
                convert(src, src)

    def test_failed_conversion_never_publishes(self):
        with tempfile.TemporaryDirectory() as root:
            src, dst = Path(root)/"src.gguf", Path(root)/"out.gguf"
            source(src, nonfinite=True)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                convert(src, dst)
            self.assertEqual([p.name for p in Path(root).iterdir()], ["src.gguf"])


if __name__ == "__main__":
    unittest.main()
