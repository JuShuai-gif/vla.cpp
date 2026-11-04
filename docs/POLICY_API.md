# π0.5 Python policy API

`vla_cpp.inspect_policy(path)` reads the GGUF and deployment contract without
loading the native library or opening a device. `vla_cpp.load_policy(path)` owns
a resident model, tokenizer, image preprocessing buffers and noise generator.
The existing low-level `load()` and C ABI are unchanged. Robot drivers belong to
the caller; this package does not import `robot_runtime` or `flash_rt`.

```python
import numpy as np
from vla_cpp import inspect_policy, load_policy

spec = inspect_policy("policy.json")
with load_policy("policy.json", task="your training instruction") as policy:
    # Supply actual RGB uint8 HWC frames under the exact GGUF camera names.
    images = {key: np.zeros((480, 640, 3), np.uint8) for key in spec.camera_order}
    state = np.zeros(spec.state_dim, np.float32)  # raw checkpoint units
    policy.warmup(images, state=state, runs=3)
    actions = policy.predict(images, state=state)  # float32 [chunk_size, action_dim]
```

Example deployment JSON (paths resolve relative to this file):

```json
{
  "schema_version": 1,
  "model": "/models/model.gguf",
  "library": "/code/vla.cpp/build-musa-official/libvla.so",
  "optimization_profile": "/code/vla.cpp/scripts/pi05_musa_profiles.json",
  "device": "0",
  "seed": 0,
  "threads": 4,
  "runtime": {"weight_dtype": "f16", "act_dtype": "f32", "mm_prec": "f32", "flash_attn": true},
  "source_contract": {
    "action_names": ["joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos", "joint_5.pos", "joint_6.pos", "gripper.pos"],
    "action_space": "absolute_joint", "joint_unit": "degrees", "gripper_unit": "mm", "verified": false
  }
}
```

The API preserves raw state/action units; declaring units does not convert them.
`verified` records the deployment owner's check against training data and robot
configuration. It is not inferred from numeric ranges. `robot_runtime` refuses
live execution while it is false. Camera order, dimensions, denoising steps,
normalization and tokenizer are read from GGUF. The MUSA profile selects pruned
or full automatically, with the existing validated F16/F32/FA settings.

Each prediction draws fresh seeded noise. Warmup restores the generator state,
so the first production prediction is reproducible regardless of warmup count.
Inference and close are serialized. Use a context manager or call `close()`.
There is one resident policy per process. Switch native runtime/profile options
in a fresh process because ggml caches environment switches; do not mix this
API with independently managed low-level models in one process.

`last_predict_timings_ms` contains CPU wall intervals `preprocess_ms`,
`native_call_ms` (including ctypes marshalling and result copying), and
`postprocess_ms`. These are not pure GPU kernel timings.

For an existing native MUSA build, add `bindings/python` to the caller's Python
path and install the `policy` dependencies listed in `bindings/python/pyproject.toml`.
For example, a virtualenv `.pth` file can contain the absolute bindings path.
This avoids accidentally rebuilding/replacing the configured MUSA library with
the default CPU wheel. Torch is CPU-only here and is used solely for the original
F32 bilinear image resize. No Torch MUSA/CUDA runtime is required.

`scripts/run_pi05.py` remains compatible and shares preprocessing and prompt
tokenization with this API. Tests: `tests/py/test_policy_api.py` and
`tests/py/test_pi05_io.py`. For the Piper adapter and device setup, see
`robot_runtime/docs/VLA_CPP.md` in the adjacent repository.


## Precision profiles

Both public functions accept `precision=None` as a keyword argument:

```python
spec = inspect_policy("policy.json", precision="q8-prefix-mlp")
with load_policy("policy.json", precision="q8-prefix-mlp", task="...") as policy:
    actions = policy.predict(images, state=state)
```

Policy JSON may declare `default_precision` (otherwise `baseline`) and
`precision_profiles`. Each named entry accepts `model`, `manifest`,
`runtime` overrides, `quant_route` (`auto`, `blas` or `load_f16`) and `status` (`experimental` by default; `validated`
only after deployment acceptance). The reserved baseline uses the original
model/runtime configuration. All relative paths resolve beside the policy JSON.
Unknown names and missing selected assets fail explicitly. Quantized GGUF files
require the converter's verified manifest; load checks its SHA256 before native
allocation. Inspection checks tensor names, shapes and types without GPU I/O.

`PolicySpec.precision`, `precision_status`, `quant_route` and `weight_types` describe the
selection; resolved parameters separately report on-disk weight types,
floating resident weight dtype, the quantized GEMM residency rule and activation
dtype. With `auto` or `blas`, a packed Q8 tensor stays Q8 even when
`runtime.weight_dtype` is `f16`.

The opt-in `load_f16` route requires `runtime.weight_dtype: "f16"`. It decodes
quantized GEMM weights once at load into F16 resident tensors, using the existing
floating GEMM and fusion paths. Explicitly requested packed tensors keep their
packed type. `resolved_parameters.quantized_gemm_resident_dtype` reports `f16`
for this route and `packed` for the default rule; `weight_types_on_disk` still
describes the original file. This trades the packed GPU-memory saving for removal
of repeated dequantization. It is storage quantization with F16 computation, not
an INT8 compute speedup. Load time includes the one-time conversion; model output
accuracy and steady-state performance still require separate validation. It does
not change the baseline or promote an experimental profile to validated.

Different native options require a
fresh process. robot_runtime forwards its precision parameter and refuses
experimental profiles during live execution; offline loading remains available.

### Reproducible offline quantization experiments

`calibrate_pi05.py` reads only the episode-separated calibration split produced by
`prepare_quant_corpus.py`. It retains GEMM inputs and records per-channel maximum
magnitude and sums of squares, including separate occurrences at each denoise
step. Its timing and memory use are diagnostic, not performance measurements.
The directory must be new; raw reports and actions are retained.

```sh
python scripts/calibrate_pi05.py --config pruned-policy.json \
  --corpus validation/quant/data/real-v1/manifest.json --samples 120 \
  --output-dir validation/quant/calibration/pruned-120
python scripts/quantize_gguf.py --in original.gguf --out calibrated-q4.gguf \
  --recipe prefix-mixed --type Q8_0 \
  --importance validation/quant/calibration/pruned-120/importance.json
```

Calibration uses a search over standard Q4_0/Q8_0 block scales, weighted by input
second moments. It does not implement full GPTQ or AWQ, change normalization,
reduce denoise steps, or imply a deployment-quality result. The manifest records
calibration provenance, the actual number of calibrated tensors, and any selected
tensors with no observed input that retain ordinary rounding (for example unused
last-prefix-layer projections). An unrelated source model or heldout-split
statistics are rejected. Final action error and runtime must still be measured.

`bench_pi05_precision.py` records named precision, actual on-disk weight types,
input/noise/token hashes, language lengths, library/script hashes, raw timings,
and all 50×32 actions. Comparisons include per-joint MAE/RMSE/P95/P99/max error and
action-increment errors. Missing physical units, limits or gripper rules remain
unassessed. Repeated observations must use fresh processes and the same policy
settings; an experimental profile is never automatically promoted by a report.

`run_precision_suite.py --config POLICY --corpus CORPUS --precisions q8-prefix-mlp
q4-mixed --samples 12 --output-dir NEW_DIRECTORY` runs a baseline and each named
candidate in a fresh process. The suite records process return codes and excludes
reports left by a native shutdown failure. Its performance check is a development
screen only; physical quality and closed-loop status remain explicitly unassessed.


### Quantized execution routes and offline quality

A precision profile can set `"quant_route": "blas"` to decode packed weights to
FP16 scratch space at each GEMM and use vendor BLAS with FP32 accumulation/output
on M1000. This is a memory/storage alternative, not integer matrix acceleration.
The decode cost is included in prediction latency. `auto` keeps ggml dispatch.
The API sets the chosen route explicitly, and clears an inherited global BLAS
compute-type override so it cannot silently change baseline attention precision.
Per-profile route selection requires restarting the process.

The accuracy reference is an actual original-checkpoint BF16 forward, including
its original FP32 vision/norm exceptions. It is not the optimized F16 performance
baseline and not a cast of the final action tensor. Use
`prepare_pi05_bf16_inputs.py` with the runtime environment to freeze identical
RGB/token/noise inputs, then `reference_pi05_bf16.py` in the installed LeRobot
environment to run the original safetensors checkpoint. The reference runner
strictly loads weights, records source hashes and observed Linear dtypes, and
adapts the checkpoint's vision layer count and attention/hard token pruning.

The user-specified acceptance gate is cosine ≥ 0.999 for **each 7-dimensional
action vector at every one of the 50 times**, for every evaluated observation.
Global/chunk/per-joint trajectory cosines are diagnostic only; they do not replace
or add to that gate. Padded dimensions are excluded. Two zero vectors score 1;
only one zero vector scores 0. Raw output units need not be confirmed for a
paired comparison in unchanged model coordinates; absolute errors remain in
those native coordinates until units are verified.

`bench_pi05_precision.py --compare BF16_REPORT CANDIDATE_REPORT
--performance-reference F16_BASELINE_REPORT --output NEW_REPORT` keeps accuracy
and timing references separate and validates input/noise/token pairing for both.
`run_precision_suite.py --accuracy-reference BF16_REPORT ...` uses the same
separation. Without a BF16 reference it reports diagnostic drift only and cannot
mark offline accuracy passed. A failed baseline process stops the remaining
suite; reports written before a shutdown crash are not accepted. Offline
acceptance does not promote a profile to live deployment.

### Experimental native INT8 on M1000

An explicit precision profile with quant_route="native_int8" requires
int8_parameters pointing to an export manifest from
scripts/export_int8_parameters.py. This route uses W8A8 matrix arithmetic with
INT32 accumulation for the 18 action-expert GateUp projections. Other projections
remain floating point; this is a mixed-precision model, not a fully INT8 model.
The outer runtime settings stay F16 weights/F32 activations/F32 floating GEMM
accumulation. Resolved parameters identify the INT8 scope and accumulator type.

Calibration manifests must match the original floating GGUF SHA256 and contain
all layer parameter files with matching hashes. Parameters may use calibrated
per-channel smoothing and optional excluded-channel F16 correction. Dynamic
per-token activation quantization and all corrections execute inside prediction.
The MP22-only route rejects unsupported backends and does not silently use CPU
fallback. Baseline selection clears inherited INT8 and load-time-decode settings.

Current profiles remain experimental. Operator accuracy or speed alone is not
model acceptance. See the INT8 experiment archive for BF16 action comparisons,
full-model measurements, failed experiments, and remaining validation.

Native INT8 calibration manifests may also opt into expert_prefix_gateup: all expert GateUp projections plus prefix GateUp layers 0–16 (the last prefix layer only produces K/V). This remains mixed precision; attention, vision and Down projections remain floating point. Parameter manifests bind every selected layer and the original checkpoint by SHA256. It is experimental until end-to-end accuracy and timing validation passes.
