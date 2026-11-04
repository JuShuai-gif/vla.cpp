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
