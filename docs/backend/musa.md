# MUSA (Moore Threads)

The MUSA build routes the VLA architectures through ggml-musa. Its public entry
points are named `ggml_backend_cuda_*`, but the runtime is MUSA, not CUDA.
`Backend::is_musa` is separate from `is_cuda`, so NVIDIA-only fused ops, BF16
activation hooks and FoldQuant integer kernels are not enabled accidentally.

## M1000 build

The baseline targets Linux aarch64, M1000 (`mp_22`), MUSA SDK package 5.1.1 (internal toolkit 5.1.0) and
driver 5.1.1-M1000. Use a separate build directory from CUDA or CPU builds.

On Ubuntu 22.04:

```bash
sudo apt-get install -y cmake ninja-build libstdc++-12-dev libomp-dev \
    libzmq3-dev libprotobuf-dev protobuf-compiler
BUILD_DIR="$PWD/build-musa-m1000" bash scripts/build_musa.sh
```

Ubuntu 22.04's `libzmq3-dev` includes `zmq.hpp`; distributions that package the
C++ headers separately also need `cppzmq-dev`.

The script selects the SDK's Clang compiler, builds the tools, server, shared
library and tests, then runs CTest. It accepts additional CMake arguments.
`MUSA_PATH` (default `/usr/local/musa`), `MUSA_ARCHITECTURES` (default `22`),
`BUILD_DIR` and `JOBS` (default `4`) can be overridden.

If mcc reports `cannot find -lstdc++`, check `clang++ -v`: it may have selected
GCC 12 while only GCC 11's C++ development files were installed. Install the
matching `libstdc++-12-dev` rather than changing the compiler's system symlinks.

The baseline disables VMM, graph capture and flash-attention kernels. These
are independent optimizations that need validation on the target device.
At configure time, `scripts/patch_ggml_musa.py` patches ggml's Softmax launch:
M1000/MUSA 5.1 rejects its opt-in dynamic shared-memory request. MUSA uses the
default shared-memory budget and the existing low-memory kernel for wider rows.
The fix also applies to an explicit `FETCHCONTENT_SOURCE_DIR_LLAMA` checkout.
The build script embeds the SDK library directory in the build RPATH, so the
binaries work from a fresh SSH shell without setting `LD_LIBRARY_PATH`.
For a minimal build without the server or tokenizer dependencies, pass
`-DVLA_BUILD_SERVER=OFF -DVLA_SPM=OFF`.

For unreliable SSH connections, run the build in a persistent terminal:

```bash
tmux new -s vla-musa
BUILD_DIR="$PWD/build-musa-m1000" bash scripts/build_musa.sh 2>&1 | tee build-musa.log
# Detach with Ctrl-b d; reconnect with: tmux attach -t vla-musa
```

## Checks without a checkpoint

```bash
./build-musa-m1000/tests/test_musa_ops
ctest --test-dir build-musa-m1000 --output-on-failure
```

`test_musa_ops` requires a live MUSA device (CTest reports a skip without one).
It checks F32, F16 and BF16 resident weights against CPU graphs, including the
actual Gemma builders used by pi0.5, multi-query attention with 256-wide heads,
prefix KV concatenation, RoPE, GEGLU, adaRMS, gated residuals, and the SigLIP
patch embedding. It uses synthetic tensors, including a two-view-length prefix
and 50 action tokens, and repeats execution with the same allocation.
Separate tests check explicit CPU selection and an out-of-range device index.

Softmax regression cases cover widths 1, 32, 51, 256, 579, 8192 and 32769,
with and without an F16 mask. The wide rows exercise the low-memory path.
They check CPU parity, row sums and zero probability at masked positions.

These checks do not establish full-model accuracy, latency or task success.
Those require the actual checkpoint and matching input preprocessing.

### Verified on skysi / aibox (2026-10-30)

- M1000, aarch64, MUSA SDK 5.1.0, driver 5.1.1-M1000.
- llama.cpp `b11223` (`4da6337767f973e2b4d0797e5b323d77d8565e4a`).
- All build targets succeeded, including `vla-cli`, `vla-bench`, `vla-server`,
  `libvla.so`, and `vla_predict_check`. The three VLA tools pass `--help` from
  a fresh SSH shell.
- CTest: **13/13 passed**, including live MUSA execution (no skipped tests).
- `test_musa_ops`: **26 CPU/device comparisons passed**.
- CMake rejects mixed accelerator flags and `GGML_BACKEND_DL=ON` with MUSA.
- The upstream patch is idempotent. `git diff --check` passes.

Largest normalized errors (`max_abs_difference / max_abs_CPU_reference`):

| Resident weight type | Vision chain | Gemma + expert chains | Chain tolerance |
|---|---:|---:|---:|
| F32 | 1.09e-3 | 1.03e-5 | 2e-3 |
| F16 | 1.99e-3 | 3.23e-3 | 5e-3 |
| BF16 | 5.39e-3 | 3.18e-4 | 1e-2 |

Softmax's largest normalized error was 3.02e-7 (tolerance 2e-5). Reduced-precision
GEMMs round activations/outputs differently from the CPU kernels; these are
synthetic-chain tolerances, not acceptance criteria for a trained policy.

The board checkout is `/home/skysi/code/vla.cpp`, and the verified build is
`build-musa-m1000`. Its pinned llama checkout is `build-musa-deps/llama`, selected
with `-DFETCHCONTENT_SOURCE_DIR_LLAMA="$PWD/build-musa-deps/llama"` to avoid
fetching every upstream tag. Both local pi0.5 variants have now passed conversion and CPU/MUSA inference checks.
See [local pi0.5 workflow](pi05-local.md) for current commands, results, and limits.

## pi0.5 checkpoint validation

Use a pi0.5 GGUF with the vision tower bundled. Raw safetensors/OpenPI weights
need conversion first (see [models](../MODELS.md)).

```bash
export PI05_GGUF=/absolute/path/to/pi05.gguf
VLA_IMG_SIZE=224 VLA_DEVICE=0 ./build-musa-m1000/tests/vla_predict_check \
    "$PI05_GGUF" '' 2 --weight-dtype bf16 > pi05-musa.txt
VLA_IMG_SIZE=224 VLA_DEVICE=cpu ./build-musa-m1000/tests/vla_predict_check \
    "$PI05_GGUF" '' 2 --weight-dtype bf16 > pi05-cpu.txt
```

The device run must print `backend = MUSA (device 0: ...)`. Compare the action
outputs with the CPU reference before measuring performance. `VLA_DEVICE=cpu`
runs the same executable on CPU; a missing/out-of-range GPU also logs a CPU
fallback, so check the backend line rather than only the exit status.

Activations default to F32; pi0.5 also supports `--act-dtype f16` with F16
weights and fusion (see [local workflow](pi05-local.md)). `--act-dtype bf16` is a CUDA/SYCL-specific path and is not
implemented for MUSA. FoldQuant checkpoints use the existing float-weight
dequantization path, not native MUSA W8A8/W4A4 execution; its numerical behavior
differs from the integer reference and needs separate model-level validation.

## Local checkpoint follow-up (2026-10-30)

See [pi05-local.md](pi05-local.md). The final suite is 14/14, including attention
score reduction and a segmented 5 GiB weight-allocation regression. MUSA buffers
are capped at 2 GiB because this M1000/5.1 stack aliases offsets 4 GiB apart
inside a single large allocation. Both standard 27-layer/256-token and pruned
15-layer/136-token checkpoints have completed full CPU/MUSA synthetic checks.

Current experiment records and submission-time validation are in the
[versioned benchmark archive](benchmarks/pi05-m1000-2026-10-30/README.md).
