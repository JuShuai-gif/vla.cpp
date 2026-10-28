# vla.cpp

![logo](assets/logo_vlacpp_white.png)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)
[![Built on llama.cpp](https://img.shields.io/badge/built%20on-llama.cpp-lightgrey)](https://github.com/ggml-org/llama.cpp)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20models-Hugging%20Face-yellow)](https://huggingface.co/vrfai)
[![arXiv](https://img.shields.io/badge/arXiv-2606.08094-b31b1b.svg)](http://arxiv.org/abs/2606.08094)
[![Docs](https://img.shields.io/badge/docs-Learn%20vla.cpp-brightgreen)](https://fai-modelopt-tech.github.io/learn-vla-cpp/)

A C++ inference engine for **Vision-Language-Action (VLA) models**, built on [`llama.cpp`](https://github.com/ggml-org/llama.cpp).
It runs the open VLA policies under one runtime, each packaged as a single self-contained GGUF that needs no Python or
PyTorch at inference time. The binaries drive robots on **CPU**, **Apple Silicon**,
**NVIDIA** (including Jetson), **AMD**, **Intel** and **Qualcomm** hardware.

---

## Install

Download a prebuilt release, or build from source for any other platform or
backend.

### Prebuilt binaries

Each [release](https://github.com/VinRobotics/vla.cpp/releases) has a
`vla.cpp-<tag>-<platform>.tar.gz` with `libvla`, `vla.h` and the binaries:

| Platform | Needs |
|---|---|
| `linux-x86_64-cpu` | AVX2 (Haswell or newer) |
| `linux-x86_64-cuda-12.8` | AVX2; sm_75/80/86/89/90/120 |
| `linux-x86_64-cuda-13.4` | AVX2; sm_75/80/86/89/90/120, driver 580 or newer |
| `linux-aarch64-cpu` | ARMv8.2-A with dotprod and fp16 (Cortex-A76, Neoverse N1 or newer) |
| `linux-aarch64-cuda-13.4` | sm_87 (Orin), sm_110 (Thor), sm_121 (DGX Spark); a CUDA 13 driver |
| `macos-arm64-metal` | `vla-cli` and `vla-bench` only |

```bash
TAG=<release tag>
curl -LO https://github.com/VinRobotics/vla.cpp/releases/download/$TAG/vla.cpp-$TAG-linux-x86_64-cpu.tar.gz
tar -xzf vla.cpp-$TAG-linux-x86_64-cpu.tar.gz
```

The Linux tarballs need Ubuntu 24.04's glibc or newer, `libzmq5` for the
servers, and a CUDA runtime for the CUDA ones (shipped as a separate `cudart-`
tarball). Details, and the macOS and Windows notes, are in
[docs/PREBUILT.md](docs/PREBUILT.md).

### Build from source

The CPU build needs CMake ≥ 3.22, a C++17 compiler (GCC 11+ or Clang 14+),
ZeroMQ and protobuf. CMake fetches and pins `llama.cpp` itself:

```bash
sudo apt-get install -y libzmq3-dev cppzmq-dev libprotobuf-dev protobuf-compiler
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

`-DVLA_BUILD_SERVER=OFF -DVLA_SPM=OFF` drops the apt packages and builds only
`vla-cli`, `vla-bench` and `libvla`; Octo then takes `--tokens` instead of `--text`.

For a GPU or NPU, follow the backend's page in [docs/backend/](docs/backend):
[CUDA](docs/backend/cuda.md), [ROCm](docs/backend/rocm.md),
[SYCL](docs/backend/sycl.md), [OpenVINO](docs/backend/ov.md),
[Metal](docs/backend/metal.md), [Hexagon](docs/backend/hexagon-windows.md).
For containers, see [docs/DOCKER.md](docs/DOCKER.md).

---

## Quickstart

Once the binaries are built, run one CPU prediction without a server or simulator:

```bash
pip install -U "huggingface_hub[cli]" transformers

# -hf fetches and caches the checkpoint (under $VLA_CACHE, default ~/.cache/vla)
./build/vla-cli -hf vrfai/smolvla-libero-gguf \
    --image assets/front.jpg --text "pick up the black bowl" --pretty

# or point at a file you already have
./build/vla-cli --ckpt models/smolvla/smolvla-libero.gguf \
    --image assets/front.jpg --text "pick up the black bowl" --pretty
```

With a prebuilt tarball, run `vla-cli` from the extracted
`vla.cpp-<tag>-<platform>/` directory instead of `./build/`.
`--pretty` prints one action row per line and `--state` sets proprioception.
Prompt tokenization, `-hf` tags, `vla-server`, runtime flags and environment
variables are in [docs/USAGE.md](docs/USAGE.md).

---

## Support matrix

Models (rows) against platforms (columns). Legend: `Y` =
supported (released and benchmarked), `~` = in progress, `-` = planned.

| Model | CPU (x86-64 / ARM) | [CUDA (NVIDIA)](docs/backend/cuda.md) | [ROCm (AMD)](docs/backend/rocm.md) | [SYCL (Intel)](docs/backend/sycl.md) | [Metal (Apple)](docs/backend/metal.md) | [OpenVINO (Intel)](docs/backend/ov.md) | [Hexagon (Qualcomm)](docs/backend/hexagon-windows.md) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| [SmolVLA](https://hf.co/vrfai/smolvla-libero-gguf)             | Y | Y | Y | Y | Y | Y | Y |
| [π0](https://hf.co/vrfai/pi0-libero-finetuned-v044-gguf)       | Y | Y | Y | Y | Y | Y | ~ |
| [π0.5](https://hf.co/vrfai/pi05-libero-gguf)                   | Y | Y | Y | Y | Y | Y | Y |
| [GR00T N1.5](https://hf.co/vrfai/gr00tn1d5-libero-object-gguf) | Y | Y | Y | Y | Y | Y | Y |
| [GR00T N1.6](https://hf.co/vrfai/gr00tn1d6-libero-gguf)        | Y | Y | Y | Y | Y | Y | Y |
| [GR00T N1.7](https://hf.co/vrfai/gr00tn1d7-libero-gguf)        | Y | Y | Y | Y | Y | Y | Y |
| [BitVLA](https://hf.co/vrfai/bitvla-libero-gguf)               | Y | Y | - | - | - | - | - |
| [Evo-1](https://hf.co/vrfai/evo1-libero-gguf)                  | Y | Y | Y | Y | Y | Y | Y |
| [VLA-Adapter](https://hf.co/vrfai/vla-adapter-libero-gguf)     | Y | Y | Y | Y | Y | Y | Y |
| [OpenVLA-OFT](https://hf.co/vrfai/openvla-oft-libero-gguf)     | Y | Y | Y | - | Y | Y | - |
| [VLA-JEPA](https://hf.co/vrfai/vla-jepa-libero)                | Y | Y | Y | Y | Y | Y | ~ |
| [Octo-Small](https://hf.co/vrfai/octo-small-libero-gguf)       | Y | Y | Y | Y | Y | - | Y |
| [TurboVLA](https://hf.co/vrfai/turbovla-libero-gguf)           | Y | Y | Y | Y | Y | Y | Y |
| [ACT](https://hf.co/ravediamond/act-aloha-sim-transfer-cube-gguf) | Y | Y | Y | ~ | Y | ~ | ~ |

Use [ggml-compiler](https://github.com/khanhnd61-vr/ggml-compiler) and follow [the instruction](docs/ARCHITECTURE.md#adding-an-architecture) to add a new model.

---

## Rollout on a real robot

Use the [LeRobot fork](https://github.com/khanhnd61-vr/lerobot/tree/vla-simd) with `lerobot-vla-cpp` client drives an SO-101 arm against a
`vla-server`. The client does the per-arch preprocessing (tokenize, resize,
normalize), so it covers SmolVLA, π0, π0.5 and GR00T N1.5/1.6/1.7.

```bash
git clone -b vla-simd https://github.com/khanhnd61-vr/lerobot.git
cd lerobot && pip install -e ".[vla-cpp]"
```

Serve the policy, converted to GGUF as in [docs/MODELS.md](docs/MODELS.md), on a
GPU host. SmolVLA takes 38 ms per query on an RTX 5090 and 218 ms on a Jetson
AGX Orin, against the 1.67 s of motion a 50-step chunk buys at 30 fps; on a
desktop CPU it takes ~1.7 s and the action queue stalls.

```bash
./build/vla-server smolvla-so101.gguf   # binds tcp://*:5555; --bind moves it
```

Check the round trip with no arm attached, then run the arm. `ROBOT` holds the
arm's `--robot.*` flags; the fork's README sets it up:

```bash
# prints round-trip latency next to the recorded actions
lerobot-vla-cpp --server_address=tcp://127.0.0.1:5555 --arch=smolvla \
  --replay.repo_id=khanhnd61/so101-multi-task-clean --replay.steps=10

lerobot-vla-cpp --server_address=tcp://127.0.0.1:5555 --arch=smolvla "${ROBOT[@]}" \
  --task="Put the tape into the box" --n_action_steps=25 --fps=30 --duration=60
```

- `--arch` selects the preprocessing (`smolvla`, `pi0`, `pi05`, `gr00t_n1_5/6/7`,
  or `passthrough` for a server that preprocesses itself). The wrong arch loads,
  runs and returns plausible, wrong actions.
- `--stats_json` is required for `pi05` and the GR00T archs; GR00T also takes
  `--embodiment`, and `--rel_stats_json` for an N1.7 checkpoint trained with
  relative actions.
- `--task` must match a trained instruction exactly, and the camera keys must
  stay `front` and `wrist` in that order.
- `vla-server` answers one request at a time, so the loop is synchronous and
  `--n_action_steps` is the feedback rate: 25 at 30 fps leaves ~0.83 s between
  observations.

The GR00T paths of the client have not been tested against real checkpoints yet.
Wiring, recording, training and queue sizing are in the
[fork's README](https://github.com/khanhnd61-vr/lerobot/tree/vla-simd#readme).

---

## Companion Repos

- [llama.cpp](https://github.com/ggml-org/llama.cpp) provides ggml and its backends. CMake fetches it at a pinned tag (`b11223`), so there is no submodule.
- [LeRobot fork](https://github.com/khanhnd61-vr/lerobot/tree/vla-simd) runs a real robot against `vla-server` ([rollout](#rollout-on-a-real-robot)).
- [ggml-compiler](https://github.com/khanhnd61-vr/ggml-compiler) to generate and verify ggml C++ from its PyTorch implementation, then integrate the graph with `vla.cpp`'s model interface.
- [Learn vla.cpp](https://fai-modelopt-tech.github.io/learn-vla-cpp/) walks through the engine design and how each policy is implemented on ggml.



---

## Documentation

| Doc | Content |
|---|---|
| [docs/PREBUILT.md](docs/PREBUILT.md) | Release tarballs: layout, glibc and CUDA runtime requirements, macOS and Windows notes |
| [docs/USAGE.md](docs/USAGE.md) | `vla-cli`, `vla-server`, `vla-bench`: prompt tokenization, `-hf` tags, runtime flags, environment variables |
| [docs/EVAL.md](docs/EVAL.md) | Installing LIBERO and SimplerEnv, running the eval clients against `vla-server` |
| [docs/MODELS.md](docs/MODELS.md) | Converting safetensors checkpoints to GGUF, quantizing to Q8_0/Q4_0 |
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md) | FoldQuant W8A8 / W4A4: the GGUF contract, the integer arithmetic, converting a FoldQuantVLA quantized model |
| [docs/DOCKER.md](docs/DOCKER.md) | Building and running the eval in containers |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Engine design: layers, the prediction path, backends, adding an architecture |
| [docs/backend/](docs/backend) | Per-backend build and run notes: [CUDA](docs/backend/cuda.md), [ROCm](docs/backend/rocm.md), [SYCL](docs/backend/sycl.md), [OpenVINO](docs/backend/ov.md), [Metal](docs/backend/metal.md), [Hexagon](docs/backend/hexagon.md), [Hexagon on Windows](docs/backend/hexagon-windows.md), [WSL2](docs/backend/wsl.md) |
| [docs/benchmark/](docs/benchmark) | Per-device latency and memory for every model, and the fastest flags per device |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | Known issues and their resolutions |
| [docs/ADOPTION.md](docs/ADOPTION.md) | C ABI, Python bindings, release packaging, and what is left |
| [docs/UPSTREAMING.md](docs/UPSTREAMING.md) | The ggml-openvino fixes to send upstream to llama.cpp |
| [CHANGELOG.md](CHANGELOG.md) | Per-release changes, with the fastest configuration and success rate per model |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Proving a change is numerically neutral, adding an architecture |
| [bindings/python/](bindings/python/README.md) | Python bindings |

---

## Citation

```bibtex
@misc{nguyen2026vlacpp,
  title         = {{vla.cpp}: A Unified Inference Runtime for Vision-Language-Action Models},
  author        = {Khanh D. Nguyen and Hung T. Ho and Chinh T. Nguyen and
                   Thanh Q. Duong and Linh D. Le and Duy M. H. Nguyen and
                   Vien A. Ngo and An T. Le},
  year          = {2026},
  eprint        = {2606.08094},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2606.08094},
  url           = {https://arxiv.org/abs/2606.08094}
}
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE.md).

---

## Acknowledgements

Supported VLA models:

- [ACT](https://arxiv.org/abs/2304.13705) - Zhao et al.
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base) - Hugging Face LeRobot team.
- [π0,π0.5](https://github.com/Physical-Intelligence/openpi) - Physical Intelligence.
- [BitVLA](https://github.com/ustcwhy/BitVLA) - Hongyu Wang et al.
- [Evo-1](https://github.com/MINT-SJTU/Evo-1/tree/main) - Tao Lin et al.
- [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) - Yihao Wang et al.
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft) - Moo Jin Kim et al.
- [GR00T N1.x](https://github.com/NVIDIA/Isaac-GR00T) - NVIDIA Isaac.
- [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) - Jingwen Sun et al.
- [Octo](https://github.com/octo-models/octo) - Octo Model Team, UC Berkeley RAIL.
- [TurboVLA](https://github.com/H-EmbodVis/TurboVLA) - Hengyi Xie et al.

Built on:

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) - LLM inference engine in C/C++.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) - benchmark suite for the success-rate sweeps.
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv) - the second simulator in the eval scaffold.
