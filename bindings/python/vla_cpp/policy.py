"""Deployment API for local pi0.5 GGUF policies; no robot or CUDA dependencies."""
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from .pi05_gguf import GGUFReader
from .pi05_preprocess import prepare_rgb


@dataclass(frozen=True)
class PolicySpec:
    checkpoint: str
    camera_order: tuple
    state_dim: int
    action_dim: int
    chunk_size: int
    action_names: tuple
    action_space: str
    joint_unit: str
    gripper_unit: str
    contract_verified: bool
    image_size: int
    num_steps: int
    variant: str
    rtc_training_max_delay: int = 0


def _config(path):
    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ValueError("Policy JSON requires schema_version=1")
    allowed = {"schema_version", "model", "library", "optimization_profile", "device",
               "seed", "threads", "runtime", "source_contract", "task"}
    if set(cfg) - allowed:
        raise ValueError(f"Unknown policy fields: {sorted(set(cfg) - allowed)}")
    cfg = deepcopy(cfg)
    for key in ("model", "library", "optimization_profile"):
        value = cfg.get(key)
        if key == "optimization_profile" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Policy requires {key}")
        candidate = (path.parent / Path(value).expanduser()).resolve()
        if not candidate.is_file():
            raise ValueError(f"Missing {key}: {candidate}")
        cfg[key] = str(candidate)
    runtime = cfg.get("runtime", {})
    if not isinstance(runtime, dict) or set(runtime) - {"weight_dtype", "act_dtype", "mm_prec", "flash_attn"}:
        raise ValueError("Unsupported runtime options")
    runtime = {"weight_dtype": "f16", "act_dtype": "f32", "mm_prec": "f32", "flash_attn": True, **runtime}
    if (runtime["weight_dtype"] not in ("f32", "f16", "bf16")
            or runtime["act_dtype"] not in ("f32", "f16")
            or runtime["mm_prec"] not in ("f32", "default")
            or type(runtime["flash_attn"]) is not bool):
        raise ValueError("Invalid runtime precision options")
    if cfg.get("optimization_profile") and runtime != {
            "weight_dtype": "f16", "act_dtype": "f32", "mm_prec": "f32", "flash_attn": True}:
        raise ValueError("Validated MUSA profiles require F16 weights/F32 activations/F32 accumulation/FA")
    cfg["runtime"] = runtime
    cfg.setdefault("seed", 0)
    cfg.setdefault("threads", 4)
    cfg.setdefault("device", "0")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**64:
        raise ValueError("seed must be an integer in [0,2**64)")
    if type(cfg["threads"]) is not int or not 1 <= cfg["threads"] <= 64:
        raise ValueError("threads must be in [1,64]")
    if not isinstance(cfg["device"], str) or not (cfg["device"] == "cpu" or cfg["device"].isdigit()):
        raise ValueError("device must be cpu or a nonnegative device index string")
    if cfg.get("optimization_profile") and cfg["device"] == "cpu":
        raise ValueError("MUSA optimization profile requires a GPU device")
    contract = cfg.get("source_contract")
    if not isinstance(contract, dict) or set(contract) != {
            "action_names", "action_space", "joint_unit", "gripper_unit", "verified"}:
        raise ValueError("Declare source_contract action names, space, units and verified flag")
    if (contract["action_space"] != "absolute_joint"
            or contract["joint_unit"] not in ("degrees", "radians")
            or contract["gripper_unit"] not in ("mm", "meters")
            or type(contract["verified"]) is not bool):
        raise ValueError("Invalid source contract; values remain in checkpoint units, no implicit conversion")
    return cfg


def _inspect(config_path):
    cfg = _config(config_path)
    reader = GGUFReader(cfg["model"])
    def field(name):
        if name not in reader.fields:
            raise ValueError(f"GGUF is missing {name}")
        return reader.fields[name].contents()
    if field("general.architecture") != "pi05":
        raise ValueError("Policy API currently supports pi05 GGUF only")
    cameras = field("pi05.camera_keys")
    if (not isinstance(cameras, list) or not cameras or not all(isinstance(x, str) for x in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError("Invalid GGUF camera order")
    dimensions = [field("pi05." + key) for key in (
        "real_state_dim", "real_action_dim", "chunk_size", "image_size", "num_steps")]
    if any(type(x) is not int or not 1 <= x <= 4096 for x in dimensions):
        raise ValueError("Invalid GGUF dimensions")
    state_dim, action_dim, horizon, size, steps = dimensions
    names = cfg["source_contract"]["action_names"]
    if (not isinstance(names, list) or len(names) != action_dim
            or not all(isinstance(x, str) and x for x in names) or len(set(names)) != len(names)):
        raise ValueError("action_names must uniquely name every action dimension")
    if field("pi05.norm_mode") != "quantiles":
        raise ValueError("Local pi05 prompt preprocessing requires quantile normalization")
    tensors = {t.name: t for t in reader.tensors}
    for name in ("state_q01", "state_q99"):
        if name not in tensors:
            raise ValueError(f"Missing {name}")
        values = np.asarray(tensors[name].data).reshape(-1)
        if values.shape != (state_dim,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name}")
    eps = field("pi05.norm_eps")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid normalization epsilon")
    if not any("pi05.tokenizer." + x in reader.fields for x in ("hf_json", "spm_model")):
        raise ValueError("GGUF must embed its training tokenizer")
    prune = reader.fields.get("pi05.vision_token_prune")
    variant = "pruned" if prune is not None and int(prune.contents()) > 0 else "full"
    contract = cfg["source_contract"]
    spec = PolicySpec(cfg["model"], tuple(cameras), state_dim, action_dim, horizon,
                      tuple(names), contract["action_space"], contract["joint_unit"],
                      contract["gripper_unit"], contract["verified"], size, steps, variant)
    if cfg.get("optimization_profile"):
        profiles = json.loads(Path(cfg["optimization_profile"]).read_text())
        if not isinstance(profiles, dict) or variant not in profiles:
            raise ValueError("Optimization profile does not contain the model variant")
        for settings in profiles.values():
            if not isinstance(settings, dict) or any(
                not isinstance(k, str) or not k.startswith(("VLA_", "GGML_"))
                or k in ("VLA_LIBRARY", "VLA_DEVICE", "VLA_N_THREADS") or not isinstance(v, str)
                for k, v in settings.items()):
                raise ValueError("Invalid optimization environment profile")
        cfg["profiles"] = profiles
    return cfg, spec, reader


def inspect_policy(config_path):
    """Read metadata and deployment contract without loading libvla or a device."""
    return _inspect(config_path)[1]


@lru_cache(maxsize=2)
def _tokenizer(kind, model, length):
    if kind == "json":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_str(model)
        tok.no_padding()
        tok.enable_truncation(max_length=length)
        return tok
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_proto=model)


def prepare_prompt(reader, text, state):
    """The local checkpoint's original state-bin/tokenizer contract."""
    tensors = {t.name: t for t in reader.tensors}
    q01 = np.asarray(tensors["state_q01"].data, dtype=np.float32).reshape(-1)
    q99 = np.asarray(tensors["state_q99"].data, dtype=np.float32).reshape(-1)
    state = np.asarray(state, dtype=np.float32)
    if state.shape != q01.shape or not np.isfinite(state).all():
        raise ValueError(f"Expected {len(q01)} finite raw state values")
    eps = reader.fields["pi05.norm_eps"].contents()
    denom = q99 - q01
    normed = 2 * (state - q01) / np.where(denom == 0, eps, denom) - 1
    bins = np.digitize(normed, np.linspace(-1, 1, 257)[:-1]) - 1
    text = text.strip().replace("_", " ").replace("\n", " ")
    prompt = f"Task: {text}, State: {' '.join(map(str, bins.tolist()))};\nAction: "
    length = int(reader.fields["pi05.tokenizer_max_length"].contents())
    if "pi05.tokenizer.hf_json" in reader.fields:
        tok = _tokenizer("json", reader.fields["pi05.tokenizer.hf_json"].contents(), length)
        ids = tok.encode(prompt).ids
    elif "pi05.tokenizer.spm_model" in reader.fields:
        field = reader.fields["pi05.tokenizer.spm_model"]
        model = field.raw_bytes() if hasattr(field, "raw_bytes") else bytes(field.contents())
        tok = _tokenizer("spm", model, length)
        ids = ([tok.bos_id()] + tok.encode(prompt))[:length]
    else:
        raise ValueError("GGUF has no local tokenizer; use convert_pi05_local.py")
    return prompt, ids


# ggml has process-global environment switches and a shared native library handle.
# One resident policy per process prevents cross-model profile contamination.
_session_lock = threading.Lock()
_process_settings = None
_inactive = {"VLA_PI05_K_ROPE_DIRECT", "VLA_PI05_PREFIX_GAMMA_FOLD", "VLA_PI05_PREFETCH_KIB",
             "VLA_MUSA_GEGLU_FAST", "VLA_PI05_F16_PART", "VLA_MUSA_FLATTEN_GEMM",
             "GGML_CUDA_DISABLE_GRAPHS", "VLA_PI05_NO_SEPARATE_GRAPHS",
             "VLA_PI05_NO_VISION_BATCH", "VLA_PI05_NO_PROJ_FUSE", "VLA_PI05_NO_FUSE"}


class Policy:
    """Persistent pi05 session. Images are named HWC RGB uint8 camera frames."""
    def __init__(self, config_path, *, task=None):
        global _process_settings
        cfg, self.spec, self._reader = _inspect(config_path)
        self.task = task if task is not None else cfg.get("task", "")
        if not isinstance(self.task, str):
            raise ValueError("task must be a string")
        if not _session_lock.acquire(blocking=False):
            raise RuntimeError("Only one resident vla_cpp policy per process; close it before switching")
        self._owns_session = True
        self._mu = threading.RLock()
        self._native = None
        self._saved_env = {}
        self.last_predict_timings_ms = {}
        try:
            profiles = cfg.get("profiles", {})
            env = {k: None for k in _inactive | set().union(*(v.keys() for v in profiles.values()))}
            env.update(profiles.get(self.spec.variant, {}))
            env.update(VLA_LIBRARY=cfg["library"], VLA_DEVICE=cfg["device"], VLA_N_THREADS=str(cfg["threads"]))
            settings = (cfg["library"], cfg["runtime"], env)
            if _process_settings is not None and _process_settings != settings:
                raise RuntimeError("Native options may be cached; switch model profiles in a fresh process")
            _process_settings = deepcopy(settings)
            for key, value in env.items():
                self._saved_env[key] = os.environ.get(key)
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            from . import _ffi
            from . import Model
            # Load the exact configured library; do not silently select a bundled CPU build.
            import ctypes
            ctypes.CDLL(cfg["library"])
            lib = _ffi.load_library()
            with tempfile.TemporaryDirectory(prefix="vla-policy-") as folder:
                path = Path(folder) / "runtime.json"
                path.write_text(json.dumps({"runtime": cfg["runtime"]}))
                handle = lib.vla_model_load(b"", cfg["model"].encode(), str(path).encode())
                if not handle:
                    raise RuntimeError("libvla failed to load the policy; see native diagnostics")
                self._native = Model(handle, lib)
            resolved = self._native.config
            if (resolved.real_state_dim != self.spec.state_dim or resolved.real_action_dim != self.spec.action_dim
                    or resolved.n_suffix != self.spec.chunk_size or resolved.num_steps != self.spec.num_steps
                    or resolved.max_state_dim < self.spec.state_dim
                    or resolved.max_action_dim < self.spec.action_dim
                    or not resolved.denormalized):
                raise ValueError("Native model config differs from metadata or output is not denormalized")
            self._rng = np.random.default_rng(cfg["seed"])
            self._buffers = {name: np.empty((self.spec.image_size, self.spec.image_size, 3), np.float32)
                             for name in self.spec.camera_order}
            self.resolved_parameters = {k: {"value": v, "source": "policy config / GGUF"} for k, v in {
                "checkpoint": self.spec.checkpoint, "task": self.task, "precision": cfg["runtime"]["weight_dtype"],
                "act_dtype": cfg["runtime"]["act_dtype"], "device_requested": cfg["device"], "seed": cfg["seed"],
                "num_steps": self.spec.num_steps, "optimization_variant": self.spec.variant if profiles else None,
                "source_contract_verified": self.spec.contract_verified}.items()}
        except BaseException:
            self.close()
            raise

    def predict(self, images, *, state, task=None):
        with self._mu:
            if self._native is None:
                raise RuntimeError("Policy is closed")
            if not isinstance(images, dict) or set(images) != set(self.spec.camera_order):
                raise ValueError(f"Images must be named exactly {self.spec.camera_order}")
            task = self.task if task is None else task
            if not isinstance(task, str) or not task.strip():
                raise ValueError("Provide the task instruction")
            started = time.perf_counter()
            _, tokens = prepare_prompt(self._reader, task, state)
            frames = [prepare_rgb(images[name], self.spec.image_size, self._buffers[name]) for name in self.spec.camera_order]
            cfg = self._native.config
            padded = np.zeros(int(cfg.max_state_dim), np.float32)
            padded[:self.spec.state_dim] = np.asarray(state, np.float32)
            noise = self._rng.standard_normal(int(cfg.n_suffix * cfg.max_action_dim)).astype(np.float32)
            prepared = time.perf_counter()
            from . import PIXEL_F32_RGB_01, TIMING_NONE
            result = np.asarray(self._native.predict(frames, tokens=tokens, state=padded, noise=noise,
                                pixel_format=PIXEL_F32_RGB_01, timing=TIMING_NONE), dtype=np.float32)
            predicted = time.perf_counter()
            if result.shape != (self.spec.chunk_size, cfg.max_action_dim) or not np.isfinite(result).all():
                raise ValueError("Native policy returned invalid actions")
            actions = np.ascontiguousarray(result[:, :self.spec.action_dim])
            self.last_predict_timings_ms = {"preprocess_ms": (prepared-started)*1000,
                "native_call_ms": (predicted-prepared)*1000, "postprocess_ms": (time.perf_counter()-predicted)*1000}
            return actions

    def warmup(self, images, *, state, task=None, runs=3):
        if type(runs) is not int or not 1 <= runs <= 1000:
            raise ValueError("warmup runs must be in [1,1000]")
        with self._mu:
            if self._native is None:
                raise RuntimeError("Policy is closed")
            saved = deepcopy(self._rng.bit_generator.state)
            try:
                for _ in range(runs):
                    self.predict(images, state=state, task=task)
            finally:
                self._rng.bit_generator.state = saved

    def close(self):
        with self._mu:
            try:
                if self._native is not None:
                    self._native.close()
            finally:
                self._native = None
                self._reader = None
                for key, value in self._saved_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                self._saved_env.clear()
                if self._owns_session:
                    self._owns_session = False
                    _session_lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_policy(config_path, *, task=None):
    """Load one policy; state/actions use the explicitly declared checkpoint units."""
    return Policy(config_path, task=task)
