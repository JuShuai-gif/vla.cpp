"""LeRobot-compatible F32 RGB preprocessing for files or uint8 camera frames."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def _image_ops():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    return torch, F


def _read_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def prepare_rgb(rgb, size, out=None):
    """Resize a uint8 HWC RGB frame; optional out is reused and returned."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected nonempty uint8 HWC RGB")
    if not isinstance(size, int) or size < 1:
        raise ValueError("Expected a positive image size")
    if out is not None and (not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.shape != (size, size, 3)
                            or not out.flags.c_contiguous or not out.flags.writeable
                            or np.shares_memory(rgb, out)):
        raise ValueError("out must be independent writable contiguous F32 [size,size,3]")
    pixels = rgb.astype(np.float32, order="C")
    pixels /= 255.0
    h, w = rgb.shape[:2]
    if (h, w) == (size, size):
        if out is None:
            return pixels
        np.copyto(out, pixels)
        return out
    torch, F = _image_ops()
    ratio = max(w / size, h / size)
    rh, rw = max(1, int(h / ratio)), max(1, int(w / ratio))
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None]
    tensor = F.interpolate(tensor, size=(rh, rw), mode="bilinear", align_corners=False).clamp_(0, 1)
    top, left = (size - rh) // 2, (size - rw) // 2
    if out is None:
        out = np.zeros((size, size, 3), dtype=np.float32)
    else:
        out.fill(0)
    out[top:top+rh, left:left+rw] = tensor[0].permute(1, 2, 0).numpy()
    return out


def prepare_image(path, size):
    return prepare_rgb(_read_rgb(path), size)


@lru_cache(maxsize=1)
def _decoders():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="pi05-image")


def prepare_images(paths, size):
    """Decode both cameras concurrently; map preserves the checkpoint's order."""
    paths = list(paths)
    if len(paths) < 2:
        return [prepare_image(p, size) for p in paths]
    frames = list(_decoders().map(_read_rgb, paths))
    return [prepare_rgb(rgb, size) for rgb in frames]
