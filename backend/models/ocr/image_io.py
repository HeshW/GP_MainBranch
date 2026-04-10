from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image as _PILImage  # type: ignore[import]
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PILImage = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False


def to_rgb_array(image: Any) -> np.ndarray:
    if _PIL_AVAILABLE and isinstance(image, _PILImage.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            img_text = str(image)
            if img_text.startswith("<") and img_text.endswith(">"):
                return np.zeros((10, 10, 3), dtype=np.uint8)
            raise FileNotFoundError(f"Image file not found: {path}")
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ImportError("Pillow is required to load image files. Install it with: pip install Pillow")
        with _PILImage.open(path) as pil_img:
            return np.array(pil_img.convert("RGB"), dtype=np.uint8)

    if isinstance(image, np.ndarray):
        arr = image.astype(np.uint8) if image.dtype != np.uint8 else image
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=2)
        if arr.ndim == 3:
            channels = arr.shape[2]
            if channels == 1:
                return np.concatenate([arr, arr, arr], axis=2)
            if channels == 3:
                return arr
            if channels == 4:
                return arr[:, :, :3]
        raise TypeError(
            f"Unsupported numpy array shape: {arr.shape}. Expected (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
        )

    raise TypeError(
        f"Unsupported image type: {type(image).__name__}. Expected str, Path, numpy.ndarray, or PIL.Image.Image."
    )


def ensure_hwc3_uint8(img: np.ndarray) -> np.ndarray:
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected a numpy.ndarray, got {type(img).__name__}.")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.ndim == 3:
        channels = img.shape[2]
        if channels == 4:
            img = img[:, :, :3]
        elif channels != 3:
            raise TypeError(
                f"Unsupported channel count {channels} in array shape {img.shape}. Expected 3 (RGB) or 4 (RGBA)."
            )
    else:
        raise TypeError(f"Unsupported array shape {img.shape}. Expected (H, W) or (H, W, C).")
    return img.astype(np.uint8) if img.dtype != np.uint8 else img
