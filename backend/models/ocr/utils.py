"""
Image pre-processing utilities for OCR.

All functions accept a file path or a NumPy array (OpenCV image) and return
a NumPy array so they can be chained together easily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

ImageInput = Union[str, Path, np.ndarray]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _load(image: ImageInput) -> np.ndarray:
    """Load *image* from disk if it is a path; pass through if already an array."""
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return img


# ---------------------------------------------------------------------------
# Public preprocessing functions
# ---------------------------------------------------------------------------

def to_grayscale(image: ImageInput) -> np.ndarray:
    """Convert *image* to grayscale.

    Parameters
    ----------
    image:
        File path (str / Path) or an OpenCV BGR NumPy array.

    Returns
    -------
    np.ndarray
        Single-channel (grayscale) NumPy array.
    """
    img = _load(image)
    if len(img.shape) == 2:
        return img  # already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_threshold(
    image: ImageInput,
    *,
    method: str = "otsu",
    block_size: int = 11,
    constant: int = 2,
) -> np.ndarray:
    """Apply binarisation thresholding to *image*.

    Parameters
    ----------
    image:
        File path (str / Path) or an OpenCV grayscale NumPy array.
    method:
        ``"otsu"`` (default) for global Otsu thresholding, or
        ``"adaptive"`` for adaptive Gaussian thresholding.
    block_size:
        Neighbourhood size used by adaptive thresholding (must be odd).
        Ignored when *method* is ``"otsu"``.
    constant:
        Constant subtracted from the mean in adaptive thresholding.
        Ignored when *method* is ``"otsu"``.

    Returns
    -------
    np.ndarray
        Binary (black & white) NumPy array.

    Raises
    ------
    ValueError
        If *method* is not one of the supported strings.
    """
    gray = to_grayscale(image)
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant,
        )
    else:
        raise ValueError(f"Unknown threshold method '{method}'. Use 'otsu' or 'adaptive'.")
    return binary


def denoise(image: ImageInput, *, h: int = 10) -> np.ndarray:
    """Remove noise with a non-local-means filter.

    Parameters
    ----------
    image:
        File path (str / Path) or a grayscale NumPy array.
    h:
        Filter strength. Higher values remove more noise but may blur edges.

    Returns
    -------
    np.ndarray
        Denoised grayscale NumPy array.
    """
    gray = to_grayscale(image)
    return cv2.fastNlMeansDenoising(gray, h=h)


def preprocess(image: ImageInput, *, method: str = "otsu") -> np.ndarray:
    """Apply the full default preprocessing pipeline.

    Applies ``denoise → to_grayscale → apply_threshold`` in order.

    Parameters
    ----------
    image:
        File path (str / Path) or an OpenCV BGR NumPy array.
    method:
        Thresholding method passed to :func:`apply_threshold`.

    Returns
    -------
    np.ndarray
        Preprocessed binary NumPy array ready for OCR.
    """
    denoised = denoise(image)
    return apply_threshold(denoised, method=method)
