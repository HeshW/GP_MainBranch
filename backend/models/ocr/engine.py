"""OCR Engine - Model A (v2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import image_io as _image_io
from .fields import extract_fields_and_sections
from .parsing import attach_confidences, extract_labs_from_text, normalise_text, parse_labs
from .raw_ocr import collect_raw_ocr, collect_text
from .types import ImageInput, OCRResult
from .utils import preprocess

try:
    from paddleocr import PaddleOCR as _PaddleOCR  # type: ignore[import]
    _PADDLEOCR_AVAILABLE = True
except Exception:  # pragma: no cover
    _PaddleOCR = None  # type: ignore[assignment, misc]
    _PADDLEOCR_AVAILABLE = False

# Backward-compatible aliases for tests/internal imports.
# Keep these module-level names so existing tests can monkeypatch them.
_PILImage = _image_io._PILImage
_PIL_AVAILABLE = _image_io._PIL_AVAILABLE


def _to_rgb_array(image: Any) -> np.ndarray:
    _image_io._PILImage = _PILImage
    _image_io._PIL_AVAILABLE = _PIL_AVAILABLE
    return _image_io.to_rgb_array(image)


def _ensure_hwc3_uint8(img: np.ndarray) -> np.ndarray:
    return _image_io.ensure_hwc3_uint8(img)


_normalise_text = normalise_text
_parse_labs = parse_labs
_collect_raw_ocr = collect_raw_ocr


class OCREngine:
    """Extract lab values from a medical report image using PaddleOCR."""

    def __init__(
        self,
        *,
        lang: str = "en",
        use_angle_cls: bool = True,
        preprocess_image: bool = True,
    ) -> None:
        if not _PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR is not installed. Install it with: pip install paddleocr paddlepaddle"
            )
        try:
            self._ocr = _PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
        except ValueError as exc:
            if "Unknown argument: show_log" in str(exc):
                self._ocr = _PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
            else:
                raise
        self._preprocess = preprocess_image

    def extract(self, image: ImageInput) -> OCRResult:
        img_array = _to_rgb_array(image)
        try:
            ocr_input: np.ndarray = preprocess(img_array) if self._preprocess else img_array
            ocr_input = _ensure_hwc3_uint8(ocr_input)
            try:
                result = self._ocr.ocr(ocr_input, cls=True)
            except TypeError as exc:
                if "unexpected keyword argument 'cls'" in str(exc):
                    result = self._ocr.ocr(ocr_input)
                else:
                    raise
        except Exception as exc:
            label = str(image) if isinstance(image, (str, Path)) else f"<{type(image).__name__}>"
            raise RuntimeError(f"PaddleOCR failed to process '{label}': {exc}") from exc

        raw_text = collect_text(result)
        labs, warnings = parse_labs(raw_text)
        fields, sections = extract_fields_and_sections(raw_text)
        raw_ocr = collect_raw_ocr(result)
        attach_confidences(labs, raw_ocr, warnings)

        return {
            "labs": labs,
            "raw_text": raw_text,
            "warnings": warnings,
            "fields": fields,
            "sections": sections,
            "raw_ocr": raw_ocr,
        }


def extract_from_text(text: str) -> OCRResult:
    return extract_labs_from_text(text)
