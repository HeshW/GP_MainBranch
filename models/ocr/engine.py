"""
OCR Engine – Model A.

Provides :class:`OCREngine`, the public interface consumed by the
Manager / Orchestrator (Model D).  Text extraction is performed by
PaddleOCR; the raw OCR output is then parsed with the regex patterns
defined in :mod:`models.ocr.patterns` to produce a structured,
JSON-serialisable result.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .patterns import LAB_PATTERNS, SYNONYM_MAP
from .utils import preprocess

# PaddleOCR is an optional heavy dependency; we import it lazily so that
# the rest of the module (and especially the tests) can be imported without
# the full PaddlePaddle stack being installed.
try:
    from paddleocr import PaddleOCR as _PaddleOCR  # type: ignore[import]
    _PADDLEOCR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PaddleOCR = None  # type: ignore[assignment, misc]
    _PADDLEOCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ImageInput = Union[str, Path]

LabEntry = Dict[str, Any]   # {"value": ..., "unit": ..., "confidence": ...}
OCRResult = Dict[str, Any]  # top-level return type of OCREngine.extract()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_text(raw: str) -> str:
    """Collapse multiple whitespace characters and strip leading/trailing space."""
    return re.sub(r"\s+", " ", raw).strip()


def _parse_labs(text: str) -> tuple[Dict[str, LabEntry], List[str]]:
    """Extract structured lab values from *text* using regex patterns.

    Parameters
    ----------
    text:
        Raw OCR text (may contain newlines and OCR artefacts).

    Returns
    -------
    tuple[dict, list]
        A ``(labs, warnings)`` pair where *labs* maps canonical keys to
        ``{"value": float, "unit": str | None}`` dicts and *warnings* is a
        list of human-readable strings about anything ambiguous.
    """
    labs: Dict[str, LabEntry] = {}
    warnings: List[str] = []
    normalised = _normalise_text(text)

    for canonical, pattern in LAB_PATTERNS.items():
        match = pattern.search(normalised)
        if match is None:
            continue

        raw_value, raw_unit = match.group(1), match.group(2)

        # Normalise decimal separator (some locales use commas).
        try:
            value = float(raw_value.replace(",", "."))
        except ValueError:
            warnings.append(
                f"Could not parse numeric value for '{canonical}': '{raw_value}'"
            )
            continue

        labs[canonical] = {
            "value": value,
            "unit": raw_unit if raw_unit else None,
        }

    return labs, warnings


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class OCREngine:
    """Extracts lab values from a medical-report image using PaddleOCR.

    Parameters
    ----------
    lang:
        Language code forwarded to PaddleOCR (default ``"en"``).
    use_angle_cls:
        Whether to enable PaddleOCR's angle classifier (default ``True``).
    preprocess_image:
        If ``True`` (default), run the OpenCV preprocessing pipeline on the
        input image before passing it to PaddleOCR.  This typically improves
        accuracy on low-quality scans.

    Examples
    --------
    >>> engine = OCREngine()
    >>> result = engine.extract("path/to/lab_report.png")
    >>> print(result["labs"]["glucose"])
    {'value': 95.0, 'unit': 'mg/dL'}
    """

    def __init__(
        self,
        *,
        lang: str = "en",
        use_angle_cls: bool = True,
        preprocess_image: bool = True,
    ) -> None:
        if not _PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Install it with: pip install paddleocr paddlepaddle"
            )
        self._ocr = _PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
        self._preprocess = preprocess_image

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def extract(self, image: ImageInput) -> OCRResult:
        """Run OCR on *image* and return structured lab values.

        Parameters
        ----------
        image:
            Path to the image file (``str`` or :class:`~pathlib.Path`).
            JPEG, PNG and most common formats are supported.

        Returns
        -------
        dict
            A JSON-serialisable dictionary with the following keys:

            ``labs``
                Dict mapping canonical lab names (e.g. ``"glucose"``) to
                ``{"value": float, "unit": str | None}`` entries.

            ``raw_text``
                Full concatenated OCR text, useful for debugging.

            ``warnings``
                List of strings describing any issues encountered during
                extraction (empty if everything parsed cleanly).

        Raises
        ------
        FileNotFoundError
            If *image* does not exist on the filesystem.
        RuntimeError
            If PaddleOCR fails to process the image.
        """
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        try:
            ocr_input: Union[str, Any]
            if self._preprocess:
                preprocessed = preprocess(path)
                ocr_input = preprocessed  # PaddleOCR accepts numpy arrays
            else:
                ocr_input = str(path)

            result = self._ocr.ocr(ocr_input, cls=True)
        except Exception as exc:
            raise RuntimeError(
                f"PaddleOCR failed to process '{path}': {exc}"
            ) from exc

        raw_text = self._collect_text(result)
        labs, warnings = _parse_labs(raw_text)

        return {
            "labs": labs,
            "raw_text": raw_text,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_text(paddle_result: Any) -> str:
        """Flatten PaddleOCR's nested result structure into a single string.

        PaddleOCR returns a list of pages, each page being a list of lines,
        where each line is ``[bounding_box, (text, confidence)]``.
        """
        lines: List[str] = []
        if not paddle_result:
            return ""
        for page in paddle_result:
            if not page:
                continue
            for line in page:
                try:
                    text, _confidence = line[1]
                    lines.append(text)
                except (IndexError, TypeError, ValueError):
                    # Malformed line – skip silently.
                    continue
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience function (allows from models.ocr import extract)
# ---------------------------------------------------------------------------

def extract_from_text(text: str) -> OCRResult:
    """Parse lab values directly from *text* without running OCR.

    This is the primary entry-point used by tests and any caller that has
    already obtained raw OCR text by other means.

    Parameters
    ----------
    text:
        OCR output text (plain string).

    Returns
    -------
    dict
        Same structure as :meth:`OCREngine.extract`.
    """
    labs, warnings = _parse_labs(text)
    return {
        "labs": labs,
        "raw_text": _normalise_text(text),
        "warnings": warnings,
    }
