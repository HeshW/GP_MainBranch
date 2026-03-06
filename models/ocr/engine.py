"""
OCR Engine – Model A (v2).

Provides :class:`OCREngine`, the public interface consumed by the
Manager / Orchestrator (Model D).  Text extraction is performed by
PaddleOCR; the raw OCR output is then parsed with the regex patterns
defined in :mod:`models.ocr.patterns` to produce a structured,
JSON-serialisable result.

v2 improvements over v1
-----------------------
* **Multi-match extraction** – ``finditer`` replaces ``search`` so every
  occurrence of a lab synonym is examined.  When more than one match is found
  the *last* match wins (deterministic duplicate policy) and a warning is
  emitted with the match count.
* **Broadened unit capture** – units that contain digits and ``^`` (e.g.
  ``10^3/µL``, ``x10^3/uL``, ``10^9/L``) are now recognised.
* **Multi-pass parsing** – extraction is attempted on (1) the full normalised
  text, (2) each individual line, and (3) a cross-line fallback that joins a
  label line with a value-only next line.
* **``source_match`` field** – every lab entry now includes the exact substring
  (or joined lines) that was used to derive the value/unit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .patterns import LAB_LABEL_PATTERNS, LAB_PATTERNS, LEADING_VALUE_PATTERN, SYNONYM_MAP
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

# Each lab entry carries value, unit, and the exact source substring.
LabEntry = Dict[str, Any]   # {"value": float, "unit": str|None, "source_match": str}
OCRResult = Dict[str, Any]  # top-level return type of OCREngine.extract()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_text(raw: str) -> str:
    """Collapse multiple whitespace characters and strip leading/trailing space."""
    return re.sub(r"\s+", " ", raw).strip()


def _build_entry(
    canonical: str,
    match: "re.Match[str]",
    warnings: List[str],
) -> Optional[LabEntry]:
    """Build a :class:`LabEntry` from a regex *match*, or ``None`` on failure.

    Parameters
    ----------
    canonical:
        The canonical key being extracted (used only in warning messages).
    match:
        A ``re.Match`` object from one of the ``LAB_PATTERNS`` patterns.
    warnings:
        Mutable list to which any parsing warning is appended.

    Returns
    -------
    LabEntry | None
        A dict with ``value``, ``unit``, and ``source_match`` keys, or
        ``None`` if the numeric value could not be parsed.
    """
    raw_value: str = match.group(1)
    raw_unit: Optional[str] = match.group(2)
    source_match: str = match.group(0)

    # Normalise decimal separator (some locales use commas).
    try:
        value = float(raw_value.replace(",", "."))
    except ValueError:
        warnings.append(
            f"Could not parse numeric value for '{canonical}': '{raw_value}'"
        )
        return None

    return {
        "value": value,
        "unit": raw_unit if raw_unit else None,
        "source_match": source_match,
    }


def _extract_from_text_block(
    text_block: str,
    labs: Dict[str, LabEntry],
    warnings: List[str],
    *,
    skip_existing: bool = True,
) -> None:
    """Run all :data:`LAB_PATTERNS` against *text_block* and populate *labs*.

    Parameters
    ----------
    text_block:
        Pre-normalised text to search.
    labs:
        Accumulator dict; modified in place.
    warnings:
        Mutable warnings list; modified in place.
    skip_existing:
        When ``True`` (default for line-by-line pass), skip any canonical key
        that is already present in *labs*.  Set to ``False`` on the first
        (full-text) pass so that multi-match warnings are still generated.
    """
    for canonical, pattern in LAB_PATTERNS.items():
        if skip_existing and canonical in labs:
            continue

        matches = list(pattern.finditer(text_block))
        if not matches:
            continue

        if len(matches) > 1:
            warnings.append(
                f"Multiple matches ({len(matches)}) for '{canonical}'; "
                "keeping last match."
            )

        entry = _build_entry(canonical, matches[-1], warnings)
        if entry is not None:
            labs[canonical] = entry


def _cross_line_fallback(
    lines: List[str],
    labs: Dict[str, LabEntry],
    warnings: List[str],
) -> None:
    """Associate a label-only line with a numeric value on the following line.

    This last-resort pass fires only for canonical keys *not* found by the
    earlier passes.  It is conservative:

    * The current line must contain a recognised synonym (label-presence check).
    * The full lab pattern must *not* match the current line (confirming there
      is no value on the label line).
    * The next non-empty line must match :data:`LEADING_VALUE_PATTERN` (i.e. it
      is clearly just a number with an optional unit).

    A warning is always emitted when this fallback is used.
    """
    missing = set(LAB_PATTERNS) - set(labs)
    if not missing:
        return

    for i, raw_line in enumerate(lines):
        if not missing:
            break

        norm_line = _normalise_text(raw_line)
        if not norm_line:
            continue

        # Find the next non-empty line.
        next_index = i + 1
        while next_index < len(lines) and not _normalise_text(lines[next_index]):
            next_index += 1
        if next_index >= len(lines):
            continue

        next_norm = _normalise_text(lines[next_index])
        vm = LEADING_VALUE_PATTERN.match(next_norm)
        if vm is None:
            continue

        # Check each still-missing canonical key against this line.
        for canonical in list(missing):
            # Label must be present on this line …
            if LAB_LABEL_PATTERNS[canonical].search(norm_line) is None:
                continue
            # … but the full pattern (with value) must NOT match – if it did,
            # the earlier passes would have found it already.
            if LAB_PATTERNS[canonical].search(norm_line) is not None:
                continue

            raw_value: str = vm.group(1)
            raw_unit: Optional[str] = vm.group(2)
            try:
                value = float(raw_value.replace(",", "."))
            except ValueError:
                continue

            warnings.append(
                f"Cross-line fallback for '{canonical}': label on line "
                f"{i + 1}, value on line {next_index + 1}."
            )
            labs[canonical] = {
                "value": value,
                "unit": raw_unit if raw_unit else None,
                "source_match": f"{norm_line} → {next_norm}",
            }
            missing.discard(canonical)


def _parse_labs(text: str) -> tuple[Dict[str, LabEntry], List[str]]:
    """Extract structured lab values from *text* using a multi-pass strategy.

    Pass 1 – full normalised text
        All patterns are searched against the entire text with ``finditer``.
        When a canonical key appears more than once the *last* match is kept
        and a warning is emitted with the match count.

    Pass 2 – line-by-line
        For any canonical key not found in Pass 1, each individual line is
        searched.  This catches labs whose label and value happen to sit on
        the same OCR line but are separated from unrelated text.

    Pass 3 – cross-line fallback
        For any canonical key still missing, a heuristic checks whether a
        label-only line is immediately followed by a value-only line.  A
        warning is always emitted when this path is taken.

    Parameters
    ----------
    text:
        Raw OCR text (may contain newlines and OCR artefacts).

    Returns
    -------
    tuple[dict, list]
        A ``(labs, warnings)`` pair where *labs* maps canonical keys to
        ``{"value": float, "unit": str | None, "source_match": str}`` dicts
        and *warnings* is a list of human-readable strings about anything
        ambiguous.
    """
    labs: Dict[str, LabEntry] = {}
    warnings: List[str] = []

    # Pass 1: full normalised text (multi-match aware).
    normalised = _normalise_text(text)
    _extract_from_text_block(normalised, labs, warnings, skip_existing=False)

    # Pass 2: line-by-line for labs not yet found.
    if len(labs) < len(LAB_PATTERNS):
        for line in text.splitlines():
            norm_line = _normalise_text(line)
            if norm_line:
                _extract_from_text_block(norm_line, labs, warnings, skip_existing=True)

    # Pass 3: cross-line fallback.
    _cross_line_fallback(text.splitlines(), labs, warnings)

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
    {'value': 95.0, 'unit': 'mg/dL', 'source_match': 'Glucose: 95 mg/dL'}
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
                ``{"value": float, "unit": str | None, "source_match": str}``
                entries.  ``source_match`` is the exact substring (or joined
                lines for cross-line matches) used to derive the value/unit.

            ``raw_text``
                Full concatenated OCR text, useful for debugging.

            ``warnings``
                List of strings describing any issues encountered during
                extraction (empty if everything parsed cleanly).  Includes
                duplicate-match warnings and cross-line fallback notices.

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
        Same structure as :meth:`OCREngine.extract`.  Each entry in ``labs``
        contains ``value`` (float), ``unit`` (str | None), and
        ``source_match`` (str – the exact matched substring).
    """
    labs, warnings = _parse_labs(text)
    return {
        "labs": labs,
        "raw_text": _normalise_text(text),
        "warnings": warnings,
    }
