"""manager/diagnosis_adapter.py

Adapter connecting ``OCREngine.extract()`` output to the diagnosis engine.

CLI usage
---------
::

    python -m manager.diagnosis_adapter path/to/report.png
    python -m manager.diagnosis_adapter path/to/report.png --rag \\
        --faiss-index-dir /data/faiss --gemini-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from models.diagnosis import diagnose


def adapt_and_diagnose(
    ocr_result: Dict[str, Any],
    *,
    use_rag: bool = False,
    faiss_index_dir: Optional[Path | str] = None,
    gemini_api_key: Optional[str] = None,
    rag_top_k: int = 5,
) -> Dict[str, Any]:
    """Run the diagnosis engine on an ``OCREngine.extract()`` result dict.

    This is the canonical integration point between the OCR and diagnosis
    layers.  It passes the full OCR result through unchanged; the engine
    reads ``ocr_result["labs"]``, ``ocr_result["fields"]``, and
    ``ocr_result["sections"]`` as needed.

    Parameters
    ----------
    ocr_result:
        The dict returned by ``OCREngine.extract()``.
    use_rag:
        Enable the ClinicalBERT + FAISS + Gemini RAG path.
    faiss_index_dir:
        Directory containing ``medical_cases.index`` and
        ``metadata_mapping.pkl`` (required when *use_rag* is ``True``).
    gemini_api_key:
        Google Gemini API key (required when *use_rag* is ``True``).
    rag_top_k:
        Number of similar cases to retrieve via FAISS.

    Returns
    -------
    dict
        Diagnosis result with keys ``findings``, ``summary``,
        ``disclaimer``, and optionally ``rag_response`` /
        ``retrieved_cases``.
    """
    return diagnose(
        ocr_result,
        use_rag=use_rag,
        faiss_index_dir=faiss_index_dir,
        gemini_api_key=gemini_api_key,
        rag_top_k=rag_top_k,
    )


def run_from_image(
    image_path: Path | str,
    *,
    use_rag: bool = False,
    faiss_index_dir: Optional[Path | str] = None,
    gemini_api_key: Optional[str] = None,
    rag_top_k: int = 5,
) -> Dict[str, Any]:
    """Extract OCR from *image_path* then diagnose.

    ``OCREngine`` (which requires PaddleOCR) is imported lazily so that
    unit tests importing this module are not blocked by heavy dependencies.

    Returns
    -------
    dict
        ``ocr``        – full ``OCREngine.extract()`` result
        ``diagnosis``  – ``adapt_and_diagnose()`` result
    """
    # Lazy import avoids loading PaddleOCR at module import time.
    from models.ocr.engine import OCREngine  # noqa: PLC0415

    engine = OCREngine()
    ocr_result = engine.extract(Path(image_path))
    return {
        "ocr": ocr_result,
        "diagnosis": adapt_and_diagnose(
            ocr_result,
            use_rag=use_rag,
            faiss_index_dir=faiss_index_dir,
            gemini_api_key=gemini_api_key,
            rag_top_k=rag_top_k,
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m manager.diagnosis_adapter",
        description=(
            "Extract lab values from a medical image and run the diagnosis engine."
        ),
    )
    p.add_argument(
        "image",
        type=Path,
        help="Path to the medical report image (PNG, JPEG, …).",
    )
    p.add_argument(
        "--rag",
        action="store_true",
        help="Enable the ClinicalBERT + FAISS + Gemini RAG path.",
    )
    p.add_argument(
        "--faiss-index-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing medical_cases.index and "
            "metadata_mapping.pkl (required with --rag)."
        ),
    )
    p.add_argument(
        "--gemini-key",
        default=None,
        metavar="KEY",
        help=(
            "Google Gemini API key.  Falls back to the "
            "GEMINI_API_KEY environment variable."
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of similar cases to retrieve via FAISS (default: 5).",
    )
    p.add_argument(
        "--no-pretty",
        action="store_true",
        help="Emit compact JSON instead of indented output.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")

    result = run_from_image(
        args.image,
        use_rag=args.rag,
        faiss_index_dir=args.faiss_index_dir,
        gemini_api_key=gemini_key,
        rag_top_k=args.top_k,
    )

    indent = None if args.no_pretty else 2
    print(json.dumps(result, indent=indent, default=str))


if __name__ == "__main__":
    main()
