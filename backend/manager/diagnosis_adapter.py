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

from manager.chat_manager import ChatManager
from models.diagnosis import diagnose


def adapt_and_diagnose(
    ocr_result: Dict[str, Any],
    *,
    use_rag: bool = False,
    faiss_index_dir: Optional[Path | str] = None,
    gemini_api_key: Optional[str] = None,
    rag_top_k: int = 5,
    rag_translate_arabic: bool = True,
    use_finetuned_classifier: bool = False,
    finetuned_model_dir: Optional[Path | str] = None,
    classifier_max_length: int = 256,
    classifier_translate_arabic: bool = True,
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
        rag_translate_arabic=rag_translate_arabic,
        use_finetuned_classifier=use_finetuned_classifier,
        finetuned_model_dir=finetuned_model_dir,
        classifier_max_length=classifier_max_length,
        classifier_translate_arabic=classifier_translate_arabic,
    )


def run_from_labs(
    labs: Dict[str, Any],
    *,
    use_rag: bool = False,
    faiss_index_dir: Optional[Path | str] = None,
    gemini_api_key: Optional[str] = None,
    rag_top_k: int = 5,
    rag_translate_arabic: bool = True,
    use_finetuned_classifier: bool = False,
    finetuned_model_dir: Optional[Path | str] = None,
    classifier_max_length: int = 256,
    classifier_translate_arabic: bool = True,
) -> Dict[str, Any]:
    """Run diagnosis directly from a lab dict, without OCR."""
    if not isinstance(labs, dict):
        raise TypeError("labs must be a dict mapping lab keys to values")

    manager = ChatManager(
        use_rag=use_rag,
        faiss_index_dir=faiss_index_dir,
        gemini_api_key=gemini_api_key,
        rag_top_k=rag_top_k,
        rag_translate_arabic=rag_translate_arabic,
        use_finetuned_classifier=use_finetuned_classifier,
        finetuned_model_dir=finetuned_model_dir,
        classifier_max_length=classifier_max_length,
        classifier_translate_arabic=classifier_translate_arabic,
    )

    return manager.run_pipeline(labs=labs)


def run_from_image(
    image_path: Path | str,
    *,
    use_rag: bool = False,
    faiss_index_dir: Optional[Path | str] = None,
    gemini_api_key: Optional[str] = None,
    rag_top_k: int = 5,
    rag_translate_arabic: bool = True,
    use_finetuned_classifier: bool = False,
    finetuned_model_dir: Optional[Path | str] = None,
    classifier_max_length: int = 256,
    classifier_translate_arabic: bool = True,
) -> Dict[str, Any]:
    """Extract OCR from *image_path* then diagnose using ChatManager."""
    manager = ChatManager(
        use_rag=use_rag,
        faiss_index_dir=faiss_index_dir,
        gemini_api_key=gemini_api_key,
        rag_top_k=rag_top_k,
        rag_translate_arabic=rag_translate_arabic,
        use_finetuned_classifier=use_finetuned_classifier,
        finetuned_model_dir=finetuned_model_dir,
        classifier_max_length=classifier_max_length,
        classifier_translate_arabic=classifier_translate_arabic,
    )
    return manager.run_pipeline(image=image_path)


def run_from_symptoms(
    text: str,
    *,
    low_confidence_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Parse free text symptoms and run through manager pipeline."""
    manager = ChatManager()
    return manager.run_from_symptoms(text, low_confidence_threshold=low_confidence_threshold)


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
