"""manager/diagnosis_adapter.py

Adapter connecting OCR extraction to the diagnosis pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from manager.chat_manager import ChatManager
from manager.runtime import run_async
from models.diagnosis import diagnose


def adapt_and_diagnose(
    ocr_result: Dict[str, Any],
    *,
    use_rag: bool = False,
    faiss_index_dir: Optional[Path | str] = None,
    clinicalbert_model_dir: Optional[Path | str] = None,
    llm_provider: str = "gemini",
    llm_api_key: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_site_url: Optional[str] = None,
    openrouter_app_name: str = "GP Medical Analysis",
    gemini_api_key: Optional[str] = None,
    gemini_model_name: str = "gemini-2.5-flash-lite",
    rag_top_k: int = 5,
    rag_translate_arabic: bool = True,
    use_finetuned_classifier: bool = False,
    finetuned_model_dir: Optional[Path | str] = None,
    classifier_max_length: int = 256,
    classifier_translate_arabic: bool = True,
) -> Dict[str, Any]:
    return run_async(
        diagnose(
            ocr_result,
            use_rag=use_rag,
            faiss_index_dir=faiss_index_dir,
            clinicalbert_model_dir=clinicalbert_model_dir,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            openrouter_base_url=openrouter_base_url,
            openrouter_site_url=openrouter_site_url,
            openrouter_app_name=openrouter_app_name,
            gemini_api_key=gemini_api_key,
            gemini_model_name=gemini_model_name,
            rag_top_k=rag_top_k,
            rag_translate_arabic=rag_translate_arabic,
            use_finetuned_classifier=use_finetuned_classifier,
            finetuned_model_dir=finetuned_model_dir,
            classifier_max_length=classifier_max_length,
            classifier_translate_arabic=classifier_translate_arabic,
        )
    )


def _build_manager(
    *,
    use_rag: bool = False,
    faiss_index_dir: Optional[Path | str] = None,
    clinicalbert_model_dir: Optional[Path | str] = None,
    llm_provider: str = "gemini",
    llm_api_key: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_site_url: Optional[str] = None,
    openrouter_app_name: str = "GP Medical Analysis",
    gemini_api_key: Optional[str] = None,
    gemini_model_name: str = "gemini-2.5-flash-lite",
    rag_top_k: int = 5,
    rag_translate_arabic: bool = True,
    use_finetuned_classifier: bool = False,
    finetuned_model_dir: Optional[Path | str] = None,
    classifier_max_length: int = 256,
    classifier_translate_arabic: bool = True,
) -> ChatManager:
    return ChatManager(
        use_rag=use_rag,
        faiss_index_dir=faiss_index_dir,
        clinicalbert_model_dir=clinicalbert_model_dir,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model_name=llm_model_name,
        openrouter_base_url=openrouter_base_url,
        openrouter_site_url=openrouter_site_url,
        openrouter_app_name=openrouter_app_name,
        gemini_api_key=gemini_api_key,
        gemini_model_name=gemini_model_name,
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
    clinicalbert_model_dir: Optional[Path | str] = None,
    llm_provider: str = "gemini",
    llm_api_key: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_site_url: Optional[str] = None,
    openrouter_app_name: str = "GP Medical Analysis",
    gemini_api_key: Optional[str] = None,
    gemini_model_name: str = "gemini-2.5-flash-lite",
    rag_top_k: int = 5,
    rag_translate_arabic: bool = True,
    use_finetuned_classifier: bool = False,
    finetuned_model_dir: Optional[Path | str] = None,
    classifier_max_length: int = 256,
    classifier_translate_arabic: bool = True,
) -> Dict[str, Any]:
    if not isinstance(labs, dict):
        raise TypeError("labs must be a dict mapping lab keys to values")

    manager = _build_manager(
        use_rag=use_rag,
        faiss_index_dir=faiss_index_dir,
        clinicalbert_model_dir=clinicalbert_model_dir,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model_name=llm_model_name,
        openrouter_base_url=openrouter_base_url,
        openrouter_site_url=openrouter_site_url,
        openrouter_app_name=openrouter_app_name,
        gemini_api_key=gemini_api_key,
        gemini_model_name=gemini_model_name,
        rag_top_k=rag_top_k,
        rag_translate_arabic=rag_translate_arabic,
        use_finetuned_classifier=use_finetuned_classifier,
        finetuned_model_dir=finetuned_model_dir,
        classifier_max_length=classifier_max_length,
        classifier_translate_arabic=classifier_translate_arabic,
    )
    return run_async(manager.run_pipeline(labs=labs))


def run_from_image(
    image_path: Path | str,
    *,
    use_rag: bool = False,
    faiss_index_dir: Optional[Path | str] = None,
    clinicalbert_model_dir: Optional[Path | str] = None,
    llm_provider: str = "gemini",
    llm_api_key: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_site_url: Optional[str] = None,
    openrouter_app_name: str = "GP Medical Analysis",
    gemini_api_key: Optional[str] = None,
    gemini_model_name: str = "gemini-2.5-flash-lite",
    rag_top_k: int = 5,
    rag_translate_arabic: bool = True,
    use_finetuned_classifier: bool = False,
    finetuned_model_dir: Optional[Path | str] = None,
    classifier_max_length: int = 256,
    classifier_translate_arabic: bool = True,
) -> Dict[str, Any]:
    manager = _build_manager(
        use_rag=use_rag,
        faiss_index_dir=faiss_index_dir,
        clinicalbert_model_dir=clinicalbert_model_dir,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model_name=llm_model_name,
        openrouter_base_url=openrouter_base_url,
        openrouter_site_url=openrouter_site_url,
        openrouter_app_name=openrouter_app_name,
        gemini_api_key=gemini_api_key,
        gemini_model_name=gemini_model_name,
        rag_top_k=rag_top_k,
        rag_translate_arabic=rag_translate_arabic,
        use_finetuned_classifier=use_finetuned_classifier,
        finetuned_model_dir=finetuned_model_dir,
        classifier_max_length=classifier_max_length,
        classifier_translate_arabic=classifier_translate_arabic,
    )
    return run_async(manager.run_pipeline(image=image_path))


def run_from_symptoms(
    text: str,
    *,
    low_confidence_threshold: float = 0.7,
) -> Dict[str, Any]:
    manager = ChatManager()
    return run_async(
        manager.run_from_symptoms(
            text,
            low_confidence_threshold=low_confidence_threshold,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m manager.diagnosis_adapter",
        description="Extract lab values from a medical image and run the diagnosis engine.",
    )
    parser.add_argument("image", type=Path, help="Path to the medical report image.")
    parser.add_argument("--rag", action="store_true", help="Enable the ClinicalBERT + FAISS + LLM-assisted RAG path.")
    parser.add_argument("--faiss-index-dir", type=Path, default=None, metavar="DIR")
    parser.add_argument("--llm-provider", default=None, metavar="PROVIDER")
    parser.add_argument("--llm-key", default=None, metavar="KEY")
    parser.add_argument("--llm-model", default=None, metavar="MODEL")
    parser.add_argument("--openrouter-base-url", default=None, metavar="URL")
    parser.add_argument("--openrouter-site-url", default=None, metavar="URL")
    parser.add_argument("--openrouter-app-name", default=None, metavar="NAME")
    parser.add_argument("--gemini-key", default=None, metavar="KEY")
    parser.add_argument("--top-k", type=int, default=5, metavar="N")
    parser.add_argument("--no-pretty", action="store_true", help="Emit compact JSON instead of indented output.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    llm_provider = args.llm_provider or os.environ.get("LLM_PROVIDER") or "gemini"
    llm_key = (
        args.llm_key
        or os.environ.get("LLM_API_KEY")
        or args.gemini_key
        or os.environ.get("GEMINI_API_KEY")
    )
    llm_model = (
        args.llm_model
        or os.environ.get("LLM_MODEL_NAME")
        or os.environ.get("GEMINI_MODEL_NAME")
    )
    result = run_from_image(
        args.image,
        use_rag=args.rag,
        faiss_index_dir=args.faiss_index_dir,
        llm_provider=llm_provider,
        llm_api_key=llm_key,
        llm_model_name=llm_model,
        openrouter_base_url=args.openrouter_base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
        openrouter_site_url=args.openrouter_site_url or os.environ.get("OPENROUTER_SITE_URL"),
        openrouter_app_name=args.openrouter_app_name or os.environ.get("OPENROUTER_APP_NAME") or "GP Medical Analysis",
        gemini_api_key=llm_key,
        gemini_model_name=llm_model or "gemini-2.5-flash-lite",
        rag_top_k=args.top_k,
    )

    indent = None if args.no_pretty else 2
    print(json.dumps(result, indent=indent, default=str))


if __name__ == "__main__":
    main()
