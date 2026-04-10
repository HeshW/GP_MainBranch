"""Console helper for exercising the ChatManager pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manager.chat_manager import ChatManager
from manager.runtime import run_async


def parse_labs(labs_json: Optional[str], labs_file: Optional[str]) -> Optional[Dict[str, Any]]:
    if labs_json:
        try:
            return json.loads(labs_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid labs JSON: {exc}") from exc

    if labs_file:
        path = Path(labs_file)
        if not path.exists():
            raise FileNotFoundError(f"Labs file not found: {labs_file}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    return None


def print_result(result: Dict[str, Any]) -> None:
    print("--- Manager result ---")
    print(json.dumps(result, indent=2, default=str))
    print("----------------------")


def run_once(
    image: Optional[str] = None,
    labs: Optional[Dict[str, Any]] = None,
    manual_input: Optional[Dict[str, Any]] = None,
    use_rag: bool = False,
    faiss_index_dir: Optional[str] = None,
    gemini_key: Optional[str] = None,
    rag_top_k: int = 5,
    rag_translate_arabic: bool = True,
    use_finetuned_classifier: bool = False,
    finetuned_model_dir: Optional[str] = None,
    classifier_max_length: int = 256,
    classifier_translate_arabic: bool = True,
) -> Dict[str, Any]:
    manager = ChatManager(
        use_rag=use_rag,
        faiss_index_dir=faiss_index_dir,
        gemini_api_key=gemini_key,
        rag_top_k=rag_top_k,
        rag_translate_arabic=rag_translate_arabic,
        use_finetuned_classifier=use_finetuned_classifier,
        finetuned_model_dir=finetuned_model_dir,
        classifier_max_length=classifier_max_length,
        classifier_translate_arabic=classifier_translate_arabic,
    )
    return run_async(manager.run_pipeline(image=image, labs=labs, manual_input=manual_input))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Manager tester for OCR + diagnosis pipeline")
    parser.add_argument("--image", help="Path to a report image.")
    parser.add_argument("--labs", help="JSON string of labs, e.g. '{\"glucose\":140.0}'")
    parser.add_argument("--labs-file", help="Path to a JSON file with labs.")
    parser.add_argument("--symptoms", help="Optional symptom text for manual_input.")
    parser.add_argument("--use-symptom-parser", action="store_true", help="Run symptom parser/validator before diagnosis.")
    parser.add_argument("--rag", action="store_true", help="Enable RAG path in DiagnosisEngine.")
    parser.add_argument("--faiss-index-dir", help="FAISS index dir for RAG.")
    parser.add_argument("--gemini-key", help="Gemini API key for RAG.")
    parser.add_argument("--top-k", type=int, default=5, help="Top K for RAG.")
    parser.add_argument("--use-finetuned-classifier", action="store_true", help="Enable fine-tuned ClinicalBERT classifier.")
    parser.add_argument("--finetuned-model-dir", help="Path to saved fine-tuned model folder.")
    parser.add_argument("--classifier-max-length", type=int, default=256, help="Max token length for fine-tuned classifier.")
    parser.add_argument("--no-classifier-arabic-translate", action="store_true", help="Disable Arabic-to-English translation before classifier inference.")
    parser.add_argument("--no-rag-arabic-translate", action="store_true", help="Disable Arabic-to-English translation before RAG encoding.")
    parser.add_argument("--no-json", action="store_true", help="Disable JSON output formatting.")
    args = parser.parse_args(argv)

    try:
        labs_obj = parse_labs(args.labs, args.labs_file)
        manual_input = None
        if args.symptoms or labs_obj is not None:
            manual_input = {}
            if args.symptoms:
                manual_input["symptoms"] = args.symptoms
            if labs_obj is not None:
                manual_input["labs"] = labs_obj

        if args.use_symptom_parser and args.symptoms:
            from manager.diagnosis_adapter import run_from_symptoms

            result = run_from_symptoms(args.symptoms, low_confidence_threshold=0.7)
        else:
            result = run_once(
                image=args.image,
                labs=None if args.image else labs_obj,
                manual_input=manual_input if args.image is None else None,
                use_rag=args.rag,
                faiss_index_dir=args.faiss_index_dir,
                gemini_key=args.gemini_key,
                rag_top_k=args.top_k,
                rag_translate_arabic=not args.no_rag_arabic_translate,
                use_finetuned_classifier=args.use_finetuned_classifier,
                finetuned_model_dir=args.finetuned_model_dir,
                classifier_max_length=args.classifier_max_length,
                classifier_translate_arabic=not args.no_classifier_arabic_translate,
            )

        if args.no_json:
            print(result)
        else:
            print_result(result)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
