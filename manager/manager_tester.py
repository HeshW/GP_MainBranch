"""Console helper for testing the ChatManager pipeline in PR#3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import os, sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from manager.chat_manager import ChatManager


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
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
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
) -> Dict[str, Any]:
    manager = ChatManager(
        use_rag=use_rag,
        faiss_index_dir=faiss_index_dir,
        gemini_api_key=gemini_key,
        rag_top_k=rag_top_k,
    )

    return manager.run_pipeline(image=image, labs=labs, manual_input=manual_input)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Manager tester for OCR + Diagnosis pipeline")
    parser.add_argument("--image", help="Path to a report image.")
    parser.add_argument("--labs", help="JSON string of labs, e.g. '{\"glucose\":140.0}'")
    parser.add_argument("--labs-file", help="Path to a JSON file with labs.")
    parser.add_argument("--symptoms", help="Optional symptom text for manual_input.")
    parser.add_argument("--rag", action="store_true", help="Enable RAG path in DiagnosisEngine.")
    parser.add_argument("--faiss-index-dir", help="FAISS index dir for RAG.")
    parser.add_argument("--gemini-key", help="Gemini API key for RAG.")
    parser.add_argument("--top-k", type=int, default=5, help="Top K for RAG.")
    parser.add_argument("--no-json", action="store_true", help="Disable JSON output formatting (debug print only).")
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

        result = run_once(
            image=args.image,
            labs=None if args.image else labs_obj,
            manual_input=manual_input if args.image is None else None,
            use_rag=args.rag,
            faiss_index_dir=args.faiss_index_dir,
            gemini_key=args.gemini_key,
            rag_top_k=args.top_k,
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
