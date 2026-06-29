"""Health check for the full diagnosis pipeline in offline evaluation mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from manager.chat_manager import ChatManager
from models.diagnosis.rules import RULES, SYMPTOM_RULES


DEFAULT_FAISS_DIR = Path("backend/artifacts/artifacts/faiss_data_targeted")
DEFAULT_CLASSIFIER_DIR = Path("backend/artifacts/artifacts/clinicalbert_classifier_targeted")


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Check full pipeline readiness without exposing secrets.")
    parser.add_argument("--faiss-index-dir", type=Path, default=Path(settings.faiss_index_dir or DEFAULT_FAISS_DIR))
    parser.add_argument(
        "--clinicalbert-model-dir",
        type=Path,
        default=Path(settings.clinicalbert_model_dir or DEFAULT_CLASSIFIER_DIR),
    )
    parser.add_argument(
        "--finetuned-model-dir",
        type=Path,
        default=Path(settings.finetuned_model_dir or DEFAULT_CLASSIFIER_DIR),
    )
    parser.add_argument("--top-k", type=int, default=settings.rag_top_k)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true")
    return parser.parse_args()


def _artifact_status(args: argparse.Namespace) -> dict[str, Any]:
    classifier_required = ["config.json", "label_map.json", "tokenizer.json", "tokenizer_config.json"]
    classifier_missing = [name for name in classifier_required if not (args.finetuned_model_dir / name).exists()]
    if not any((args.finetuned_model_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
        classifier_missing.append("model weights")
    rag_required = ["medical_cases.index", "metadata_mapping.pkl"]
    rag_missing = [name for name in rag_required if not (args.faiss_index_dir / name).exists()]
    label_count = 0
    label_map_path = args.finetuned_model_dir / "label_map.json"
    if label_map_path.exists():
        with label_map_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        label_count = len(payload.get("label2id") or payload.get("label_to_id") or {})
    return {
        "classifier": {
            "path": str(args.finetuned_model_dir),
            "exists": args.finetuned_model_dir.is_dir(),
            "missing_files": classifier_missing,
            "label_count": label_count,
        },
        "rag": {
            "path": str(args.faiss_index_dir),
            "exists": args.faiss_index_dir.is_dir(),
            "missing_files": rag_missing,
        },
        "clinicalbert_model_dir": str(args.clinicalbert_model_dir),
        "rules": {
            "lab_rules_count": len(RULES),
            "symptom_rules_count": len(SYMPTOM_RULES),
            "available": bool(RULES or SYMPTOM_RULES),
        },
    }


def _has_output_structure(result: dict[str, Any]) -> dict[str, bool]:
    diagnosis = result.get("diagnosis") or {}
    return {
        "final_diagnosis": "final_diagnosis" in diagnosis,
        "confidence": "confidence" in (diagnosis.get("final_diagnosis") or {}),
        "candidates": bool(diagnosis.get("diagnostic_candidates") or diagnosis.get("classifier_prediction")),
        "safety": "safety" in diagnosis,
    }


async def _run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    manager = ChatManager(
        use_rag=True,
        faiss_index_dir=args.faiss_index_dir,
        clinicalbert_model_dir=args.clinicalbert_model_dir,
        gemini_api_key="",
        llm_api_key="",
        rag_top_k=args.top_k,
        rag_translate_arabic=False,
        use_finetuned_classifier=True,
        finetuned_model_dir=args.finetuned_model_dir,
        classifier_translate_arabic=False,
        enable_therapy=False,
    )
    in_scope = await manager.run_from_symptoms(
        "Chest pressure with exertion that improves with rest and no fever."
    )
    out_of_scope = await manager.run_from_symptoms(
        "Frequent urination, excessive thirst, weight loss, and glucose 240 mg/dL."
    )
    return {
        "in_scope_case": {
            "status": in_scope.get("status"),
            "final_diagnosis": ((in_scope.get("diagnosis") or {}).get("final_diagnosis") or {}).get("diagnosis"),
            "structure": _has_output_structure(in_scope),
        },
        "out_of_scope_safety_case": {
            "status": out_of_scope.get("status"),
            "final_diagnosis": ((out_of_scope.get("diagnosis") or {}).get("final_diagnosis") or {}).get("diagnosis"),
            "structure": _has_output_structure(out_of_scope),
            "rag_scope_status": ((out_of_scope.get("diagnosis") or {}).get("rag_metadata") or {}).get("rag_scope_status"),
        },
        "external_llm_required_for_evaluation_mode": False,
    }


async def main_async() -> None:
    args = parse_args()
    payload: dict[str, Any] = {
        "status": "ok",
        "artifacts": _artifact_status(args),
        "smoke": None,
    }
    blockers = []
    if payload["artifacts"]["classifier"]["missing_files"]:
        blockers.append("classifier files missing")
    if payload["artifacts"]["rag"]["missing_files"]:
        blockers.append("rag files missing")
    if not payload["artifacts"]["rules"]["available"]:
        blockers.append("rules unavailable")

    if args.skip_model_load:
        payload["status"] = "ok" if not blockers else "blocked"
        payload["blockers"] = blockers
    elif blockers:
        payload["status"] = "blocked"
        payload["blockers"] = blockers
    else:
        try:
            payload["smoke"] = await _run_smoke(args)
            structures = [
                payload["smoke"]["in_scope_case"]["structure"],
                payload["smoke"]["out_of_scope_safety_case"]["structure"],
            ]
            if not all(all(item.values()) for item in structures):
                payload["status"] = "error"
                payload["blockers"] = ["pipeline output structure missing required fields"]
        except Exception as exc:
            payload["status"] = "error"
            payload["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    if payload.get("status") != "ok":
        raise SystemExit(1)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
