"""Validate AI assets/configuration and run a few smoke checks.

Usage:
    .\\.venv_rag\\Scripts\\python.exe backend\\scripts\\validate_ai_setup.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from manager.chat_manager import ChatManager
from models.diagnosis.rag import MedicalCaseSearcher


def faiss_metadata_exists(faiss_dir: Path) -> bool:
    return (faiss_dir / "metadata_mapping.json").exists() or (faiss_dir / "metadata_mapping.pkl").exists()


def find_missing_files(model_dir: Path) -> list[str]:
    expected = [
        "config.json",
        "tokenizer_config.json",
        "label_map.json",
    ]
    missing = [name for name in expected if not (model_dir / name).exists()]

    has_weights = any(
        (model_dir / name).exists()
        for name in ["model.safetensors", "pytorch_model.bin"]
    )
    if not has_weights:
        missing.append("model weights (model.safetensors or pytorch_model.bin)")
    return missing


def find_missing_clinicalbert_files(model_dir: Path) -> list[str]:
    expected = ["config.json", "tokenizer_config.json"]
    missing = [name for name in expected if not (model_dir / name).exists()]

    has_tokenizer = any(
        (model_dir / name).exists()
        for name in ["tokenizer.json", "vocab.txt"]
    )
    if not has_tokenizer:
        missing.append("tokenizer.json or vocab.txt")

    has_weights = any(
        (model_dir / name).exists()
        for name in ["model.safetensors", "pytorch_model.bin"]
    )
    if not has_weights:
        missing.append("model weights (model.safetensors or pytorch_model.bin)")

    return missing


def inspect_faiss_metadata_quality(faiss_dir: Path, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "metadata_has_natural_text": None,
            "metadata_warning": None,
        }

    try:
        searcher = MedicalCaseSearcher(faiss_dir)
    except Exception as exc:
        return {
            "metadata_has_natural_text": False,
            "metadata_warning": f"Could not inspect FAISS metadata: {exc}",
        }

    has_natural_text = searcher.metadata_has_natural_text()
    warning = None
    if not has_natural_text:
        warning = (
            "FAISS metadata appears to contain encoded evidence values instead of natural clinical text. "
            "RAG retrieval will be technically functional but clinically weak until the index is rebuilt."
        )
    return {
        "metadata_has_natural_text": has_natural_text,
        "metadata_warning": warning,
    }


async def run_smoke_tests(manager: ChatManager) -> dict[str, Any]:
    labs_result = await manager.run_pipeline(labs={"glucose": 130.0, "hemoglobin": 9.2})
    symptoms_result = await manager.run_pipeline(manual_input={"symptoms": "fatigue", "labs": {"hemoglobin": 9.2}})
    return {
        "labs_status": labs_result["status"],
        "labs_findings": len(labs_result["diagnosis"]["findings"]),
        "symptoms_status": symptoms_result["status"],
        "symptoms_findings": len(symptoms_result["diagnosis"]["findings"]),
        "rag_present": "rag_response" in labs_result["diagnosis"],
        "classifier_present": "classifier_prediction" in labs_result["diagnosis"],
        "therapy_mode": labs_result["therapy"].get("metadata", {}).get("mode"),
    }


def main() -> None:
    settings = get_settings()
    faiss_dir = Path(settings.faiss_index_dir or "")
    clinicalbert_dir = Path(settings.clinicalbert_model_dir or "")
    model_dir = Path(settings.finetuned_model_dir or "")

    report: dict[str, Any] = {
        "config": {
            "use_rag": settings.use_rag,
            "faiss_index_dir": settings.faiss_index_dir,
            "clinicalbert_model_dir": settings.clinicalbert_model_dir,
            "use_finetuned_classifier": settings.use_finetuned_classifier,
            "finetuned_model_dir": settings.finetuned_model_dir,
            "gemini_key_present": bool(settings.gemini_api_key),
        },
        "assets": {
            "faiss_index_exists": (faiss_dir / "medical_cases.index").exists() if settings.faiss_index_dir else False,
            "faiss_metadata_exists": faiss_metadata_exists(faiss_dir) if settings.faiss_index_dir else False,
            "clinicalbert_model_dir_exists": clinicalbert_dir.is_dir() if settings.clinicalbert_model_dir else False,
            "clinicalbert_missing_files": (
                find_missing_clinicalbert_files(clinicalbert_dir)
                if clinicalbert_dir.is_dir()
                else ["model directory missing"]
            ) if settings.use_rag else [],
            "finetuned_model_dir_exists": model_dir.is_dir() if settings.finetuned_model_dir else False,
            "finetuned_missing_files": find_missing_files(model_dir) if model_dir.is_dir() else ["model directory missing"],
        },
    }
    report["assets"].update(
        inspect_faiss_metadata_quality(
            faiss_dir=faiss_dir,
            enabled=bool(settings.use_rag and settings.faiss_index_dir and faiss_dir.is_dir()),
        )
    )

    can_run_rag = report["assets"]["faiss_index_exists"] and report["assets"]["faiss_metadata_exists"]
    if settings.use_rag:
        can_run_rag = (
            can_run_rag
            and report["assets"]["clinicalbert_model_dir_exists"]
            and not report["assets"]["clinicalbert_missing_files"]
        )
    can_run_classifier = (
        report["assets"]["finetuned_model_dir_exists"]
        and not report["assets"]["finetuned_missing_files"]
    )
    can_run_manager = True
    blockers: list[str] = []
    if settings.use_rag and not can_run_rag:
        can_run_manager = False
        blockers.append("RAG is enabled but FAISS assets or local ClinicalBERT files are missing.")
    if settings.use_finetuned_classifier and not can_run_classifier:
        can_run_manager = False
        blockers.append("Fine-tuned classifier is enabled but model files are missing.")

    try:
        if can_run_manager:
            manager = ChatManager(
                use_rag=settings.use_rag,
                faiss_index_dir=settings.faiss_index_dir,
                clinicalbert_model_dir=settings.clinicalbert_model_dir,
                gemini_api_key=settings.gemini_api_key,
                rag_top_k=settings.rag_top_k,
                rag_translate_arabic=settings.rag_translate_arabic,
                use_finetuned_classifier=settings.use_finetuned_classifier,
                finetuned_model_dir=settings.finetuned_model_dir,
                classifier_max_length=settings.classifier_max_length,
                classifier_translate_arabic=settings.classifier_translate_arabic,
            )
            report["smoke"] = asyncio.run(run_smoke_tests(manager))
            report["status"] = "ok"
        else:
            report["status"] = "blocked"
            report["blockers"] = blockers
    except Exception as exc:
        report["status"] = "error"
        report["error"] = str(exc)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
