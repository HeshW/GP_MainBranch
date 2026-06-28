"""Runtime health check for the local FAISS-backed medical RAG bundle.

The command intentionally avoids LLM calls and never prints secrets. It loads
the same configured FAISS/ClinicalBERT assets used by the backend, then runs one
in-scope retrieval and one out-of-scope safety retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from models.diagnosis.rag import ClinicalBERTEmbedder, FineTunedDiagnosisClassifier, MedicalCaseSearcher, MedicalRAGAssistant
from models.diagnosis.text import build_combined_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check RAG artifact health without printing secrets.")
    parser.add_argument("--faiss-index-dir", type=Path, default=None)
    parser.add_argument("--clinicalbert-model-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def _case_report(raw_text: str, symptoms: list[str], age: int, sex: str, labs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "raw_text": raw_text,
        "symptoms": symptoms,
        "labs": labs or {},
        "fields": {"sex_age": f"{age} year old {sex}"},
    }


def _run_retrieval_probe(
    *,
    searcher: MedicalCaseSearcher,
    embedder: ClinicalBERTEmbedder,
    raw_text: str,
    symptoms: list[str],
    age: int,
    sex: str,
    top_k: int,
    labs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_text = build_combined_text(_case_report(raw_text, symptoms, age, sex, labs))
    embedding = embedder.encode_text(query_text)
    retrieved = searcher.search(
        embedding,
        k=top_k,
        query_text=query_text,
        query_symptoms=symptoms,
    )
    detected_out_of_scope = MedicalCaseSearcher.detect_out_of_scope_signals(query_text, symptoms)
    confidence = MedicalRAGAssistant._build_confidence_metadata(
        retrieved,
        detected_out_of_scope_signals=detected_out_of_scope,
    )
    return {
        "query_text_preview": query_text[:240],
        "top_pathology": str(retrieved[0].get("pathology", "")) if retrieved else None,
        "top_rerank_score": retrieved[0].get("rerank_score") if retrieved else None,
        "retrieved_pathologies": [str(item.get("pathology", "")) for item in retrieved],
        "rag_confidence": confidence,
    }


def build_health_payload(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    faiss_dir = args.faiss_index_dir or Path(settings.faiss_index_dir or "")
    model_dir = args.clinicalbert_model_dir or Path(settings.clinicalbert_model_dir or "")
    if not faiss_dir:
        raise ValueError("FAISS index directory is not configured.")
    if not model_dir:
        raise ValueError("ClinicalBERT model directory is not configured.")

    searcher = MedicalCaseSearcher(
        faiss_dir,
        allow_unsafe_pickle=bool(settings.allow_unsafe_pickle_metadata),
    )
    pathologies = [str(item).strip() for item in searcher.metadata.get("pathologies", []) if str(item).strip()]
    metadata_rows = len(searcher.metadata.get("patient_ids", []) or [])
    embedder = ClinicalBERTEmbedder(model_dir=model_dir)

    classifier_status: dict[str, Any] = {
        "enabled": bool(settings.use_finetuned_classifier),
        "loaded": False,
    }
    if settings.use_finetuned_classifier:
        try:
            FineTunedDiagnosisClassifier(
                model_dir=Path(settings.finetuned_model_dir or ""),
                max_length=settings.classifier_max_length,
            )
            classifier_status["loaded"] = True
        except Exception as exc:
            classifier_status["error"] = f"{type(exc).__name__}: {exc}"

    in_scope_probe = _run_retrieval_probe(
        searcher=searcher,
        embedder=embedder,
        raw_text="Exertional chest pressure radiating to the left arm that improves with rest.",
        symptoms=["chest pain", "chest pressure", "exertion"],
        age=62,
        sex="M",
        top_k=args.top_k,
    )
    out_of_scope_probe = _run_retrieval_probe(
        searcher=searcher,
        embedder=embedder,
        raw_text="Fatigue, increased thirst, frequent urination, weight loss, and fasting glucose is elevated.",
        symptoms=["fatigue", "thirst", "polyuria", "weight loss"],
        labs={"glucose": {"value": 230, "unit": "mg/dL"}},
        age=50,
        sex="F",
        top_k=args.top_k,
    )

    return {
        "status": "ok",
        "faiss_index_dir": str(faiss_dir),
        "clinicalbert_model_dir": str(model_dir),
        "index_loaded": True,
        "metadata_loaded": True,
        "faiss_vector_count": int(searcher.index.ntotal),
        "metadata_row_count": metadata_rows,
        "unique_pathologies": len(set(pathologies)),
        "top_pathology_counts": Counter(pathologies).most_common(10),
        "faiss_index_type": type(searcher.index).__name__,
        "faiss_nprobe": getattr(searcher.index, "nprobe", None),
        "clinicalbert_model_loaded": True,
        "clinicalbert_model_source": str(getattr(embedder, "model_source", model_dir)),
        "classifier": classifier_status,
        "in_scope_smoke_query": in_scope_probe,
        "out_of_scope_safety_query": out_of_scope_probe,
    }


def main() -> None:
    args = parse_args()
    payload = build_health_payload(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
