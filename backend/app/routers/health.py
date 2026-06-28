"""Health and metadata endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from app.config import get_settings
from app.schemas.pipeline import HealthResponse, MetaResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/meta", response_model=MetaResponse)
def meta(request: Request) -> MetaResponse:
    s = get_settings()
    runtime = getattr(request.app.state, "runtime_features", {})
    faiss_ok = bool(s.faiss_index_dir and Path(s.faiss_index_dir).is_dir())
    clinicalbert_ok = bool(s.clinicalbert_model_dir and Path(s.clinicalbert_model_dir).is_dir())
    finetuned_ok = bool(s.finetuned_model_dir and Path(s.finetuned_model_dir).is_dir())
    return MetaResponse(
        api_version=s.api_version,
        rag_enabled=bool(runtime.get("rag_enabled", s.use_rag and faiss_ok and clinicalbert_ok)),
        faiss_configured=faiss_ok,
        finetuned_classifier_enabled=bool(
            runtime.get("finetuned_classifier_enabled", s.use_finetuned_classifier and finetuned_ok)
        ),
        finetuned_model_configured=finetuned_ok,
        therapy_enabled=bool(runtime.get("therapy_enabled", getattr(s, "enable_therapy", False))),
    )
