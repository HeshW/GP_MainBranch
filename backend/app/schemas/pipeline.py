"""Pydantic models for API request/response bodies."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "gp-medical-api"


class MetaResponse(BaseModel):
    api_version: str
    project: str = "GP Medical Report Analysis"
    rag_enabled: bool
    faiss_configured: bool
    finetuned_classifier_enabled: bool = False
    finetuned_model_configured: bool = False


class LabsPipelineRequest(BaseModel):
    """Manual lab values (same shape as OCR ``labs`` — numbers or ``{value, unit}``)."""

    labs: Dict[str, Any] = Field(
        default_factory=dict,
        examples=[{"glucose": 145.0, "hemoglobin": 11.2}],
    )
    symptoms: Optional[str] = Field(
        None,
        description="Optional free-text symptoms merged into the report.",
    )


class SymptomsPipelineRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Symptoms in Arabic or English.")
    use_symptom_parser: bool = Field(
        True,
        description="If true, run symptom parser/validator before diagnosis.",
    )
    low_confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
