"""API routes for the separate mental-health support chatbot."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.deps import require_service_api_key
from models.mental_health import mental_llm

router = APIRouter(
    prefix="/mental-health",
    tags=["Mental Health Support"],
    dependencies=[Depends(require_service_api_key)],
)


class MentalHealthChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    language: str | None = Field(default=None, pattern="^(en|ar)$")
    max_new_tokens: int | None = Field(default=None, ge=16, le=1000)


class MentalHealthChatResponse(BaseModel):
    reply: str
    safety_status: str
    detected_language: str
    model: str = mental_llm.MODEL_ID
    disclaimer: str = mental_llm.DISCLAIMER
    model_loaded: bool = False
    latency_ms: int | None = None


@router.post("/chat", response_model=MentalHealthChatResponse)
def post_mental_health_chat(payload: MentalHealthChatRequest) -> dict[str, Any]:
    result = mental_llm.generate_mental_support_reply(
        payload.message,
        language=payload.language,
        max_new_tokens=payload.max_new_tokens,
    )
    return {
        "reply": result.get("reply", ""),
        "safety_status": result.get("safety_status", "safe"),
        "detected_language": result.get("detected_language", payload.language or "en"),
        "model": mental_llm.MODEL_ID,
        "disclaimer": mental_llm.DISCLAIMER,
        "model_loaded": bool(result.get("model_loaded", False)),
        "latency_ms": result.get("latency_ms"),
    }
