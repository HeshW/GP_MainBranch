"""backend/app/schemas/ai.py

Pydantic models for structured AI outputs (Diagnosis & Therapy).
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class AIDiagnosisFinding(BaseModel):
    """A single medical finding from AI analysis."""
    condition: str = Field(..., description="The name of the suspected condition or finding.")
    confidence: str = Field(..., description="Level of confidence: High, Moderate, or Low.")
    evidence: str = Field(..., description="Clinical evidence or rationale for this finding.")
    severity: str = Field(..., description="Severity level: Critical, High, Moderate, Low, or Info.")


class AIDiagnosisResponse(BaseModel):
    """The full structured response for AI diagnosis."""
    assessment_summary: str = Field(..., description="A professional, narrative summary of the medical assessment.")
    findings: List[AIDiagnosisFinding] = Field(..., description="List of specific clinical findings identified.")
    follow_up_questions: List[str] = Field(..., description="3 critical questions to ask the patient to further clarify the diagnosis.")
    source_attribution: Optional[str] = Field(None, description="Note on data sources or similar cases matched.")


class AITherapyRecommendation(BaseModel):
    """Specific therapy/treatment recommendation."""
    category: str = Field(..., description="Category: Medications, Lifestyle, Further Testing, etc.")
    description: str = Field(..., description="Detailed medical recommendation.")
    urgency: str = Field(..., description="Urgency: Immediate, Routine, or Optional.")


class AITherapyPlanResponse(BaseModel):
    """The full structured response for AI therapy plan."""
    clinical_analysis: str = Field(..., description="AI evaluation of the diagnosis in the context of the therapy plan.")
    recommendations: List[AITherapyRecommendation] = Field(..., description="List of specific medical recommendations.")
    lifestyle_advice: str = Field(..., description="Practical advice on nutrition, exercise, or habits.")
    emergency_signs: List[str] = Field(..., description="Specific red flags that require immediate emergency care.")
    disclaimer: str = Field(..., description="Mandatory medical safety disclaimer.")


class AIClinicalResponse(BaseModel):
    """Structured final medical response synthesized after fusion."""

    diagnosis_summary: str = Field(..., description="Short explanation of the most likely diagnosis and why.")
    patient_friendly_explanation: str = Field(..., description="Clear explanation suitable for the patient.")
    recommended_next_steps: List[str] = Field(..., description="Conservative next clinical steps or follow-up actions.")
    red_flags: List[str] = Field(..., description="Urgent red-flag symptoms that require immediate care.")
    disclaimer: str = Field(..., description="Mandatory medical safety disclaimer.")
