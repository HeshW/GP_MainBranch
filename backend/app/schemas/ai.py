"""backend/app/schemas/ai.py

Pydantic models for structured AI outputs (Diagnosis & Therapy).
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class AssessmentState(str, Enum):
    """The state of the diagnostic assessment."""
    NEEDS_CLARIFICATION = "needs_clarification"
    FINAL = "final"
    ABSTAINED = "abstained"


class DifferentialDiagnosis(BaseModel):
    """A single diagnosis within a ranked differential list."""
    condition: str = Field(..., description="The name of the condition.")
    confidence: float = Field(..., description="Numerical confidence score (0.0 to 1.0).")
    rationale: str = Field(..., description="Brief rationale for including this condition.")


class AIDiagnosisResponse(BaseModel):
    """The full structured response for AI diagnosis."""
    assessment_state: AssessmentState = Field(..., description="The overall state of the assessment, guiding the UI.")
    assessment_summary: str = Field(..., description="A professional, narrative summary of the current assessment. This should explain the leading possibilities and the rationale for clarification if needed.")
    differential_diagnosis: List[DifferentialDiagnosis] = Field(..., description="A ranked list of possible conditions.")
    follow_up_questions: List[str] = Field(default_factory=list, description="Critical questions to ask the patient to further clarify the diagnosis. Empty if assessment_state is 'final'.")
    source_attribution: Optional[str] = Field(None, description="Note on data sources or similar cases matched, if applicable.")


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
