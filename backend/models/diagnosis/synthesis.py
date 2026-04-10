from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from app.schemas.ai import AIClinicalResponse
from models.common.ai_provider import GeminiProvider

logger = logging.getLogger(__name__)


class DiagnosisResponseSynthesizer:
    """Generate a final patient-facing medical response after diagnosis fusion."""

    def __init__(self, gemini_api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        self.api_key_valid = bool(gemini_api_key and "AIza" in gemini_api_key)
        self._provider: Optional[GeminiProvider] = None

        if self.api_key_valid:
            self._provider = GeminiProvider(api_key=gemini_api_key, model_name=model_name)
        else:
            logger.warning(
                "Invalid or missing GEMINI_API_KEY. DiagnosisResponseSynthesizer will operate in fallback mode."
            )

    @staticmethod
    def _fallback_payload(diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        final_diagnosis = diagnosis.get("final_diagnosis") or {}
        label = str(final_diagnosis.get("diagnosis", "an undetermined condition")).strip()
        confidence = final_diagnosis.get("confidence", "unknown")
        source = final_diagnosis.get("source", "fusion")
        summary = diagnosis.get("summary") or f"Preliminary assessment suggests {label}."
        return {
            "response_text": (
                f"{summary}\n\n"
                f"Most likely diagnosis: {label} (confidence {confidence}, source: {source}).\n"
                "AI narrative synthesis is currently unavailable. Please review the structured findings."
            ),
            "structured_response": None,
            "metadata": {
                "mode": "fallback",
                "final_diagnosis": label,
            },
        }

    @staticmethod
    def _build_prompt(report: Dict[str, Any], diagnosis: Dict[str, Any]) -> str:
        final_diagnosis = diagnosis.get("final_diagnosis") or {}
        classifier_prediction = diagnosis.get("classifier_prediction") or {}
        retrieved_cases = diagnosis.get("retrieved_cases") or []

        retrieval_preview = [
            {
                "pathology": case.get("pathology"),
                "similarity": case.get("similarity"),
                "case_text": case.get("case_text") or case.get("symptoms"),
            }
            for case in retrieved_cases[:3]
        ]

        payload = {
            "patient_input": {
                "raw_text": report.get("raw_text", ""),
                "symptoms": report.get("symptoms", []),
                "labs": report.get("labs", {}),
            },
            "final_diagnosis": final_diagnosis,
            "rule_findings": diagnosis.get("findings", []),
            "decision_fusion": diagnosis.get("decision_fusion", {}),
            "safety": diagnosis.get("safety", {}),
            "classifier_prediction": classifier_prediction,
            "retrieved_cases_preview": retrieval_preview,
        }
        return (
            "You are a conservative clinical decision-support assistant.\n"
            "Use the fused diagnosis context below to produce a patient-facing explanation.\n"
            "Do not invent evidence. Prefer the final fused diagnosis over raw retrieved examples.\n"
            "If the evidence is conflicting, say so clearly and recommend clinician review.\n\n"
            "[FUSED DIAGNOSTIC CONTEXT]\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    async def synthesize(self, report: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key_valid or not self._provider:
            return self._fallback_payload(diagnosis)

        prompt = self._build_prompt(report, diagnosis)
        system_instruction = (
            "You are a professional medical AI assistant. "
            "Return valid JSON matching the requested schema. "
            "Be conservative, explain uncertainty, and emphasize clinician review."
        )

        try:
            response_json = await self._provider.generate_content(
                prompt,
                system_instruction=system_instruction,
                response_model=AIClinicalResponse,
            )
            structured = json.loads(response_json)
            response_text = (
                f"{structured['diagnosis_summary']}\n\n"
                f"{structured['patient_friendly_explanation']}\n\n"
                "Recommended next steps:\n"
                + "\n".join(f"- {item}" for item in structured["recommended_next_steps"])
                + "\n\nRed flags:\n"
                + "\n".join(f"- {item}" for item in structured["red_flags"])
                + f"\n\n{structured['disclaimer']}"
            )
            return {
                "response_text": response_text,
                "structured_response": structured,
                "metadata": {
                    "mode": "llm",
                    "final_diagnosis": (diagnosis.get("final_diagnosis") or {}).get("diagnosis"),
                },
            }
        except Exception as exc:
            logger.error("Diagnosis response synthesis failed: %s", exc)
            return self._fallback_payload(diagnosis)
