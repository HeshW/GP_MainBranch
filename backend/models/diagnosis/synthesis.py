from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from app.schemas.ai import AIClinicalResponse
from models.common.ai_provider import GeminiProvider
from models.common.language import detect_preferred_language, normalize_language

logger = logging.getLogger(__name__)


class DiagnosisResponseSynthesizer:
    """Generate a final patient-facing medical response after diagnosis fusion."""

    def __init__(self, gemini_api_key: str, model_name: str = "gemini-2.5-flash-lite") -> None:
        self.api_key_valid = bool(str(gemini_api_key or "").strip())
        self._provider: Optional[GeminiProvider] = None

        if self.api_key_valid:
            self._provider = GeminiProvider(api_key=gemini_api_key, model_name=model_name)
        else:
            logger.warning(
                "Missing GEMINI_API_KEY. DiagnosisResponseSynthesizer will operate in fallback mode."
            )

    @staticmethod
    def _provider_status_from_exception(exc: Exception) -> str:
        message = str(exc).lower()
        if any(token in message for token in ("401", "unauthorized", "invalid api key", "permission denied")):
            return "provider_unauthorized"
        if any(token in message for token in ("429", "rate limit", "quota", "resource exhausted")):
            return "provider_rate_limited"
        if any(token in message for token in ("timeout", "timed out")):
            return "provider_timeout"
        return "provider_unavailable"

    @staticmethod
    def _fallback_payload(
        diagnosis: Dict[str, Any],
        *,
        provider_status: str,
        response_language: str = "en",
    ) -> Dict[str, Any]:
        language = normalize_language(response_language)
        arabic_mode = language == "ar"
        final_diagnosis = diagnosis.get("final_diagnosis") or {}
        label = str(final_diagnosis.get("diagnosis", "an undetermined condition")).strip()
        confidence = final_diagnosis.get("confidence", "unknown")
        source = final_diagnosis.get("source", "fusion")
        summary = diagnosis.get("summary") or (
            f"يشير التقييم الأولي إلى {label}."
            if arabic_mode
            else f"Preliminary assessment suggests {label}."
        )
        likely_label = "التشخيص الأرجح" if arabic_mode else "Most likely diagnosis"
        unavailable_line = (
            "السرد التوضيحي المدعوم بالذكاء الاصطناعي غير متاح حالياً. يرجى مراجعة النتائج المنظمة."
            if arabic_mode
            else "AI narrative synthesis is currently unavailable. Please review the structured findings."
        )
        return {
            "response_text": (
                f"{summary}\n\n"
                f"{likely_label}: {label} (confidence {confidence}, source: {source}).\n"
                f"{unavailable_line}"
            ),
            "structured_response": None,
            "metadata": {
                "mode": "fallback",
                "final_diagnosis": label,
                "provider_status": provider_status,
                "response_language": language,
            },
        }

    @staticmethod
    def _build_prompt(
        report: Dict[str, Any],
        diagnosis: Dict[str, Any],
        *,
        response_language: str = "en",
    ) -> str:
        language = normalize_language(response_language)
        language_name = "Arabic" if language == "ar" else "English"
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
            "If the evidence is conflicting, say so clearly and recommend clinician review.\n"
            f"Target response language: {language_name}.\n"
            "All patient-facing content in your JSON fields must be in that language.\n\n"
            "[FUSED DIAGNOSTIC CONTEXT]\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    async def synthesize(
        self,
        report: Dict[str, Any],
        diagnosis: Dict[str, Any],
        *,
        response_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_language = normalize_language(
            response_language or str(diagnosis.get("response_language", "")).strip(),
            default=detect_preferred_language(
                report.get("raw_text"),
                report.get("follow_up_answers"),
                report.get("symptoms"),
                default="en",
            ),
        )
        arabic_mode = selected_language == "ar"

        if not self.api_key_valid or not self._provider:
            return self._fallback_payload(
                diagnosis,
                provider_status="missing_api_key",
                response_language=selected_language,
            )

        prompt = self._build_prompt(
            report,
            diagnosis,
            response_language=selected_language,
        )
        system_instruction = (
            "You are a professional medical AI assistant. "
            "Return valid JSON matching the requested schema. "
            "Be conservative, explain uncertainty, and emphasize clinician review. "
            + ("Respond strictly in Arabic." if arabic_mode else "Respond strictly in English.")
        )

        try:
            response_json = await self._provider.generate_content(
                prompt,
                system_instruction=system_instruction,
                response_model=AIClinicalResponse,
            )
            structured = json.loads(response_json)
            next_steps_header = "الخطوات التالية الموصى بها:" if arabic_mode else "Recommended next steps:"
            red_flags_header = "علامات الخطر:" if arabic_mode else "Red flags:"
            response_text = (
                f"{structured['diagnosis_summary']}\n\n"
                f"{structured['patient_friendly_explanation']}\n\n"
                f"{next_steps_header}\n"
                + "\n".join(f"- {item}" for item in structured["recommended_next_steps"])
                + f"\n\n{red_flags_header}\n"
                + "\n".join(f"- {item}" for item in structured["red_flags"])
                + f"\n\n{structured['disclaimer']}"
            )
            return {
                "response_text": response_text,
                "structured_response": structured,
                "metadata": {
                    "mode": "llm",
                    "final_diagnosis": (diagnosis.get("final_diagnosis") or {}).get("diagnosis"),
                    "response_language": selected_language,
                },
            }
        except Exception as exc:
            logger.error("Diagnosis response synthesis failed: %s", exc)
            return self._fallback_payload(
                diagnosis,
                provider_status=self._provider_status_from_exception(exc),
                response_language=selected_language,
            )
