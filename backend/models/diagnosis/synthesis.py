from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from app.schemas.ai import AIClinicalResponse
from models.common.language import detect_preferred_language, normalize_language
from models.common.provider_factory import create_model_provider

logger = logging.getLogger(__name__)


class DiagnosisResponseSynthesizer:
    """Generate a final patient-facing medical response after diagnosis fusion."""

    def __init__(
        self,
        gemini_api_key: str,
        gemini_model_name: str = "gemini-2.5-flash-lite",
        *,
        llm_provider: str = "gemini",
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        openrouter_site_url: Optional[str] = None,
        openrouter_app_name: str = "GP Medical Analysis",
    ) -> None:
        self.provider_name, self._provider, self.model_name = create_model_provider(
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            gemini_api_key=gemini_api_key,
            gemini_model_name=gemini_model_name,
            openrouter_base_url=openrouter_base_url,
            openrouter_site_url=openrouter_site_url,
            openrouter_app_name=openrouter_app_name,
        )
        self.api_key_valid = self._provider is not None

        if not self.api_key_valid:
            logger.warning(
                "Missing LLM_API_KEY for provider '%s'. DiagnosisResponseSynthesizer will operate in fallback mode.",
                self.provider_name,
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
    def _as_text_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _first_text(*values: Any) -> str:
        for value in values:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
        return ""

    @classmethod
    def _normalize_structured_response(
        cls,
        parsed_payload: Any,
        diagnosis: Dict[str, Any],
        *,
        response_language: str,
    ) -> Dict[str, Any]:
        if not isinstance(parsed_payload, dict):
            raise ValueError("Synthesis provider returned a non-object JSON payload")

        language = normalize_language(response_language)
        arabic_mode = language == "ar"

        # Common OpenRouter fallback shape observed in production:
        # {"patient_explanation": {...}, "clinician_review_status": {...}}
        patient_explanation = parsed_payload.get("patient_explanation")
        if not isinstance(patient_explanation, dict):
            patient_explanation = {}

        clinician_review = parsed_payload.get("clinician_review_status")
        if not isinstance(clinician_review, dict):
            clinician_review = {}

        diagnosis_summary = cls._first_text(
            parsed_payload.get("diagnosis_summary"),
            parsed_payload.get("summary"),
            patient_explanation.get("summary"),
            diagnosis.get("summary"),
        )

        patient_explanation_text = cls._first_text(
            parsed_payload.get("patient_friendly_explanation"),
            parsed_payload.get("explanation"),
            patient_explanation.get("uncertainty"),
            patient_explanation.get("summary"),
            diagnosis.get("summary"),
        )

        recommended_next_steps = cls._as_text_list(parsed_payload.get("recommended_next_steps"))
        if not recommended_next_steps:
            recommended_next_steps = cls._as_text_list(parsed_payload.get("next_steps"))
        if not recommended_next_steps:
            recommended_next_steps = cls._as_text_list(patient_explanation.get("next_steps"))

        red_flags = cls._as_text_list(parsed_payload.get("red_flags"))
        if not red_flags:
            red_flags = cls._as_text_list(parsed_payload.get("emergency_signs"))
        if not red_flags:
            red_flags = cls._as_text_list(clinician_review.get("emergency_recommendation"))
        if not red_flags:
            safety = diagnosis.get("safety") if isinstance(diagnosis, dict) else {}
            red_flags = cls._as_text_list((safety or {}).get("reasons"))

        default_disclaimer = (
            "هذه المعلومات لغرض الدعم الإكلينيكي الأولي فقط ولا تغني عن تقييم الطبيب المختص."
            if arabic_mode
            else "This information is for preliminary clinical decision support only and does not replace clinician evaluation."
        )
        disclaimer = cls._first_text(
            parsed_payload.get("disclaimer"),
            parsed_payload.get("safety_disclaimer"),
            default_disclaimer,
        )

        if not diagnosis_summary:
            diagnosis_summary = (
                "التقييم الأولي يشير إلى ضرورة مراجعة الطبيب لتأكيد التشخيص."
                if arabic_mode
                else "Preliminary assessment indicates clinician review is needed to confirm the diagnosis."
            )
        if not patient_explanation_text:
            patient_explanation_text = (
                "توجد درجة من عدم اليقين، ويلزم تقييم سريري مباشر لتأكيد الحالة."
                if arabic_mode
                else "There is residual uncertainty, and direct clinical evaluation is required for confirmation."
            )
        if not recommended_next_steps:
            recommended_next_steps = [
                "Arrange clinician follow-up for confirmation and management."
                if not arabic_mode
                else "يُرجى ترتيب متابعة مع الطبيب لتأكيد التشخيص ووضع الخطة العلاجية."
            ]
        if not red_flags:
            red_flags = [
                "Worsening chest pain, breathing difficulty, or fainting requires urgent care."
                if not arabic_mode
                else "تفاقم ألم الصدر أو صعوبة التنفس أو الإغماء يستدعي رعاية طبية عاجلة."
            ]

        validated = AIClinicalResponse.model_validate(
            {
                "diagnosis_summary": diagnosis_summary,
                "patient_friendly_explanation": patient_explanation_text,
                "recommended_next_steps": recommended_next_steps,
                "red_flags": red_flags,
                "disclaimer": disclaimer,
            }
        )
        return validated.model_dump()

    def _fallback_payload(
        self,
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
                "provider_name": self.provider_name,
                "model_name": self.model_name,
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
            structured = self._normalize_structured_response(
                json.loads(response_json),
                diagnosis,
                response_language=selected_language,
            )
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
                    "provider_name": self.provider_name,
                    "model_name": self.model_name,
                },
            }
        except Exception as exc:
            logger.error("Diagnosis response synthesis failed: %s", exc)
            return self._fallback_payload(
                diagnosis,
                provider_status=self._provider_status_from_exception(exc),
                response_language=selected_language,
            )
