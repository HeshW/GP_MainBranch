from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from app.schemas.ai import AITherapyPlanResponse
from models.common.provider_factory import create_model_provider

logger = logging.getLogger(__name__)

NO_FINDINGS_MESSAGE = (
    "No abnormal diagnostic findings were detected that require an urgent therapy plan. "
    "Routine clinical follow-up is still recommended."
)

FALLBACK_MESSAGE = (
    "Fallback mode: AI-generated therapy guidance is unavailable right now. "
    "Please review the diagnostic findings with a qualified clinician before acting on them."
)


class TherapyEngine:
    """Generate a structured therapy plan from diagnosis findings."""

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
                "Missing LLM_API_KEY for provider '%s'. TherapyEngine will operate in fallback mode.",
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
    def _format_findings(findings: list[Dict[str, Any]]) -> str:
        lines: list[str] = []
        for finding in findings:
            lines.append(
                f"- {finding.get('condition', 'Unknown finding')} "
                f"(severity: {finding.get('severity', 'unknown')}, "
                f"confidence: {finding.get('confidence', 'unknown')})"
            )
            evidence = finding.get("evidence")
            if evidence:
                lines.append(f"  Evidence: {evidence}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_payload(
        findings: list[Dict[str, Any]],
        patient_info: str,
        *,
        provider_status: str,
    ) -> Dict[str, Any]:
        return {
            "therapy_plan": FALLBACK_MESSAGE,
            "structured_therapy": None,
            "metadata": {
                "mode": "fallback",
                "findings_count": len(findings),
                "patient_info": patient_info or "unknown",
                "provider_status": provider_status,
            },
        }

    async def generate_therapy(
        self,
        diagnosis: Dict[str, Any],
        patient_info: str = "",
    ) -> Dict[str, Any]:
        """Generate therapy guidance using Gemini when available."""
        findings = diagnosis.get("findings", [])
        if not findings:
            return {
                "therapy_plan": NO_FINDINGS_MESSAGE,
                "structured_therapy": None,
                "metadata": {
                    "mode": "no_findings",
                    "findings_count": 0,
                    "patient_info": patient_info or "unknown",
                },
            }

        if not self.api_key_valid or not self._provider:
            return self._fallback_payload(
                findings,
                patient_info,
                provider_status="missing_api_key",
            )

        findings_text = self._format_findings(findings)
        safety = diagnosis.get("safety", {})

        prompt = (
            "You are an expert medical consultant.\n"
            "Generate a conservative therapy and follow-up plan in Arabic based on the "
            "preliminary findings below.\n\n"
            "[Diagnostic findings]\n"
            f"{findings_text}\n\n"
            "[Safety flags]\n"
            f"{json.dumps(safety, ensure_ascii=False)}\n\n"
            "[Patient info]\n"
            f"{patient_info or 'Unknown'}"
        )

        system_instruction = (
            "You are a helpful and professional Medical Consultant AI. "
            "You must provide medical recommendations in Arabic. "
            "Be conservative, emphasize clinician review, and return valid JSON "
            "matching the requested schema."
        )

        try:
            response_json = await self._provider.generate_content(
                prompt,
                system_instruction=system_instruction,
                response_model=AITherapyPlanResponse,
            )
            structured_data = json.loads(response_json)

            recommendations = "\n".join(
                [
                    f"- **{item['category']}**: {item['description']} ({item['urgency']})"
                    for item in structured_data["recommendations"]
                ]
            )
            emergency = "\n".join(f"- {item}" for item in structured_data["emergency_signs"])

            therapy_markdown = (
                f"{structured_data['disclaimer']}\n\n"
                f"### Clinical Analysis\n{structured_data['clinical_analysis']}\n\n"
                f"### Recommendations\n{recommendations}\n\n"
                f"### Lifestyle Advice\n{structured_data['lifestyle_advice']}\n\n"
                f"### Emergency Signs\n{emergency}"
            )

            return {
                "therapy_plan": therapy_markdown,
                "structured_therapy": structured_data,
                "metadata": {
                    "mode": "llm",
                    "findings_count": len(findings),
                    "patient_info": patient_info or "unknown",
                },
            }
        except Exception as exc:
            logger.error("Therapy generation failed: %s", exc)
            return self._fallback_payload(
                findings,
                patient_info,
                provider_status=self._provider_status_from_exception(exc),
            )
