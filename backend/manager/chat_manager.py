"""Manager orchestrator for OCR + diagnosis, therapy, and chat flows."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from manager.chat_support import (
    SYSTEM_INSTRUCTION,
    build_chat_prompt,
    build_unavailable_payload,
    get_chat_error_message,
    get_stream_error_message,
)
from manager.pipeline_support import build_manual_input_from_validated, build_report
from manager.session_store import ChatSessionStore
from models.diagnosis import DiagnosisEngine
from models.therapy import TherapyEngine

logger = logging.getLogger(__name__)


class ChatManager:
    """Public facade for the project's diagnosis pipeline and chat tools."""

    def __init__(
        self,
        *,
        use_rag: bool = False,
        faiss_index_dir: Optional[Path | str] = None,
        clinicalbert_model_dir: Optional[Path | str] = None,
        allow_unsafe_pickle_metadata: bool = False,
        gemini_api_key: Optional[str] = None,
        gemini_model_name: str = "gemini-2.5-flash-lite",
        rag_top_k: int = 5,
        rag_translate_arabic: bool = True,
        use_finetuned_classifier: bool = False,
        finetuned_model_dir: Optional[Path | str] = None,
        classifier_max_length: int = 256,
        classifier_translate_arabic: bool = True,
    ) -> None:
        self._diagnosis_engine = DiagnosisEngine(
            use_rag=use_rag,
            faiss_index_dir=faiss_index_dir,
            clinicalbert_model_dir=clinicalbert_model_dir,
            allow_unsafe_pickle_metadata=allow_unsafe_pickle_metadata,
            gemini_api_key=gemini_api_key,
            gemini_model_name=gemini_model_name,
            rag_top_k=rag_top_k,
            rag_translate_arabic=rag_translate_arabic,
            use_finetuned_classifier=use_finetuned_classifier,
            finetuned_model_dir=finetuned_model_dir,
            classifier_max_length=classifier_max_length,
            classifier_translate_arabic=classifier_translate_arabic,
        )
        self._therapy_engine = TherapyEngine(
            gemini_api_key=gemini_api_key if gemini_api_key else "",
            model_name=gemini_model_name,
        )
        self._chat_sessions = ChatSessionStore()
        self._ocr_engine: Any | None = None

    def _get_ocr_engine(self):
        if self._ocr_engine is None:
            from models.ocr.engine import OCREngine

            self._ocr_engine = OCREngine()
        return self._ocr_engine

    async def run_ocr(self, image: Union[str, Path, Any]) -> Dict[str, Any]:
        """Run OCR on an image."""
        return self._get_ocr_engine().extract(image)

    async def prepare_report(
        self,
        *,
        image: Optional[Union[str, Path, Any]] = None,
        labs: Optional[Dict[str, Any]] = None,
        manual_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a normalized report payload without running diagnosis."""
        report, warnings = await build_report(
            run_ocr=self.run_ocr,
            image=image,
            labs=labs,
            manual_input=manual_input,
        )
        return {
            "status": "ok",
            "report": report,
            "warnings": warnings,
        }

    async def run_ocr_only(self, image: Union[str, Path, Any]) -> Dict[str, Any]:
        """Run OCR only and return the extracted report fields."""
        start = time.time()
        ocr = await self.run_ocr(image)
        return {
            "status": "ok",
            "ocr": ocr,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
        }

    async def run_diagnosis(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Run diagnosis on a prepared report."""
        return await self._diagnosis_engine.diagnose(report)

    async def run_diagnosis_only(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Run diagnosis and therapy on an already prepared report."""
        start = time.time()
        diagnosis = await self.run_diagnosis(report)
        therapy = await self._therapy_engine.generate_therapy(
            diagnosis,
            "Age: Unknown, Sex: Unknown",
        )
        return {
            "status": "ok",
            "report": report,
            "diagnosis": diagnosis,
            "therapy": therapy,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
        }

    async def run_pipeline(
        self,
        *,
        image: Optional[Union[str, Path, Any]] = None,
        labs: Optional[Dict[str, Any]] = None,
        manual_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the end-to-end OCR/diagnosis/therapy pipeline."""
        start = time.time()
        prepared = await self.prepare_report(image=image, labs=labs, manual_input=manual_input)
        report = prepared["report"]
        warnings = prepared["warnings"]
        diagnosis_result = await self.run_diagnosis_only(report)

        return {
            "status": "ok",
            "id": None,
            "ocr": report if image is not None else None,
            "report": report,
            "diagnosis": diagnosis_result["diagnosis"],
            "therapy": diagnosis_result["therapy"],
            "warnings": warnings,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
        }

    async def run_from_symptoms(
        self,
        text: str,
        *,
        low_confidence_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Parse free-text symptoms into structured diagnosis input."""
        from manager.symptom_parser import parse_symptoms
        from manager.symptom_normalizer import build_normalized_symptom_text
        from manager.symptom_validator import validate_parsed

        parsed = parse_symptoms(text)
        validated = validate_parsed(
            parsed,
            low_confidence_threshold=low_confidence_threshold,
        )
        normalized_text = build_normalized_symptom_text(parsed, validated)
        validated["raw_text"] = normalized_text
        manual_input = build_manual_input_from_validated(validated)

        pipeline_result = await self.run_pipeline(manual_input=manual_input)
        pipeline_result.update(
            {
                "parsed": parsed,
                "validated": validated,
                "normalized_text": normalized_text,
                "review_required": validated.get("review_required", False),
            }
        )
        return pipeline_result

    async def run_clarification(
        self,
        report: Dict[str, Any],
        answers: list[str],
        *,
        prior_diagnosis: Optional[Dict[str, Any]] = None,
        low_confidence_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Re-run diagnosis after incorporating follow-up answers into the report."""
        from manager.pipeline_support import merge_follow_up_into_report
        from manager.symptom_parser import parse_symptoms
        from manager.symptom_normalizer import build_normalized_symptom_text
        from manager.symptom_validator import validate_parsed

        if not isinstance(report, dict):
            raise TypeError("report must be a dictionary")
        if not isinstance(answers, list):
            raise TypeError("answers must be a list of strings")

        joined_answers = " ".join(str(item).strip() for item in answers if str(item).strip()).strip()
        parsed = parse_symptoms(joined_answers)
        validated = validate_parsed(
            parsed,
            low_confidence_threshold=low_confidence_threshold,
        )
        normalized_text = build_normalized_symptom_text(parsed, validated)
        updated_report = merge_follow_up_into_report(
            report,
            normalized_follow_up_text=normalized_text,
            follow_up_symptoms=list(validated.get("symptoms", [])),
            follow_up_labs={
                key: value["value"]
                for key, value in validated.get("labs", {}).items()
            },
            raw_follow_up_answers=answers,
        )

        diagnosis_result = await self.run_diagnosis_only(updated_report)
        diagnosis_result["diagnosis"] = self._diagnosis_engine.apply_follow_up_scoring(
            diagnosis_result["diagnosis"],
            answers=answers,
            prior_diagnosis=prior_diagnosis,
        )
        diagnosis_result.update(
            {
                "follow_up": {
                    "answers": [str(item).strip() for item in answers if str(item).strip()],
                    "parsed": parsed,
                    "validated": validated,
                    "normalized_text": normalized_text,
                    "updated_report": updated_report,
                }
            }
        )
        return diagnosis_result

    async def run_chat(self, session_id: str, message: str) -> Dict[str, Any]:
        """Generate a non-streaming chat reply with lightweight session memory."""
        if not self._therapy_engine.api_key_valid or not self._therapy_engine._provider:
            return build_unavailable_payload(session_id, message)

        history = self._chat_sessions.append(session_id, "user", message)
        prompt = build_chat_prompt(history)

        try:
            reply_text = await self._therapy_engine._provider.generate_content(
                prompt,
                system_instruction=SYSTEM_INSTRUCTION,
            )
        except Exception as exc:
            logger.error("Chat failed: %s", exc)
            reply_text = get_chat_error_message(message)

        self._chat_sessions.append(session_id, "model", reply_text)
        return {
            "session_id": session_id,
            "message": message,
            "response": reply_text,
        }

    async def stream_chat(
        self,
        session_id: str,
        message: str,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat response chunk by chunk."""
        history = self._chat_sessions.append(session_id, "user", message)

        if not self._therapy_engine.api_key_valid or not self._therapy_engine._provider:
            unavailable_response = build_unavailable_payload(session_id, message)["response"]
            yield unavailable_response
            self._chat_sessions.append(session_id, "model", unavailable_response)
            return

        prompt = build_chat_prompt(history)

        full_response: list[str] = []
        try:
            async for chunk in self._therapy_engine._provider.generate_stream(
                prompt,
                system_instruction=SYSTEM_INSTRUCTION,
            ):
                full_response.append(chunk)
                yield chunk
        except Exception as exc:
            logger.error("Stream chat failed: %s", exc)
            error_chunk = get_stream_error_message(message)
            full_response.append(error_chunk)
            yield error_chunk

        self._chat_sessions.append(session_id, "model", "".join(full_response))
