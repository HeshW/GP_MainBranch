"""Manager orchestrator for OCR + Diagnosis pipelines.

This module is the core of PR#1. It provides a simple orchestrator interface
for:
 - image input (OCR -> Diagnosis)
 - manual labs input (Diagnosis only)
 - a placeholder manual symptom path

Later phases can add therapy and chat state.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from models.diagnosis import DiagnosisEngine
from models.therapy import TherapyEngine

logger = logging.getLogger(__name__)


class ChatManager:
    """Orchestrate OCR + Diagnosis for the GP project using async processing."""

    def __init__(
        self,
        *,
        use_rag: bool = False,
        faiss_index_dir: Optional[Path | str] = None,
        gemini_api_key: Optional[str] = None,
        rag_top_k: int = 5,
        rag_translate_arabic: bool = True,
    ) -> None:
        self._diagnosis_engine = DiagnosisEngine(
            use_rag=use_rag,
            faiss_index_dir=faiss_index_dir,
            gemini_api_key=gemini_api_key,
            rag_top_k=rag_top_k,
            rag_translate_arabic=rag_translate_arabic,
        )
        self._therapy_engine = TherapyEngine(
            gemini_api_key=gemini_api_key if gemini_api_key else ""
        )
        
        # In-memory chat sessions
        self._chat_sessions: Dict[str, List[Dict[str, str]]] = {}

    async def run_ocr(self, image: Union[str, Path, Any]) -> Dict[str, Any]:
        """Run OCR on an image (async wrapper for potential future async OCR)."""
        from models.ocr.engine import OCREngine  # lazy import
        # For now, OCR is blocking, so we run in a thread if needed, 
        # but the engine itself is relatively fast.
        engine = OCREngine()
        return engine.extract(image)

    async def _build_report(
        self,
        image: Optional[Union[str, Path, Any]] = None,
        labs: Optional[Dict[str, Any]] = None,
        manual_input: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Combine inputs into a diagnosis report structure asynchronously."""
        report: Dict[str, Any] = {}
        warnings: List[str] = []

        if image is not None:
            try:
                ocr_data = await self.run_ocr(image)
                report.update(ocr_data or {})
            except Exception as exc:
                warnings.append(f"OCR failed: {exc}")
                logger.warning("OCR failed: %s", exc, exc_info=True)

        if labs is not None:
            if not isinstance(labs, dict):
                raise TypeError("labs must be a dict mapping lab keys to values")
            report["labs"] = labs

        # Ensure labs exists
        report.setdefault("labs", {})

        if manual_input:
            if not isinstance(manual_input, dict):
                raise TypeError("manual_input must be a dict")

            symptoms = manual_input.get("symptoms")
            if symptoms:
                report["raw_text"] = report.get("raw_text", "") + " " + str(symptoms)
            manual_labs = manual_input.get("labs")
            if manual_labs:
                if not isinstance(manual_labs, dict):
                    raise TypeError("manual_input['labs'] must be a dict")
                report["labs"] = {**report.get("labs", {}), **manual_labs}

        return report, warnings

    async def run_diagnosis(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Run diagnosis on a prepared report."""
        return await self._diagnosis_engine.diagnose(report)

    async def run_pipeline(
        self,
        *,
        image: Optional[Union[str, Path, Any]] = None,
        labs: Optional[Dict[str, Any]] = None,
        manual_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run end-to-end pipeline asynchronously."""
        start = time.time()

        report, warnings = await self._build_report(
            image=image,
            labs=labs,
            manual_input=manual_input,
        )

        diagnosis = await self.run_diagnosis(report)
        
        # Generate Therapy Plan based on Diagnosis
        patient_info = "Age: Unknown, Sex: Unknown" 
        therapy_result = await self._therapy_engine.generate_therapy(diagnosis, patient_info)

        return {
            "status": "ok",
            "id": None,
            "ocr": report if image is not None else None,
            "diagnosis": diagnosis,
            "therapy": therapy_result,
            "warnings": warnings,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
        }

    async def run_from_symptoms(
        self,
        text: str,
        *,
        low_confidence_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Convert free-text symptoms into structured report + run diagnosis."""
        from manager.symptom_parser import parse_symptoms
        from manager.symptom_validator import validate_parsed

        parsed = parse_symptoms(text)
        validated = validate_parsed(parsed, low_confidence_threshold=low_confidence_threshold)

        manual_input = {
            "symptoms": " ".join(validated.get("symptoms", [])),
            "labs": {k: v["value"] for k, v in validated.get("labs", {}).items()},
        }

        pipeline_result = await self.run_pipeline(manual_input=manual_input)
        pipeline_result.update({
            "parsed": parsed,
            "validated": validated,
            "review_required": validated.get("review_required", False)
        })

        return pipeline_result

    async def run_chat(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle chat conversation with memory synchronously using AIProvider."""
        if not self._therapy_engine.api_key_valid or not self._therapy_engine._provider:
            return {
                "session_id": session_id,
                "message": message,
                "response": "عذراً، نظام المحادثة غير متاح حالياً. يرجى التحقق من إعدادات API."
            }
            
        if session_id not in self._chat_sessions:
            self._chat_sessions[session_id] = []
            
        history = self._chat_sessions[session_id]
        history.append({"role": "user", "content": message})
        
        system_instruction = (
            "أنت استشاري طبي ذكي ولطيف. قدم إجابات دقيقة ومهنية للمرضى. "
            "أكد دائماً على ضرورة استشارة الطبيب المختص. "
            "إذا سُئلت عن مطورك، أجب بفخر: 'مطوري هو Mr.Bondo2'."
        )
        
        # Build prompt from history
        context = ""
        for msg in history[-8:]: # Last 8 messages for context
            role_tag = "المريض" if msg["role"] == "user" else "الطبيب"
            context += f"{role_tag}: {msg['content']}\n"
            
        prompt = f"سياق المحادثة:\n{context}\nرد الطبيب باللغة العربية:"

        try:
            reply_text = await self._therapy_engine._provider.generate_content(
                prompt, 
                system_instruction=system_instruction
            )
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            reply_text = "عذراً، حدث خطأ تقني في معالجة طلبك. يرجى المحاولة لاحقاً."
            
        history.append({"role": "model", "content": reply_text})
        
        # Keep clean memory
        if len(history) > 12:
            self._chat_sessions[session_id] = history[-10:]

        return {
            "session_id": session_id,
            "message": message,
            "response": reply_text,
        }

    async def stream_chat(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        """Stream a chat response chunk by chunk."""
        if not self._therapy_engine.api_key_valid or not self._therapy_engine._provider:
            yield "عذراً، نظام المحادثة غير متاح حالياً."
            return

        if session_id not in self._chat_sessions:
            self._chat_sessions[session_id] = []
            
        history = self._chat_sessions[session_id]
        history.append({"role": "user", "content": message})
        
        system_instruction = (
            "أنت استشاري طبي ذكي ولطيف. قدم إجابات دقيقة ومهنية للمرضى. "
            "أكد دائماً على ضرورة استشارة الطبيب المختص. "
            "إذا سُئلت عن مطورك، أجب بفخر: 'مطوري هو Mr.Bondo2'."
        )
        
        context = ""
        for msg in history[-8:]: 
            role_tag = "المريض" if msg["role"] == "user" else "الطبيب"
            context += f"{role_tag}: {msg['content']}\n"
            
        prompt = f"سياق المحادثة:\n{context}\nرد الطبيب باللغة العربية:"

        full_response = []
        try:
            async for chunk in self._therapy_engine._provider.generate_stream(
                prompt, 
                system_instruction=system_instruction
            ):
                full_response.append(chunk)
                yield chunk
        except Exception as e:
            logger.error(f"Stream Chat failed: {e}")
            yield "❌ حدث خطأ أثناء بث الرد."
            
        history.append({"role": "model", "content": "".join(full_response)})
        if len(history) > 12:
            self._chat_sessions[session_id] = history[-10:]
