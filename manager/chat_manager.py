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

logger = logging.getLogger(__name__)


class ChatManager:
    """Orchestrate OCR + Diagnosis for the GP project."""

    def __init__(
        self,
        *,
        use_rag: bool = False,
        faiss_index_dir: Optional[Path | str] = None,
        gemini_api_key: Optional[str] = None,
        rag_top_k: int = 5,
    ) -> None:
        self._diagnosis_engine = DiagnosisEngine(
            use_rag=use_rag,
            faiss_index_dir=faiss_index_dir,
            gemini_api_key=gemini_api_key,
            rag_top_k=rag_top_k,
        )

    def run_ocr(self, image: Union[str, Path, Any]) -> Dict[str, Any]:
        """Run OCR on an image and return structured output."""
        from models.ocr.engine import OCREngine  # lazy import

        engine = OCREngine()
        ocr_result = engine.extract(image)
        return ocr_result

    def _build_report(
        self,
        image: Optional[Union[str, Path, Any]] = None,
        labs: Optional[Dict[str, Any]] = None,
        manual_input: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Combine inputs into a diagnosis report structure."""
        report: Dict[str, Any] = {}
        warnings: List[str] = []

        if image is not None:
            try:
                ocr_data = self.run_ocr(image)
                report.update(ocr_data or {})
            except Exception as exc:
                warnings.append(f"OCR failed: {exc}")
                logger.warning("OCR failed: %s", exc, exc_info=True)

        if labs is not None:
            if not isinstance(labs, dict):
                raise TypeError("labs must be a dict mapping lab keys to values")
            report["labs"] = labs

        # Ensure labs exists for diagnosis engine even if empty.
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

    def run_diagnosis(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Run diagnosis on a prepared report."""
        if not isinstance(report, dict):
            raise TypeError("report must be a dict")
        return self._diagnosis_engine.diagnose(report)

    def run_pipeline(
        self,
        *,
        image: Optional[Union[str, Path, Any]] = None,
        labs: Optional[Dict[str, Any]] = None,
        manual_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run end-to-end pipeline and return a unified response."""
        start = time.time()

        report, warnings = self._build_report(
            image=image,
            labs=labs,
            manual_input=manual_input,
        )

        diagnosis = self.run_diagnosis(report)

        response = {
            "status": "ok",
            "id": None,
            "ocr": report if image is not None else None,
            "diagnosis": diagnosis,
            "therapy": None,
            "warnings": warnings,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
        }

        return response

    def run_from_symptoms(
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

        pipeline_result = self.run_pipeline(manual_input=manual_input)

        pipeline_result["parsed"] = parsed
        pipeline_result["validated"] = validated
        pipeline_result["review_required"] = validated.get("review_required", False)

        return pipeline_result

    def run_chat(self, session_id: str, message: str) -> Dict[str, Any]:
        """Simplified chat hook for prototype.

        This method currently returns the last diagnosis summary for a session
        (a real conversation manager can be added later).
        """
        return {
            "session_id": session_id,
            "message": message,
            "response": "Chat mode is in prototype state: run_pipeline for diagnosis.",
        }
