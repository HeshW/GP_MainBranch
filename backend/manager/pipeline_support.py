from __future__ import annotations

import logging
from inspect import isawaitable
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

ImageInput = Optional[Union[str, Path, Any]]
OcrRunner = Callable[[Union[str, Path, Any]], Union[Awaitable[Dict[str, Any]], Dict[str, Any]]]


async def build_report(
    *,
    run_ocr: OcrRunner,
    image: ImageInput = None,
    labs: Optional[Dict[str, Any]] = None,
    manual_input: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    report: Dict[str, Any] = {}
    warnings: List[str] = []

    if image is not None:
        try:
            ocr_result = run_ocr(image)
            ocr_data = await ocr_result if isawaitable(ocr_result) else ocr_result
            report.update(ocr_data or {})
        except Exception as exc:
            warnings.append(f"OCR failed: {exc}")
            logger.warning("OCR failed: %s", exc, exc_info=True)

    if labs is not None:
        if not isinstance(labs, dict):
            raise TypeError("labs must be a dict mapping lab keys to values")
        report["labs"] = labs

    report.setdefault("labs", {})

    if manual_input:
        if not isinstance(manual_input, dict):
            raise TypeError("manual_input must be a dict")

        raw_text = manual_input.get("raw_text")
        if raw_text:
            report["raw_text"] = str(raw_text).strip()

        symptoms = manual_input.get("symptoms")
        if symptoms:
            if report.get("raw_text"):
                report["raw_text"] = f"{report.get('raw_text', '')} {symptoms}".strip()
            else:
                report["raw_text"] = str(symptoms).strip()

        symptom_list = manual_input.get("symptom_list")
        if symptom_list:
            if not isinstance(symptom_list, list):
                raise TypeError("manual_input['symptom_list'] must be a list")
            report["symptoms"] = [str(item).strip().lower() for item in symptom_list if str(item).strip()]

        manual_labs = manual_input.get("labs")
        if manual_labs:
            if not isinstance(manual_labs, dict):
                raise TypeError("manual_input['labs'] must be a dict")
            report["labs"] = {**report.get("labs", {}), **manual_labs}

    return report, warnings


def build_manual_input_from_validated(validated: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symptoms": " ".join(validated.get("symptoms", [])),
        "symptom_list": list(validated.get("symptoms", [])),
        "raw_text": validated.get("raw_text", ""),
        "labs": {
            key: value["value"]
            for key, value in validated.get("labs", {}).items()
        },
    }
