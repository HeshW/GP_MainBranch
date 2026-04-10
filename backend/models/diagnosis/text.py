from __future__ import annotations

import re
from typing import Any, Dict, List


def build_combined_text(report: Dict[str, Any]) -> str:
    parts: List[str] = []
    sex_age = (report.get("fields") or {}).get("sex_age", "")
    if sex_age:
        parts.append(f"Patient: {sex_age}.")

    raw_text = str(report.get("raw_text") or "").strip()
    if raw_text:
        parts.append(f"Clinical text: {raw_text[:400]}")

    symptom_list = report.get("symptoms") or []
    if symptom_list:
        normalized_symptoms = [str(item).strip() for item in symptom_list if str(item).strip()]
        if normalized_symptoms:
            parts.append("Symptoms: " + ", ".join(normalized_symptoms[:25]) + ".")

    labs = report.get("labs") or {}
    if labs:
        lab_parts = []
        for key, entry in labs.items():
            if isinstance(entry, dict):
                value = entry.get("value", "?")
                unit = (entry.get("unit") or "").strip()
                lab_parts.append(f"{key}={value} {unit}".rstrip())
            else:
                lab_parts.append(f"{key}={entry}")
        parts.append("Labs: " + ", ".join(lab_parts) + ".")

    sections = report.get("sections") or {}
    for section_name in ("Clinical", "Diagnosis", "Microscopic"):
        text = (sections.get(section_name) or "").strip()
        if text:
            parts.append(f"{section_name}: {text[:300]}")

    combined = " ".join(parts)
    return combined[:512] if combined else (report.get("raw_text") or "")[:512]


class EvidenceMapper:
    """Maps evidence or symptom codes to human-readable text."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def get_text(self, code: str) -> str:
        if code in self._cache:
            return self._cache[code]
        cleaned = str(code).replace("_", " ").replace("-", " ")
        cleaned = re.sub(r"^[Ee]\s+", "", cleaned)
        readable = " ".join(word.capitalize() for word in cleaned.split())
        result = readable if readable else str(code)
        self._cache[code] = result
        return result
