from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _normalize_sections(sections_raw: Any) -> Dict[str, str]:
    normalized: Dict[str, str] = {}

    if isinstance(sections_raw, dict):
        for label, text in sections_raw.items():
            normalized_label = str(label).strip().lower()
            normalized_text = str(text or "").strip()
            if normalized_label and normalized_text:
                normalized[normalized_label] = normalized_text
        return normalized

    if isinstance(sections_raw, list):
        for item in sections_raw:
            if not isinstance(item, dict):
                continue

            normalized_label = str(item.get("label", "")).strip().lower()
            normalized_text = str(item.get("text", "")).strip()
            if not normalized_label or not normalized_text:
                continue

            if normalized_label in normalized and normalized[normalized_label]:
                normalized[normalized_label] = f"{normalized[normalized_label]}\n{normalized_text}".strip()
            else:
                normalized[normalized_label] = normalized_text

    return normalized


def _parse_sex_age(value: Any) -> tuple[Optional[str], Optional[str]]:
    text = str(value or "").strip()
    if not text:
        return None, None
    age_match = re.search(r"\b(\d{1,3})\s*(?:year|yr)s?\s*old\b", text, re.IGNORECASE)
    sex_match = re.search(r"\b(female|male|[fm])\b", text, re.IGNORECASE)
    age = age_match.group(1) if age_match else None
    sex = sex_match.group(1).upper()[0] if sex_match else None
    return age, sex


def _format_labs(labs: Dict[str, Any]) -> str:
    lab_parts = []
    signal_parts = []
    for key, entry in labs.items():
        if isinstance(entry, dict):
            value = entry.get("value", "?")
            unit = (entry.get("unit") or "").strip()
            lab_parts.append(f"{key}={value} {unit}".rstrip())
        else:
            value = entry
            lab_parts.append(f"{key}={entry}")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        normalized_key = str(key).strip().lower()
        if normalized_key in {"glucose", "blood_glucose", "fasting_glucose"}:
            if numeric_value >= 200:
                signal_parts.append("marked_hyperglycemia")
            elif numeric_value >= 126:
                signal_parts.append("elevated_glucose")
        if normalized_key in {"hemoglobin", "hb", "hgb"} and numeric_value < 12:
            signal_parts.append("low_hemoglobin")

    if signal_parts:
        lab_parts.append("lab_signals=" + ", ".join(dict.fromkeys(signal_parts)))
    return ", ".join(lab_parts)


def _extract_negative_symptoms(report: Dict[str, Any], raw_text: str) -> List[str]:
    explicit = [
        str(item).strip().lower()
        for item in (report.get("negated_symptoms") or [])
        if str(item).strip()
    ]
    patterns = {
        "fever": r"\b(?:no|without|denies)\s+fever\b",
        "productive cough": r"\b(?:(?:no|without|denies)\s+productive\s+cough|without\s+fever\s+or\s+productive\s+cough)\b",
        "flank pain": r"\b(?:no|without|denies)\s+flank\s+pain\b",
        "neck stiffness": r"\b(?:no|without|denies)\s+neck\s+stiffness\b",
    }
    found = []
    for label, pattern in patterns.items():
        if re.search(pattern, raw_text, re.IGNORECASE):
            found.append(label)
    return list(dict.fromkeys(explicit + found))


def build_combined_text(report: Dict[str, Any]) -> str:
    parts: List[str] = []
    sex_age = (report.get("fields") or {}).get("sex_age", "")
    age, sex = _parse_sex_age(sex_age)
    natural_parts: List[str] = []
    raw_text = str(report.get("raw_text") or "").strip()
    symptom_list = report.get("symptoms") or []
    normalized_symptoms = [str(item).strip() for item in symptom_list if str(item).strip()]
    if sex_age:
        natural_parts.append(f"Patient: {sex_age}.")
    if raw_text:
        natural_parts.append(f"Clinical text: {raw_text[:500]}")
    if normalized_symptoms:
        natural_parts.append("Symptoms: " + ", ".join(normalized_symptoms[:25]) + ".")
    if natural_parts:
        parts.append(" ".join(natural_parts))

    if age:
        parts.append(f"age: {age}")
    if sex:
        parts.append(f"sex: {sex}")

    if raw_text:
        parts.append(f"clinical_context: {raw_text[:600]}")

    if normalized_symptoms:
        if normalized_symptoms:
            parts.append("positive_symptoms: " + ", ".join(normalized_symptoms[:25]))
            parts.append("normalized_symptoms: " + ", ".join(normalized_symptoms[:25]))

    negative_symptoms = _extract_negative_symptoms(report, raw_text)
    if negative_symptoms:
        parts.append("negative_symptoms: " + ", ".join(negative_symptoms[:20]))

    labs = report.get("labs") or {}
    if labs:
        parts.append("labs: " + _format_labs(labs))

    sections = _normalize_sections(report.get("sections"))
    for section_name in ("Clinical", "Diagnosis", "Microscopic"):
        text = sections.get(section_name.lower(), "")
        if text:
            parts.append(f"{section_name}: {text[:300]}")

    combined = "\n".join(parts)
    return combined[:1600] if combined else (report.get("raw_text") or "")[:1600]


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
