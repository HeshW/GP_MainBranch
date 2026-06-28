"""Validator and normalizer for parsed symptom/lab data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from manager.symptom_parser import _get_lab_aliases


LOW_CONFIDENCE_THRESHOLD = 0.7
SYMPTOM_CANONICAL_MAP = {
    "dyspnea": "shortness of breath",
    "breathlessness": "shortness of breath",
    "irregular heartbeat": "palpitations",
    "rapid heartbeat": "palpitations",
    "heartburn": "reflux",
    "acid reflux": "reflux",
    "droopy eyelid": "ptosis",
    "drooping eyelid": "ptosis",
    "blurry vision": "blurred vision",
}


def _canonical_lab_name(name: str) -> str:
    aliases = _get_lab_aliases()
    lower = name.lower().strip()
    if lower in aliases:
        return aliases[lower]

    # Fuzzy fallback for OCR/user misspellings in lab labels.
    try:
        from manager.fuzzy_utils import fuzzy_match_lab_name

        result = fuzzy_match_lab_name(lower, aliases)
        if result is not None:
            return result[0]
    except ImportError:
        pass

    return lower


def _normalize_unit(lab_key: str, unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None

    unit = unit.strip().lower()
    unit = unit.replace("µ", "u").replace("μ", "u")
    unit = unit.replace("x10^3", "x10^3").replace("/ul", "/ul")

    replacements = {
        "mg/dl": "mg/dL",
        "g/dl": "g/dL",
        "mmol/l": "mmol/L",
        "meq/l": "mEq/L",
        "%": "%",
    }
    return replacements.get(unit, unit)


def _canonical_symptom_name(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return SYMPTOM_CANONICAL_MAP.get(normalized, normalized)


def validate_parsed(parsed: Dict[str, Any], low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        raise TypeError("parsed must be a dict")

    labs_in = parsed.get("labs", {}) or {}
    symptoms_in = parsed.get("symptoms", []) or []

    validated_labs: Dict[str, Dict[str, Any]] = {}
    confidence_map: Dict[str, float] = {}
    warnings: List[str] = []

    for raw_lab_key, lab_data in labs_in.items():
        if not isinstance(lab_data, dict):
            warnings.append(f"Invalid lab format for '{raw_lab_key}'.")
            continue

        canonical_key = _canonical_lab_name(raw_lab_key)
        if canonical_key == "":
            warnings.append(f"Empty lab key from '{raw_lab_key}'.")
            continue

        value = lab_data.get("value")
        if value is None or not isinstance(value, (int, float)):
            warnings.append(f"Lab '{raw_lab_key}' missing numeric value.")
            continue

        unit = _normalize_unit(canonical_key, lab_data.get("unit"))
        confidence = float(lab_data.get("confidence", 0.6))

        if value < 0:
            warnings.append(f"Lab '{canonical_key}' has negative value {value}.")

        validated_labs[canonical_key] = {
            "value": float(value),
            "unit": unit,
            "source": lab_data.get("source", ""),
        }
        confidence_map[canonical_key] = max(0.0, min(1.0, confidence))

    negated_symptoms: set[str] = set()
    symptom_texts: List[str] = []
    seen_positive_symptoms: set[str] = set()
    for sym in symptoms_in:
        if not isinstance(sym, dict):
            continue
        s = sym.get("symptom")
        if s and isinstance(s, str):
            normalized_symptom = _canonical_symptom_name(s)
            if sym.get("negated"):
                negated_symptoms.add(normalized_symptom)
                continue
            if normalized_symptom in seen_positive_symptoms:
                continue
            seen_positive_symptoms.add(normalized_symptom)
            symptom_texts.append(normalized_symptom)

    symptom_texts = [
        symptom
        for symptom in dict.fromkeys(symptom_texts)
        if symptom and symptom not in negated_symptoms
    ]

    review_required = False
    details: List[Dict[str, Any]] = []

    for lab_key, conf in confidence_map.items():
        if conf < low_confidence_threshold:
            review_required = True
            details.append({"lab": lab_key, "confidence": conf})

    if not symptom_texts and not validated_labs:
        review_required = True
        warnings.append("No symptoms or lab data could be validated from free-text input.")

    return {
        "labs": validated_labs,
        "symptoms": symptom_texts,
        "raw_text": str(parsed.get("raw_text", "") or ""),
        "context": parsed.get("context", {}) or {},
        "negated_symptoms": sorted(negated_symptoms),
        "confidence": confidence_map,
        "warnings": warnings,
        "review_required": review_required,
        "review_details": details,
    }
