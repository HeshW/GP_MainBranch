"""Symptom parser for free-text clinical input."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SYMPTOM_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("fatigue", ("fatigue", "tired", "tiredness", "exhaustion")),
    ("weakness", ("weakness", "weak", "fatigable weakness")),
    ("dizziness", ("dizziness", "dizzy", "vertigo")),
    ("lightheaded", ("lightheaded", "lightheadedness")),
    ("nausea", ("nausea", "nauseated")),
    ("vomiting", ("vomiting", "vomit", "vomiting")),
    ("fever", ("fever", "febrile", "high temperature")),
    ("cough", ("cough", "coughing")),
    ("productive cough", ("productive cough",)),
    ("dry cough", ("dry cough",)),
    ("shortness of breath", ("shortness of breath", "dyspnea", "breathlessness")),
    ("wheezing", ("wheezing", "wheeze")),
    ("chest pain", ("chest pain", "retrosternal pain", "pleuritic chest pain")),
    ("chest tightness", ("chest tightness", "tight chest")),
    ("palpitations", ("palpitations", "rapid heartbeat", "irregular heartbeat", "tachycardia")),
    ("sore throat", ("sore throat", "throat pain")),
    ("hoarseness", ("hoarseness", "hoarse voice", "voice hoarseness")),
    ("headache", ("headache", "head pain")),
    ("abdominal pain", ("abdominal pain", "abdomen pain", "epigastric pain", "stomach pain")),
    ("loss of appetite", ("loss of appetite", "poor appetite", "reduced appetite")),
    ("weight loss", ("weight loss", "unexplained weight loss")),
    ("nasal congestion", ("nasal congestion", "stuffy nose")),
    ("facial pressure", ("facial pressure", "sinus pressure")),
    ("ear pain", ("ear pain", "otalgia")),
    ("flushing", ("flushing", "flushed")),
    ("tingling", ("tingling", "pins and needles", "paresthesia")),
    ("gait difficulty", ("gait difficulty", "difficulty walking", "trouble walking")),
    ("ptosis", ("ptosis", "drooping eyelid", "droopy eyelid")),
    ("difficulty speaking", ("difficulty speaking", "slurred speech", "speech difficulty")),
    ("reflux", ("reflux", "acid reflux", "heartburn", "sour taste")),
    ("thirst", ("thirst", "excessive thirst")),
    ("polyuria", ("polyuria", "frequent urination")),
]

UNIT_PATTERNS = [
    "mg/dl",
    "g/dl",
    "mmol/l",
    "mEq/l",
    "%",
    "ug/dl",
    "times10^3/ul",
    "x10^3/ul",
    "x10^6/ul",
]

ALIASES_PATH = Path(__file__).resolve().parents[1] / "models" / "ocr" / "synonyms_v15.json"


def _load_aliases() -> Dict[str, str]:
    if not ALIASES_PATH.exists():
        return {}
    with open(ALIASES_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {k.lower(): v.lower() for k, v in data.get("aliases", {}).items()}


_lab_aliases = None


def _get_lab_aliases() -> Dict[str, str]:
    global _lab_aliases
    if _lab_aliases is None:
        _lab_aliases = _load_aliases()
    return _lab_aliases


def _extract_labs(raw_text: str) -> Dict[str, Dict[str, Any]]:
    text = raw_text.lower()
    aliases = _get_lab_aliases()

    labs: Dict[str, Dict[str, Any]] = {}

    # 1) explicit "name value unit" patterns
    base_pattern = re.compile(
        r"([a-z0-9\s]+?)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*([a-z%/\^0-9\u00B5\u03BC]*)",
        re.IGNORECASE,
    )

    for match in base_pattern.finditer(raw_text):
        name_raw = match.group(1).strip().lower()
        value_raw = match.group(2)
        unit_raw = match.group(3).strip().lower()

        # Attempt to match lab name from aliases
        key_candidates = []
        for alias, canonical in aliases.items():
            if alias in name_raw:
                key_candidates.append((len(alias), canonical))
        if key_candidates:
            canonical = sorted(key_candidates, key=lambda x: -x[0])[0][1]
        else:
            # too generic or phrase, skip
            continue

        if canonical in labs:
            continue

        try:
            value = float(value_raw)
        except ValueError:
            continue

        clean_unit = unit_raw.strip().lower() or None
        if clean_unit and clean_unit not in UNIT_PATTERNS:
            # Keep whatever parsed, but normalize case
            clean_unit = clean_unit

        labs[canonical] = {
            "value": value,
            "unit": clean_unit,
            "source": match.group(0).strip(),
            "confidence": 0.92,
        }

    # 2) simple lab name + number tokens for remaining known labs
    for alias, canonical in aliases.items():
        if canonical in labs:
            continue
        pattern = re.compile(rf"\b{re.escape(alias)}\b\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
        m = pattern.search(raw_text)
        if m:
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            labs[canonical] = {
                "value": value,
                "unit": None,
                "source": m.group(0).strip(),
                "confidence": 0.7,
            }

    return labs


def _extract_symptoms(raw_text: str) -> List[Dict[str, Any]]:
    text = raw_text.lower()
    found: List[Dict[str, Any]] = []

    for canonical, aliases in SYMPTOM_PATTERNS:
        matched_alias = None
        for alias in aliases:
            pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            if pattern.search(text):
                matched_alias = alias
                break
        if matched_alias:
            found.append({"symptom": canonical, "source": matched_alias, "confidence": 0.85})

    # detect negated symptoms
    negation_pattern = re.compile(r"\b(no|denies?|without)\s+([a-z\s]+?)\b(?:[.,;]|$)", re.IGNORECASE)
    for m in negation_pattern.finditer(text):
        symptom_text = m.group(2).strip()
        found.append({"symptom": symptom_text, "source": m.group(0).strip(), "confidence": 0.5, "negated": True})

    return found


def parse_symptoms(raw_text: str) -> Dict[str, Any]:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    labs = _extract_labs(raw_text)
    symptoms = _extract_symptoms(raw_text)

    return {
        "raw_text": raw_text,
        "labs": labs,
        "symptoms": symptoms,
    }
