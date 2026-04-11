"""Symptom parser for free-text clinical input."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SYMPTOM_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("fatigue", ("fatigue", "tired", "tiredness", "exhaustion", "تعب", "إرهاق", "إجهاد")),
    ("weakness", ("weakness", "weak", "fatigable weakness", "ضعف", "وهن")),
    ("dizziness", ("dizziness", "dizzy", "vertigo")),
    ("lightheaded", ("lightheaded", "lightheadedness")),
    ("nausea", ("nausea", "nauseated", "غثيان")),
    ("vomiting", ("vomiting", "vomit", "vomiting", "قيء", "ترجيع")),
    ("fever", ("fever", "febrile", "high temperature", "حمى", "سخونية", "حرارة")),
    ("cough", ("cough", "coughing", "كحة", "سعال")),
    ("productive cough", ("productive cough",)),
    ("dry cough", ("dry cough",)),
    ("shortness of breath", ("shortness of breath", "dyspnea", "breathlessness", "ضيق تنفس", "نهجان", "صعوبة في التنفس")),
    ("wheezing", ("wheezing", "wheeze", "صفير", "أزيز")),
    ("chest pain", ("chest pain", "retrosternal pain", "pleuritic chest pain", "ألم صدر", "وجع صدر")),
    ("chest tightness", ("chest tightness", "tight chest")),
    ("palpitations", ("palpitations", "rapid heartbeat", "irregular heartbeat", "tachycardia", "خفقان", "رفرفة", "ضربات قلب سريعة")),
    ("irregular heartbeat", ("irregular heartbeat", "irregular pulse")),
    ("sore throat", ("sore throat", "throat pain", "ألم الحلق", "التهاب الحلق")),
    ("hoarseness", ("hoarseness", "hoarse voice", "voice hoarseness", "بحة", "بحة صوت")),
    ("headache", ("headache", "head pain", "صداع")),
    ("abdominal pain", ("abdominal pain", "abdomen pain", "epigastric pain", "stomach pain", "ألم بطن", "ألم معدة")),
    ("loss of appetite", ("loss of appetite", "poor appetite", "reduced appetite")),
    ("weight loss", ("weight loss", "unexplained weight loss", "فقدان وزن", "نقصان وزن")),
    ("nasal congestion", ("nasal congestion", "stuffy nose", "احتقان الأنف", "انسداد الأنف")),
    ("facial pressure", ("facial pressure", "sinus pressure")),
    ("ear pain", ("ear pain", "otalgia")),
    ("flushing", ("flushing", "flushed", "احمرار")),
    ("tingling", ("tingling", "pins and needles", "paresthesia")),
    ("gait difficulty", ("gait difficulty", "difficulty walking", "trouble walking")),
    ("ptosis", ("ptosis", "drooping eyelid", "droopy eyelid", "تدلي الجفن", "ارتخاء الجفن")),
    ("difficulty speaking", ("difficulty speaking", "slurred speech", "speech difficulty", "صعوبة الكلام", "تلعثم")),
    ("difficulty swallowing", ("difficulty swallowing", "trouble swallowing", "hard to swallow", "صعوبة البلع")),
    ("double vision", ("double vision", "diplopia")),
    ("reflux", ("reflux", "acid reflux", "heartburn", "sour taste", "ارتجاع", "حموضة", "حرقان")),
    ("thirst", ("thirst", "excessive thirst", "عطش")),
    ("polyuria", ("polyuria", "frequent urination", "كثرة التبول", "تبول متكرر")),
    ("blurred vision", ("blurred vision", "blurry vision", "زغللة", "تشوش الرؤية")),
    ("leg swelling", ("leg swelling", "swollen leg", "leg edema", "تورم الساق", "ورم الساق")),
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
            alias_text = str(alias).lower()
            has_arabic = any("\u0600" <= char <= "\u06ff" for char in alias_text)
            pattern = re.compile(rf"\b{re.escape(alias_text)}\b", re.IGNORECASE)
            if (has_arabic and alias_text in text) or pattern.search(text):
                matched_alias = alias
                break
        if matched_alias:
            found.append({"symptom": canonical, "source": matched_alias, "confidence": 0.85})

    negation_cues = ("no", "denies", "deny", "without", "not having", "negative for", "لا يوجد", "بدون", "ينفي", "مش")
    for canonical, aliases in SYMPTOM_PATTERNS:
        phrases = (canonical, *aliases)
        for phrase in phrases:
            escaped = re.escape(phrase)
            negated = any(
                re.search(rf"\b{cue}\b[^.!,;]{{0,20}}\b{escaped}\b", text, re.IGNORECASE)
                for cue in negation_cues
            )
            if negated:
                found.append(
                    {
                        "symptom": canonical,
                        "source": phrase,
                        "confidence": 0.6,
                        "negated": True,
                    }
                )
                break

    return found


def _extract_context(raw_text: str) -> Dict[str, List[str]]:
    patterns = {
        "duration": [
            r"\bfor\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b",
            r"\bsince\s+(?:yesterday|today|last night|last week|last month)\b",
            r"منذ\s+\w+",
            r"لمدة\s+\d+\s+\w+",
        ],
        "onset": [
            r"\bsudden(?:ly)?\b",
            r"\bgradual(?:ly)?\b",
            r"\bstarted suddenly\b",
            r"فجأة",
            r"تدريجي",
        ],
        "triggers": [
            r"\bafter meals?\b",
            r"\bwith exertion\b",
            r"\bon exertion\b",
            r"\bwhen lying down\b",
            r"\bworse with breathing\b",
            r"بعد الأكل",
            r"مع المجهود",
            r"مع التنفس",
            r"عند الاستلقاء",
        ],
        "severity": [
            r"\bmild\b",
            r"\bmoderate\b",
            r"\bsevere\b",
            r"\bprogressive(?:ly)?\b",
            r"\bworsening\b",
            r"شديد",
            r"متزايد",
            r"يزداد",
        ],
    }
    lowered = raw_text.lower()
    extracted: Dict[str, List[str]] = {}
    for key, regexes in patterns.items():
        matches: List[str] = []
        for regex in regexes:
            for item in re.findall(regex, lowered, flags=re.IGNORECASE):
                value = item if isinstance(item, str) else " ".join(part for part in item if part)
                value = str(value).strip()
                if value and value not in matches:
                    matches.append(value)
        if matches:
            extracted[key] = matches
    return extracted


def parse_symptoms(raw_text: str) -> Dict[str, Any]:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    labs = _extract_labs(raw_text)
    symptoms = _extract_symptoms(raw_text)

    return {
        "raw_text": raw_text,
        "labs": labs,
        "symptoms": symptoms,
        "context": _extract_context(raw_text),
    }
