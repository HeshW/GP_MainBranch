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
    (
        "shortness of breath",
        (
            "shortness of breath",
            "short of breath",
            "dyspnea",
            "breathlessness",
            "ضيق تنفس",
            "نهجان",
            "صعوبة في التنفس",
        ),
    ),
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
    (
        "ptosis",
        ("ptosis", "drooping eyelid", "droopy eyelid", "drooping eyelids", "droopy eyelids", "تدلي الجفن", "ارتخاء الجفن"),
    ),
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
_ENCODED_PRESENTING_SYMPTOMS_PATTERN = re.compile(
    r"\bPresenting\s+symptoms?\s*:\s*(?:\d+(?:\s*@\s*(?:[Vv]\s*)?\d+)?\s*,\s*){6,}\d+(?:\s*@\s*(?:[Vv]\s*)?\d+)?\.?",
    re.IGNORECASE,
)
_ENCODED_SEQUENCE_PATTERN = re.compile(
    r"(?:\d+(?:\s*@\s*(?:[Vv]\s*)?\d+)?\s*,\s*){8,}\d+(?:\s*@\s*(?:[Vv]\s*)?\d+)?",
    re.IGNORECASE,
)


def _load_aliases() -> Dict[str, str]:
    if not ALIASES_PATH.exists():
        return {}
    with open(ALIASES_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {k.lower(): v.lower() for k, v in data.get("aliases", {}).items()}


_lab_aliases = None
_symptom_alias_index: Optional[Dict[str, str]] = None


def _get_lab_aliases() -> Dict[str, str]:
    global _lab_aliases
    if _lab_aliases is None:
        _lab_aliases = _load_aliases()
    return _lab_aliases


def _get_symptom_alias_index() -> Optional[Dict[str, str]]:
    """Return cached fuzzy alias index, or None when rapidfuzz is unavailable."""
    global _symptom_alias_index
    if _symptom_alias_index is None:
        try:
            from manager.fuzzy_utils import _build_alias_index

            _symptom_alias_index = _build_alias_index(SYMPTOM_PATTERNS)
        except ImportError:
            return None
    return _symptom_alias_index


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

    # Fuzzy fallback: recover misspellings/OCR noise not captured by exact aliases.
    already_matched = {item["symptom"] for item in found}
    alias_index = _get_symptom_alias_index()
    if alias_index is not None:
        try:
            from manager.fuzzy_utils import (
                FUZZY_SYMPTOM_CONFIDENCE,
                extract_candidate_tokens,
                fuzzy_match_symptom,
            )

            for token in extract_candidate_tokens(text):
                result = fuzzy_match_symptom(token, alias_index)
                if result is None:
                    continue

                canonical_match, _matched_alias, _score = result
                if canonical_match in already_matched:
                    continue

                found.append(
                    {
                        "symptom": canonical_match,
                        "source": token,
                        "confidence": FUZZY_SYMPTOM_CONFIDENCE,
                        "fuzzy": True,
                    }
                )
                already_matched.add(canonical_match)
        except ImportError:
            pass

    negation_cues = ("no", "denies", "deny", "without", "not having", "negative for", "لا يوجد", "بدون", "ينفي", "مش")
    negation_cues = negation_cues + ("ما في", "مافي", "ما عندي", "ليس")
    for canonical, aliases in SYMPTOM_PATTERNS:
        phrases = (canonical, *aliases)
        for phrase in phrases:
            escaped = re.escape(phrase)
            negated = False
            for cue in negation_cues:
                pattern = re.compile(
                    rf"\b{cue}\b[^.!,;\n]{{0,45}}\b{escaped}\b",
                    re.IGNORECASE,
                )
                match = pattern.search(text)
                if not match:
                    continue
                scope = text[match.start():match.end()]
                if any(boundary in scope for boundary in (" لكن ", " but ", " however ", " although ")):
                    continue
                negated = True
                break
            if negated and str(phrase).strip().lower() == "cough":
                productive_only_negation = any(
                    re.search(
                        rf"\b{cue}\b[^.!,;\n]{{0,45}}\bproductive\s+cough\b",
                        text,
                        re.IGNORECASE,
                    )
                    for cue in negation_cues
                )
                explicit_plain_cough_negation = any(
                    re.search(
                        rf"\b{cue}\b[^.!,;\n]{{0,45}}\b(?<!productive\s)cough\b",
                        text,
                        re.IGNORECASE,
                    )
                    for cue in negation_cues
                )
                if productive_only_negation and not explicit_plain_cough_negation:
                    negated = False
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
            r"\bfor\s+\w+\s+(?:day|days|week|weeks|month|months|year|years)\b",
            r"\bsince\s+(?:yesterday|today|last night|last week|last month)\b",
            r"منذ\s+\w+",
            r"لمدة\s+\d+\s+\w+",
        ],
        "onset": [
            r"\bsudden(?:ly)?\b",
            r"\bgradual(?:ly)?\b",
            r"\bstarted suddenly\b",
            r"\bacute\b",
            r"فجأة",
            r"تدريجي",
        ],
        "triggers": [
            r"\bafter meals?\b",
            r"\bwith exertion\b",
            r"\bon exertion\b",
            r"\bimproves with rest\b",
            r"\bbetter with rest\b",
            r"\brelief with rest\b",
            r"\bat rest\b",
            r"\bwhen lying down\b",
            r"\bworse with breathing\b",
            r"بعد الأكل",
            r"مع المجهود",
            r"مع التنفس",
            r"يتحسن مع الراحة",
            r"يرتاح مع الراحة",
            r"يخف مع الراحة",
            r"في الراحة",
            r"أثناء الراحة",
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


def _strip_encoded_symptom_tails(raw_text: str) -> str:
    """Remove long encoded DDX symptom-id blocks that pollute natural text input."""
    if not raw_text.strip():
        return raw_text

    cleaned = _ENCODED_PRESENTING_SYMPTOMS_PATTERN.sub(" ", raw_text)
    cleaned = _ENCODED_SEQUENCE_PATTERN.sub(" ", cleaned)

    # Normalize punctuation spacing introduced by encoded-block removal.
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])(?=[^\s])", r"\1 ", cleaned)
    cleaned = re.sub(r"(?<!\d)\.(?=[^\s\d])", ". ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def parse_symptoms(raw_text: str) -> Dict[str, Any]:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    cleaned_text = _strip_encoded_symptom_tails(raw_text)

    labs = _extract_labs(cleaned_text)
    symptoms = _extract_symptoms(cleaned_text)

    parsed = {
        "raw_text": cleaned_text,
        "labs": labs,
        "symptoms": symptoms,
        "context": _extract_context(cleaned_text),
    }
    if cleaned_text != raw_text:
        parsed["raw_text_original"] = raw_text
    return parsed
