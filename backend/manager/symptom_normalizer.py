from __future__ import annotations

import re
from typing import Any, Dict, List


_DURATION_PATTERNS = [
    re.compile(r"\bfor\s+\d+\s+(day|days|week|weeks|month|months|year|years)\b", re.IGNORECASE),
    re.compile(r"\bfor\s+\w+\s+(day|days|week|weeks|month|months|year|years)\b", re.IGNORECASE),
    re.compile(r"\bsince\s+(yesterday|today|last night|last week|last month)\b", re.IGNORECASE),
]

_CONTEXT_PATTERNS = [
    re.compile(r"\bafter meals?\b", re.IGNORECASE),
    re.compile(r"\bwith exertion\b", re.IGNORECASE),
    re.compile(r"\bat rest\b", re.IGNORECASE),
    re.compile(r"\bworsening\b", re.IGNORECASE),
    re.compile(r"\bprogressive(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\bsudden(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\bgradual(?:ly)?\b", re.IGNORECASE),
]

_DDX_STYLE_TEMPLATES = {
    "fatigue": "Have you noticed any new fatigue, generalized discomfort, or reduced well-being related to your consultation today?",
    "weakness": "Do you feel weakness in your body or limbs?",
    "dizziness": "Do you feel dizzy or lightheaded?",
    "lightheaded": "Do you feel lightheaded or close to fainting?",
    "nausea": "Do you feel nauseated?",
    "vomiting": "Have you vomited?",
    "fever": "Do you have a fever (either felt or measured with a thermometer)?",
    "cough": "Do you have a cough?",
    "productive cough": "Are you coughing up mucus or phlegm?",
    "dry cough": "Do you have a dry cough?",
    "shortness of breath": "Are you experiencing shortness of breath or difficulty breathing in a significant way?",
    "wheezing": "Have you noticed a wheezing sound when you exhale?",
    "chest pain": "Do you have pain somewhere in your chest related to your reason for consulting?",
    "chest tightness": "Do you feel chest tightness or pressure?",
    "palpitations": "Do you feel your heart is beating fast, irregularly, or do you feel palpitations?",
    "sore throat": "Do you have throat pain or a sore throat?",
    "hoarseness": "Have you noticed hoarseness or a change in your voice?",
    "headache": "Do you have a headache?",
    "abdominal pain": "Do you have abdominal or epigastric pain?",
    "loss of appetite": "Have you had a reduced appetite or loss of appetite?",
    "weight loss": "Have you experienced weight loss?",
    "nasal congestion": "Do you have nasal congestion or a runny nose?",
    "ear pain": "Do you have ear pain?",
    "flushing": "Have you noticed flushing or becoming red in the face?",
    "ptosis": "Do you have a hard time opening or raising one or both eyelids?",
    "difficulty speaking": "Do you have difficulty articulating words or speaking?",
    "reflux": "Have you ever been diagnosed with gastroesophageal reflux or do you have reflux symptoms?",
    "thirst": "Do you feel more thirsty than usual?",
    "polyuria": "Have you been urinating more often than usual?",
}


def _extract_matches(raw_text: str, patterns: List[re.Pattern[str]]) -> List[str]:
    matches: List[str] = []
    for pattern in patterns:
        for match in pattern.finditer(raw_text):
            value = match.group(0).strip()
            if value and value not in matches:
                matches.append(value)
    return matches


def _symptom_phrases(parsed_symptoms: List[Dict[str, Any]]) -> List[str]:
    phrases: List[str] = []
    for item in parsed_symptoms:
        if not isinstance(item, dict):
            continue
        if item.get("negated"):
            continue
        source = str(item.get("source") or "").strip().lower()
        canonical = str(item.get("symptom") or "").strip().lower()
        phrase = source or canonical
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def build_normalized_symptom_text(parsed: Dict[str, Any], validated: Dict[str, Any]) -> str:
    raw_text = str(parsed.get("raw_text", "") or "").strip()
    validated_symptoms = [
        str(item).strip().lower()
        for item in (validated.get("symptoms", []) or [])
        if str(item).strip()
    ]
    parsed_symptoms = parsed.get("symptoms", []) or []
    symptom_phrases = _symptom_phrases(parsed_symptoms)
    duration_mentions = _extract_matches(raw_text, _DURATION_PATTERNS)
    context_mentions = _extract_matches(raw_text, _CONTEXT_PATTERNS)

    parts: List[str] = []
    if raw_text:
        parts.append(f"Patient-reported complaint: {raw_text}")

    if symptom_phrases:
        parts.append("Patient reports: " + ", ".join(symptom_phrases[:8]))

    if validated_symptoms:
        parts.append("Normalized symptoms: " + ", ".join(validated_symptoms[:12]))

    ddx_style_cues = [
        _DDX_STYLE_TEMPLATES[item]
        for item in validated_symptoms
        if item in _DDX_STYLE_TEMPLATES
    ]
    if ddx_style_cues:
        parts.append("DDX-style cues: " + " ".join(dict.fromkeys(ddx_style_cues)))

    if duration_mentions:
        parts.append("Duration: " + ", ".join(duration_mentions))

    if context_mentions:
        parts.append("Context: " + ", ".join(context_mentions))

    return ". ".join(parts).strip() or raw_text
