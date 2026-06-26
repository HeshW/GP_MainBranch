from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    condition: str
    confidence: str
    evidence: str
    severity: str


@dataclass
class Rule:
    lab: str
    condition: str
    operator: str
    evidence_fmt: str
    confidence: str = "moderate"
    severity: str = "moderate"
    limit: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def matches(self, value: float) -> bool:
        if self.operator == "lt":
            return value < self.limit
        if self.operator == "le":
            return value <= self.limit
        if self.operator == "gt":
            return value > self.limit
        if self.operator == "ge":
            return value >= self.limit
        if self.operator == "eq":
            return value == self.limit
        if self.operator == "range":
            return self.min_val is not None and self.max_val is not None and self.min_val <= value < self.max_val
        return False


def load_clinical_rules() -> List[Rule]:
    rules_path = Path(__file__).parent / "clinical_rules.yaml"
    if not rules_path.exists():
        logger.warning("clinical_rules.yaml not found at %s. No threshold rules loaded.", rules_path)
        return []
    try:
        with rules_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        parsed = [
            Rule(
                lab=rule["lab"],
                condition=rule["condition"],
                operator=rule["operator"],
                evidence_fmt=rule["evidence_fmt"],
                confidence=rule.get("confidence", "moderate"),
                severity=rule.get("severity", "moderate"),
                limit=rule.get("limit"),
                min_val=rule.get("min"),
                max_val=rule.get("max"),
            )
            for rule in data.get("rules", [])
        ]
        logger.info("Loaded %d clinical rules from YAML", len(parsed))
        return parsed
    except Exception as exc:
        logger.error("Failed to load clinical rules: %s", exc)
        return []


RULES: List[Rule] = load_clinical_rules()


def diagnose_from_labs(labs: Dict[str, Any]) -> List[Finding]:
    findings: List[Finding] = []
    seen: set[tuple[str, str]] = set()

    for rule in RULES:
        entry = labs.get(rule.lab)
        if entry is None:
            continue
        try:
            value = float(entry.get("value", entry) if isinstance(entry, dict) else entry)
        except (TypeError, ValueError):
            logger.debug("Cannot parse lab value for %r: %r", rule.lab, entry)
            continue
        if not rule.matches(value):
            continue
        key = (rule.lab, rule.condition)
        if key in seen:
            continue
        seen.add(key)
        unit = entry.get("unit", "") if isinstance(entry, dict) else ""
        findings.append(
            Finding(
                condition=rule.condition,
                confidence=rule.confidence,
                evidence=rule.evidence_fmt.format(val=value, unit=unit or "?"),
                severity=rule.severity,
            )
        )
    return findings


SYMPTOM_RULES: list[dict[str, Any]] = [
    {
        "symptoms": {"thirst", "polyuria", "fatigue"},
        "condition": "Possible hyperglycemia / diabetes symptom pattern",
        "confidence": "low",
        "severity": "moderate",
        "evidence_fmt": "Symptoms suggestive of hyperglycemia: {matched}.",
    },
    {
        "symptoms": {"fatigue", "weakness", "dizziness", "lightheaded", "shortness of breath", "palpitations"},
        "condition": "Possible anemia-related symptom pattern",
        "confidence": "low",
        "severity": "moderate",
        "evidence_fmt": "Symptoms compatible with anemia: {matched}.",
    },
    {
        "symptoms": {"reflux", "chest pain", "abdominal pain", "nausea"},
        "condition": "Possible gastroesophageal reflux pattern",
        "confidence": "low",
        "severity": "moderate",
        "min_matches": 1,
        "text_any": ("after meals", "heartburn", "sour taste", "retrosternal", "burning"),
        "evidence_fmt": "Symptoms and context compatible with reflux: {matched}{context_clause}.",
    },
    {
        "symptoms": {"fever", "cough", "shortness of breath"},
        "condition": "Possible lower respiratory infection pattern",
        "confidence": "low",
        "severity": "high",
        "min_matches": 2,
        "text_any": ("fever", "productive cough", "pleuritic", "infection", "chills", "sputum"),
        "evidence_fmt": "Respiratory symptom cluster detected: {matched}.",
    },
    {
        "symptoms": {"sore throat", "nasal congestion", "cough", "hoarseness", "fever"},
        "condition": "Possible upper respiratory tract infection pattern",
        "confidence": "low",
        "severity": "moderate",
        "min_matches": 2,
        "text_any": ("sore throat", "nasal congestion", "runny nose", "hoarseness", "cold", "upper respiratory"),
        "evidence_fmt": "Upper-respiratory symptom cluster detected: {matched}{context_clause}.",
    },
    {
        "symptoms": {"chest pain", "shortness of breath"},
        "condition": "Possible cardiopulmonary red-flag symptom pattern",
        "confidence": "low",
        "severity": "high",
        "evidence_fmt": "Red-flag symptoms detected: {matched}.",
    },
    {
        "symptoms": {"fever", "headache", "fatigue"},
        "condition": "Possible acute viral illness pattern",
        "confidence": "low",
        "severity": "moderate",
        "evidence_fmt": "Systemic symptom cluster detected: {matched}.",
    },
    {
        "symptoms": {"thirst", "fatigue", "headache", "dizziness"},
        "condition": "Possible dehydration / fluid depletion pattern",
        "confidence": "low",
        "severity": "moderate",
        "min_matches": 1,
        "text_any": ("thirst", "thirsty", "dry mouth", "dry lips", "dark urine", "reduced intake"),
        "evidence_fmt": "Symptoms compatible with dehydration/fluid depletion: {matched}{context_clause}.",
    },
    {
        "symptoms": {"chest pain", "shortness of breath", "dyspnea", "palpitations", "fatigue", "viral prodrome"},
        "condition": "Myocarditis",
        "confidence": "low",
        "severity": "high",
        "min_matches": 2,
        "text_any": ("viral", "chest pain", "worse when lying down"),
        "evidence_fmt": "Symptoms compatible with myocarditis: {matched}{context_clause}.",
    },
]


def diagnose_from_symptoms(symptoms: List[str], raw_text: str = "") -> List[Finding]:
    normalized = {str(symptom).strip().lower() for symptom in symptoms if str(symptom).strip()}
    normalized_text = str(raw_text or "").strip().lower()
    findings: List[Finding] = []

    for rule in SYMPTOM_RULES:
        matched = sorted(rule["symptoms"].intersection(normalized))
        min_matches = int(rule.get("min_matches", 2))
        if len(matched) < min_matches:
            continue

        text_any = tuple(str(item).strip().lower() for item in rule.get("text_any", ()) if str(item).strip())
        if text_any and not any(token in normalized_text for token in text_any):
            continue

        context_hits = [token for token in text_any if token in normalized_text]
        context_clause = f" with context: {', '.join(context_hits)}" if context_hits else ""
        findings.append(
            Finding(
                condition=rule["condition"],
                confidence=rule["confidence"],
                evidence=rule["evidence_fmt"].format(
                    matched=", ".join(matched),
                    context_clause=context_clause,
                ),
                severity=rule["severity"],
            )
        )

    return findings
