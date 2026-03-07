"""
Regex patterns and synonym mapping for lab-value extraction.

Each entry in LAB_PATTERNS maps a canonical key to a compiled regex that
captures an optional numeric value and an optional unit from OCR text.
SYNONYM_MAP normalises the many common spellings/abbreviations found on
lab report printouts to the same canonical key.
"""

import re
from typing import Dict, Pattern

# ---------------------------------------------------------------------------
# Synonym → canonical-key mapping
# ---------------------------------------------------------------------------
SYNONYM_MAP: Dict[str, str] = {
    # Glucose
    "glucose": "glucose",
    "gluc": "glucose",
    "glu": "glucose",
    "blood sugar": "glucose",
    "blood glucose": "glucose",
    # Hemoglobin
    "hemoglobin": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    # Iron
    "iron": "iron",
    "fe": "iron",
    "serum iron": "iron",
    # White blood cells
    "wbc": "wbc",
    "white blood cells": "wbc",
    "white blood count": "wbc",
    "leukocytes": "wbc",
    # Red blood cells
    "rbc": "rbc",
    "red blood cells": "rbc",
    "red blood count": "rbc",
    "erythrocytes": "rbc",
    # Platelets
    "platelets": "platelets",
    "plt": "platelets",
    "thrombocytes": "platelets",
    # Hematocrit
    "hematocrit": "hematocrit",
    "haematocrit": "hematocrit",
    "hct": "hematocrit",
    "pcv": "hematocrit",
    # Cholesterol
    "cholesterol": "cholesterol",
    "chol": "cholesterol",
    "total cholesterol": "cholesterol",
    # Creatinine
    "creatinine": "creatinine",
    "creat": "creatinine",
    "cr": "creatinine",
    # Urea / BUN
    "urea": "urea",
    "bun": "urea",
    "blood urea nitrogen": "urea",
    # Sodium
    "sodium": "sodium",
    "na": "sodium",
    # Potassium
    "potassium": "potassium",
    "k": "potassium",
    # Calcium
    "calcium": "calcium",
    "ca": "calcium",
}

# ---------------------------------------------------------------------------
# Per-canonical-key capture patterns
# ---------------------------------------------------------------------------
# Pattern structure (built dynamically below):
#   synonyms  [:\-\s]+  value  \s*  unit?
# Handles OCR artifacts such as extra whitespace, colons and dashes between
# the label and the numeric value.

_SEP = r"[\s:\-]+\s*"           # separator between label and value
_VALUE = r"(\d+(?:[.,]\d+)?)"   # integer or decimal (handles comma decimals)
_UNIT_FRAG = r"([a-zA-Z/%µμ]+(?:/[a-zA-Z]+)?)"  # e.g. mg/dL, g/L, mmol/L

# Build one pattern per canonical key that matches *any* synonym for that key.
def _build_pattern(synonyms: list[str]) -> Pattern[str]:
    alts = "|".join(re.escape(s) for s in sorted(synonyms, key=len, reverse=True))
    return re.compile(
        rf"(?i)(?:{alts}){_SEP}{_VALUE}\s*{_UNIT_FRAG}?",
        re.IGNORECASE,
    )


def _group_synonyms() -> Dict[str, list[str]]:
    grouped: Dict[str, list[str]] = {}
    for alias, canonical in SYNONYM_MAP.items():
        grouped.setdefault(canonical, []).append(alias)
    return grouped


LAB_PATTERNS: Dict[str, Pattern[str]] = {
    canonical: _build_pattern(aliases)
    for canonical, aliases in _group_synonyms().items()
}
