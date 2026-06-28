"""Fuzzy-matching helpers for symptom and lab-name normalization.

This module is intentionally optional at runtime: if rapidfuzz is unavailable,
callers can skip fuzzy fallback and preserve exact/regex behavior.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from rapidfuzz import fuzz, process as rf_process

    _RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    fuzz = None  # type: ignore[assignment]
    rf_process = None  # type: ignore[assignment]
    _RAPIDFUZZ_AVAILABLE = False


SYMPTOM_FUZZY_THRESHOLD: int = 75
LAB_FUZZY_THRESHOLD: int = 80
FUZZY_SYMPTOM_CONFIDENCE: float = 0.65
MIN_TOKEN_LENGTH: int = 4


def is_available() -> bool:
    return _RAPIDFUZZ_AVAILABLE


def _is_arabic_text(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in text)


def _build_alias_index(symptom_patterns: List[Tuple[str, Tuple[str, ...]]]) -> Dict[str, str]:
    """Build a flat alias -> canonical map from symptom patterns.

    Arabic aliases are excluded from fuzzy matching to avoid noisy edit-distance
    behavior; Arabic exact matching remains in the parser.
    """
    index: Dict[str, str] = {}
    for canonical, aliases in symptom_patterns:
        for alias in aliases:
            normalized = str(alias).strip().lower()
            if normalized and not _is_arabic_text(normalized):
                index[normalized] = canonical
    return index


def fuzzy_match_symptom(
    token: str,
    alias_index: Dict[str, str],
    *,
    threshold: int = SYMPTOM_FUZZY_THRESHOLD,
) -> Optional[Tuple[str, str, float]]:
    if not _RAPIDFUZZ_AVAILABLE or len(token) < MIN_TOKEN_LENGTH:
        return None

    result = rf_process.extractOne(  # type: ignore[union-attr]
        token,
        alias_index.keys(),
        scorer=fuzz.WRatio,  # type: ignore[union-attr]
        score_cutoff=threshold,
    )
    if result is None:
        return None

    matched_alias, score, _ = result
    return alias_index[matched_alias], matched_alias, float(score)


def fuzzy_match_lab_name(
    name: str,
    aliases: Dict[str, str],
    *,
    threshold: int = LAB_FUZZY_THRESHOLD,
) -> Optional[Tuple[str, str, float]]:
    if not _RAPIDFUZZ_AVAILABLE or len(name) < MIN_TOKEN_LENGTH:
        return None

    result = rf_process.extractOne(  # type: ignore[union-attr]
        name,
        aliases.keys(),
        scorer=fuzz.WRatio,  # type: ignore[union-attr]
        score_cutoff=threshold,
    )
    if result is None:
        return None

    matched_alias, score, _ = result
    return aliases[matched_alias], matched_alias, float(score)


def extract_candidate_tokens(text: str) -> List[str]:
    """Extract de-duplicated English tokens and bigrams for fuzzy fallback."""
    lowered = str(text or "").lower()
    raw_words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", lowered)
    words = [
        w
        for w in raw_words
        if len(w) >= MIN_TOKEN_LENGTH and not w.isdigit() and not _is_arabic_text(w)
    ]

    seen: Set[str] = set()
    candidates: List[str] = []

    for word in words:
        if word not in seen:
            seen.add(word)
            candidates.append(word)

    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        if bigram not in seen:
            seen.add(bigram)
            candidates.append(bigram)

    return candidates
