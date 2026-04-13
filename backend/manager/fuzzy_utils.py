"""Fuzzy-matching utilities for symptom and lab-name normalization.

Uses ``rapidfuzz`` (already declared in requirements-runtime.txt) to recover
from OCR errors and user misspellings that exact-regex matching cannot handle.

Intended as a *fallback* layer only: all exact/regex matches are tried first;
fuzzy matching kicks in only when no exact hit is found for a token, keeping
false-positive rates low while catching common misspelling patterns.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from rapidfuzz import fuzz, process as rf_process

    _RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover -- rapidfuzz is listed in requirements-runtime.txt
    _RAPIDFUZZ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Minimum WRatio score (0-100) to accept a fuzzy symptom match.
SYMPTOM_FUZZY_THRESHOLD: int = 75

#: Minimum WRatio score (0-100) to accept a fuzzy lab-name match.
LAB_FUZZY_THRESHOLD: int = 80

#: Confidence assigned to fuzzy-matched symptoms (lower than 0.85 for exact).
FUZZY_SYMPTOM_CONFIDENCE: float = 0.65

#: Minimum token length to attempt fuzzy matching (avoids false hits on very
#: short strings like "a" or "is").
MIN_TOKEN_LENGTH: int = 4


def is_available() -> bool:
    """Return ``True`` if *rapidfuzz* is importable at runtime."""
    return _RAPIDFUZZ_AVAILABLE


def _is_arabic_text(text: str) -> bool:
    """Return ``True`` if *text* contains any Arabic Unicode character."""
    return any("\u0600" <= ch <= "\u06ff" for ch in text)


# ---------------------------------------------------------------------------
# Symptom fuzzy matching
# ---------------------------------------------------------------------------


def _build_alias_index(
    symptom_patterns: List[Tuple[str, Tuple[str, ...]]],
) -> Dict[str, str]:
    """Return a flat ``{alias_lower: canonical}`` dict from *symptom_patterns*.

    Only English aliases are included (Arabic aliases contain characters above
    U+0600 and rarely benefit from edit-distance matching).
    """
    index: Dict[str, str] = {}
    for canonical, aliases in symptom_patterns:
        for alias in aliases:
            if not _is_arabic_text(alias):
                index[alias.lower()] = canonical
    return index


def fuzzy_match_symptom(
    token: str,
    alias_index: Dict[str, str],
    *,
    threshold: int = SYMPTOM_FUZZY_THRESHOLD,
) -> Optional[Tuple[str, str, float]]:
    """Try to fuzzy-match *token* against *alias_index*.

    Parameters
    ----------
    token:
        A word or short phrase extracted from user/OCR text (already lowercased).
    alias_index:
        Mapping of ``{alias_lower: canonical}`` built by :func:`_build_alias_index`.
    threshold:
        Minimum WRatio score to accept a match (default: :data:`SYMPTOM_FUZZY_THRESHOLD`).

    Returns
    -------
    ``(canonical, matched_alias, score)`` on success, or ``None`` if no match
    meets the threshold or if *rapidfuzz* is not available.
    """
    if not _RAPIDFUZZ_AVAILABLE:
        return None
    if len(token) < MIN_TOKEN_LENGTH:
        return None

    result = rf_process.extractOne(
        token,
        alias_index.keys(),
        scorer=fuzz.WRatio,
        score_cutoff=threshold,
    )
    if result is None:
        return None

    matched_alias, score, _ = result
    canonical = alias_index[matched_alias]
    return canonical, matched_alias, float(score)


# ---------------------------------------------------------------------------
# Lab-name fuzzy matching
# ---------------------------------------------------------------------------


def fuzzy_match_lab_name(
    name: str,
    aliases: Dict[str, str],
    *,
    threshold: int = LAB_FUZZY_THRESHOLD,
) -> Optional[Tuple[str, str, float]]:
    """Try to fuzzy-match a raw lab label *name* against *aliases*.

    Parameters
    ----------
    name:
        Raw lab name string as extracted from OCR or free text (already lowercased).
    aliases:
        Mapping of ``{alias_lower: canonical}`` (e.g. from
        ``symptom_parser._get_lab_aliases()``).
    threshold:
        Minimum WRatio score to accept (default: :data:`LAB_FUZZY_THRESHOLD`).

    Returns
    -------
    ``(canonical, matched_alias, score)`` on success, or ``None``.
    """
    if not _RAPIDFUZZ_AVAILABLE:
        return None
    if len(name) < MIN_TOKEN_LENGTH:
        return None

    result = rf_process.extractOne(
        name,
        aliases.keys(),
        scorer=fuzz.WRatio,
        score_cutoff=threshold,
    )
    if result is None:
        return None

    matched_alias, score, _ = result
    canonical = aliases[matched_alias]
    return canonical, matched_alias, float(score)


# ---------------------------------------------------------------------------
# Text-level helpers used by the symptom parser
# ---------------------------------------------------------------------------


def extract_candidate_tokens(text: str) -> List[str]:
    """Return a de-duplicated list of candidate tokens (words and bigrams) from *text*.

    Tokens are extracted with a word-boundary regex so that punctuation
    attached to words (e.g. ``"fatigue."`` or ``"nausea,"``) is stripped.
    Very short tokens, pure-numeric tokens, and Arabic tokens are excluded
    to reduce noise.  Arabic symptoms are handled separately by exact patterns.
    """
    raw_words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text)
    words = [w for w in raw_words if len(w) >= MIN_TOKEN_LENGTH and not w.isdigit()
             and not _is_arabic_text(w)]
    seen: Set[str] = set()
    candidates: List[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            candidates.append(w)
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        if bigram not in seen:
            seen.add(bigram)
            candidates.append(bigram)
    return candidates
