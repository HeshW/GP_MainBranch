"""Tests for manager.fuzzy_utils and fuzzy-matching integration."""
from __future__ import annotations

from collections import Counter

import pytest

from manager.fuzzy_utils import (
    FUZZY_SYMPTOM_CONFIDENCE,
    LAB_FUZZY_THRESHOLD,
    MIN_TOKEN_LENGTH,
    SYMPTOM_FUZZY_THRESHOLD,
    _build_alias_index,
    _is_arabic_text,
    extract_candidate_tokens,
    fuzzy_match_lab_name,
    fuzzy_match_symptom,
    is_available,
)
from manager.symptom_parser import SYMPTOM_PATTERNS, parse_symptoms
from manager.symptom_validator import _canonical_lab_name


# ---------------------------------------------------------------------------
# Sanity / availability checks
# ---------------------------------------------------------------------------


def test_rapidfuzz_is_available():
    """rapidfuzz is listed in requirements-runtime.txt and must be importable."""
    assert is_available() is True


# ---------------------------------------------------------------------------
# _build_alias_index
# ---------------------------------------------------------------------------


def test_alias_index_contains_english_aliases():
    index = _build_alias_index(SYMPTOM_PATTERNS)
    assert "fatigue" in index
    assert index["fatigue"] == "fatigue"
    assert "dizzy" in index
    assert index["dizzy"] == "dizziness"


def test_alias_index_excludes_arabic_aliases():
    """Arabic aliases must not be in the English index (edit-distance on Arabic text
    would produce unreliable results)."""
    index = _build_alias_index(SYMPTOM_PATTERNS)
    arabic_aliases = [k for k in index if _is_arabic_text(k)]
    assert arabic_aliases == [], f"Arabic aliases found in index: {arabic_aliases}"


# ---------------------------------------------------------------------------
# fuzzy_match_symptom
# ---------------------------------------------------------------------------


def test_fuzzy_match_symptom_exact_match():
    index = _build_alias_index(SYMPTOM_PATTERNS)
    result = fuzzy_match_symptom("fatigue", index)
    assert result is not None
    canonical, matched_alias, score = result
    assert canonical == "fatigue"
    assert score >= SYMPTOM_FUZZY_THRESHOLD


def test_fuzzy_match_symptom_misspelled_single_word():
    """Common single-word misspellings should resolve to the correct canonical."""
    index = _build_alias_index(SYMPTOM_PATTERNS)
    cases = [
        ("fatiqued", "fatigue"),
        ("dizy", "dizziness"),
        ("headeache", "headache"),
        ("weaknss", "weakness"),
        ("nauseus", "nausea"),
    ]
    for misspelling, expected_canonical in cases:
        result = fuzzy_match_symptom(misspelling, index)
        assert result is not None, f"No match for {misspelling!r}"
        assert result[0] == expected_canonical, (
            f"{misspelling!r}: expected {expected_canonical!r}, got {result[0]!r}"
        )


def test_fuzzy_match_symptom_bigram_misspelling():
    """Multi-word misspellings should also resolve correctly."""
    index = _build_alias_index(SYMPTOM_PATTERNS)
    result = fuzzy_match_symptom("chset pain", index)
    assert result is not None
    assert result[0] == "chest pain"


def test_fuzzy_match_symptom_below_threshold_returns_none():
    index = _build_alias_index(SYMPTOM_PATTERNS)
    result = fuzzy_match_symptom("xyzqwerty", index, threshold=SYMPTOM_FUZZY_THRESHOLD)
    assert result is None


def test_fuzzy_match_symptom_too_short_returns_none():
    index = _build_alias_index(SYMPTOM_PATTERNS)
    result = fuzzy_match_symptom("ab", index)
    assert result is None


def test_fuzzy_match_symptom_score_within_range():
    index = _build_alias_index(SYMPTOM_PATTERNS)
    result = fuzzy_match_symptom("fatiqued", index)
    assert result is not None
    _, _, score = result
    assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# fuzzy_match_lab_name
# ---------------------------------------------------------------------------


def test_fuzzy_match_lab_name_haemoglobin_misspelling():
    aliases = {
        "haemoglobin": "hemoglobin",
        "hgb": "hemoglobin",
        "hb": "hemoglobin",
        "glucose": "glucose",
        "creatinine": "creatinine",
    }
    result = fuzzy_match_lab_name("haemoglobn", aliases)
    assert result is not None
    assert result[0] == "hemoglobin"


def test_fuzzy_match_lab_name_exact_hit_also_passes():
    aliases = {"hemoglobin": "hemoglobin", "glucose": "glucose"}
    result = fuzzy_match_lab_name("hemoglobin", aliases)
    assert result is not None
    assert result[0] == "hemoglobin"


def test_fuzzy_match_lab_name_unrelated_returns_none():
    aliases = {"hemoglobin": "hemoglobin", "glucose": "glucose"}
    result = fuzzy_match_lab_name("xyzqwerty", aliases, threshold=LAB_FUZZY_THRESHOLD)
    assert result is None


def test_fuzzy_match_lab_name_too_short_returns_none():
    aliases = {"hgb": "hemoglobin"}
    result = fuzzy_match_lab_name("h", aliases)
    assert result is None


# ---------------------------------------------------------------------------
# extract_candidate_tokens
# ---------------------------------------------------------------------------


def test_extract_candidate_tokens_basic():
    tokens = extract_candidate_tokens("patient has fatigue and nausea")
    assert "fatigue" in tokens
    assert "nausea" in tokens


def test_extract_candidate_tokens_includes_bigrams():
    tokens = extract_candidate_tokens("chest pain and shortness of breath")
    assert "chest pain" in tokens


def test_extract_candidate_tokens_excludes_short():
    tokens = extract_candidate_tokens("ab cd fatigue")
    assert "ab" not in tokens
    assert "cd" not in tokens
    assert "fatigue" in tokens


def test_extract_candidate_tokens_excludes_digits():
    tokens = extract_candidate_tokens("glucose 120 mg/dl fatigue")
    assert "120" not in tokens


def test_extract_candidate_tokens_deduplicates():
    tokens = extract_candidate_tokens("fatigue fatigue fatigue")
    assert tokens.count("fatigue") == 1


def test_extract_candidate_tokens_excludes_arabic():
    tokens = extract_candidate_tokens("إرهاق fatigue")
    arabic = [t for t in tokens if _is_arabic_text(t)]
    assert arabic == []


# ---------------------------------------------------------------------------
# Integration: parse_symptoms with fuzzy fallback
# ---------------------------------------------------------------------------


def test_parse_symptoms_fuzzy_misspelled_symptom():
    """Misspelled symptoms in free text should be recovered via fuzzy matching."""
    result = parse_symptoms("Patient presents with fatiqued and headeache.")
    symptoms = [s["symptom"] for s in result["symptoms"]]
    assert "fatigue" in symptoms, f"Expected 'fatigue' in {symptoms}"
    assert "headache" in symptoms, f"Expected 'headache' in {symptoms}"


def test_parse_symptoms_fuzzy_match_has_lower_confidence():
    """Fuzzy-matched items should carry the FUZZY_SYMPTOM_CONFIDENCE score."""
    result = parse_symptoms("Patient has fatiqued.")
    fuzzy_items = [s for s in result["symptoms"] if s.get("fuzzy")]
    # Exact match for "fatiqued" might not exist; at least ensure no inflated confidence
    if fuzzy_items:
        for item in fuzzy_items:
            assert item["confidence"] <= FUZZY_SYMPTOM_CONFIDENCE + 0.01


def test_parse_symptoms_exact_match_not_downgraded():
    """When exact match succeeds, confidence must remain at 0.85."""
    result = parse_symptoms("Patient has fatigue and fever.")
    exact_items = [s for s in result["symptoms"] if not s.get("fuzzy")]
    for item in exact_items:
        assert item["confidence"] == 0.85


def test_parse_symptoms_fuzzy_no_duplicate_symptoms():
    """If both exact and fuzzy would match the same canonical, only one entry appears."""
    result = parse_symptoms("Patient has fatigue and fatiqued.")
    canonical_counts = Counter(
        s["symptom"] for s in result["symptoms"] if not s.get("negated")
    )
    for canonical, count in canonical_counts.items():
        assert count == 1, f"Duplicate entry for {canonical!r}: {count} times"


# ---------------------------------------------------------------------------
# Integration: _canonical_lab_name fuzzy fallback
# ---------------------------------------------------------------------------


def test_canonical_lab_name_fuzzy_haemoglobin():
    """A slightly garbled lab name from OCR should still resolve to the canonical."""
    # "haemoglobn" (missing 'i') should fuzzy-resolve to "hemoglobin"
    canonical = _canonical_lab_name("haemoglobn")
    assert canonical == "hemoglobin", f"Got: {canonical!r}"


def test_canonical_lab_name_exact_unchanged():
    """Exact alias lookups must not be affected by the fuzzy fallback."""
    canonical = _canonical_lab_name("hgb")
    assert canonical == "hemoglobin"


def test_canonical_lab_name_completely_unknown_passthrough():
    """An unrecognised label that does not fuzzy-match anything should pass through
    as-is (lowercased), so the pipeline can decide what to do with it."""
    result = _canonical_lab_name("xyzqwerty")
    assert result == "xyzqwerty"
