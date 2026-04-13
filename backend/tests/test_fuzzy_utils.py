"""Tests for fuzzy matching utilities and integration points."""

from __future__ import annotations

from collections import Counter

import pytest

from manager import symptom_validator as symptom_validator_module
from manager.fuzzy_utils import (
    FUZZY_SYMPTOM_CONFIDENCE,
    _build_alias_index,
    _is_arabic_text,
    extract_candidate_tokens,
    fuzzy_match_lab_name,
    fuzzy_match_symptom,
    is_available,
)
from manager.symptom_parser import SYMPTOM_PATTERNS, parse_symptoms


pytestmark = pytest.mark.skipif(not is_available(), reason="rapidfuzz is not installed")


def test_alias_index_contains_english_aliases() -> None:
    index = _build_alias_index(SYMPTOM_PATTERNS)
    assert "fatigue" in index
    assert index["fatigue"] == "fatigue"
    assert "dizzy" in index
    assert index["dizzy"] == "dizziness"


def test_alias_index_excludes_arabic_aliases() -> None:
    index = _build_alias_index(SYMPTOM_PATTERNS)
    assert all(not _is_arabic_text(alias) for alias in index)


def test_fuzzy_match_symptom_misspelled_word() -> None:
    index = _build_alias_index(SYMPTOM_PATTERNS)
    result = fuzzy_match_symptom("fatiqued", index)
    assert result is not None
    assert result[0] == "fatigue"


def test_fuzzy_match_symptom_bigram_misspelling() -> None:
    index = _build_alias_index(SYMPTOM_PATTERNS)
    result = fuzzy_match_symptom("chset pain", index)
    assert result is not None
    assert result[0] == "chest pain"


def test_fuzzy_match_symptom_unrelated_returns_none() -> None:
    index = _build_alias_index(SYMPTOM_PATTERNS)
    assert fuzzy_match_symptom("xyzqwerty", index) is None


def test_extract_candidate_tokens_includes_words_and_bigrams() -> None:
    tokens = extract_candidate_tokens("chest pain and shortness of breath")
    assert "chest" in tokens
    assert "pain" in tokens
    assert "chest pain" in tokens


def test_extract_candidate_tokens_deduplicates_values() -> None:
    tokens = extract_candidate_tokens("fatigue fatigue fatigue")
    assert tokens.count("fatigue") == 1


def test_fuzzy_match_lab_name_misspelling() -> None:
    aliases = {
        "haemoglobin": "hemoglobin",
        "hgb": "hemoglobin",
        "glucose": "glucose",
    }
    result = fuzzy_match_lab_name("haemoglobn", aliases)
    assert result is not None
    assert result[0] == "hemoglobin"


def test_parse_symptoms_fuzzy_recovers_misspelling() -> None:
    parsed = parse_symptoms("Patient has fatiqued for two days.")
    symptoms = [item["symptom"] for item in parsed["symptoms"] if not item.get("negated")]
    assert "fatigue" in symptoms


def test_parse_symptoms_exact_not_downgraded() -> None:
    parsed = parse_symptoms("Patient has fatigue and fever")
    non_negated = [item for item in parsed["symptoms"] if not item.get("negated")]
    exact_items = [item for item in non_negated if not item.get("fuzzy")]
    assert exact_items
    assert all(item["confidence"] == 0.85 for item in exact_items)


def test_parse_symptoms_fuzzy_confidence_is_lower() -> None:
    parsed = parse_symptoms("Patient has fatiqued.")
    fuzzy_items = [item for item in parsed["symptoms"] if item.get("fuzzy")]
    if fuzzy_items:
        assert all(item["confidence"] <= FUZZY_SYMPTOM_CONFIDENCE + 0.01 for item in fuzzy_items)


def test_parse_symptoms_no_duplicate_canonical_entries() -> None:
    parsed = parse_symptoms("Patient has fatigue and fatiqued")
    non_negated = [item for item in parsed["symptoms"] if not item.get("negated")]
    counts = Counter(item["symptom"] for item in non_negated)
    assert counts["fatigue"] == 1


def test_canonical_lab_name_uses_fuzzy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        symptom_validator_module,
        "_get_lab_aliases",
        lambda: {
            "haemoglobin": "hemoglobin",
            "hgb": "hemoglobin",
            "glucose": "glucose",
        },
    )
    assert symptom_validator_module._canonical_lab_name("haemoglobn") == "hemoglobin"


def test_canonical_lab_name_unknown_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(symptom_validator_module, "_get_lab_aliases", lambda: {"hgb": "hemoglobin"})
    assert symptom_validator_module._canonical_lab_name("xyzqwerty") == "xyzqwerty"
