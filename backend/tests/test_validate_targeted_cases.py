from __future__ import annotations

from scripts.validate_targeted_cases import validate_case


def test_validate_case_accepts_well_formed_case() -> None:
    errors = validate_case(
        {
            "id": "case_1",
            "language": "en",
            "difficulty": "hard",
            "raw_text": "I have chest pain and shortness of breath.",
            "expected_conditions": ["Pulmonary embolism"],
            "follow_up_answers": ["The pain is pleuritic.", "It started suddenly."],
            "discriminative_symptoms": ["pleuritic chest pain"],
            "negated_symptoms": ["fever"],
        },
        seen_ids=set(),
        index=1,
    )
    assert errors == []


def test_validate_case_rejects_duplicate_or_missing_fields() -> None:
    seen = {"case_1"}
    errors = validate_case(
        {
            "id": "case_1",
            "raw_text": "",
            "expected_conditions": [],
            "follow_up_answers": [],
            "language": "fr",
            "difficulty": "extreme",
            "unknown": "value",
        },
        seen_ids=seen,
        index=2,
    )
    assert any("duplicate id" in error for error in errors)
    assert any("raw_text" in error for error in errors)
    assert any("expected_conditions" in error for error in errors)
    assert any("follow_up_answers" in error for error in errors)
    assert any("unsupported language" in error for error in errors)
    assert any("unsupported difficulty" in error for error in errors)
    assert any("unknown fields present" in error for error in errors)
