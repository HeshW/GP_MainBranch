from __future__ import annotations

from scripts.build_targeted_training_csv import build_rows, split_rows
from scripts.merge_training_csvs import dedupe_rows


def test_build_rows_can_include_follow_up_variants() -> None:
    rows = build_rows(
        [
            {
                "id": "case_1",
                "language": "en",
                "difficulty": "hard",
                "ambiguity_group": "af_vs_psvt",
                "raw_text": "I have palpitations and dizziness.",
                "expected_conditions": ["Atrial fibrillation"],
                "follow_up_answers": ["The rhythm feels irregular."],
            }
        ],
        include_follow_up=True,
    )
    assert len(rows) == 2
    assert rows[0]["case_source"] == "targeted_first_turn"
    assert rows[1]["case_source"] == "targeted_clarified"
    assert rows[1]["pathology"] == "Atrial fibrillation"


def test_split_rows_preserves_total_count() -> None:
    rows = [{"patient_id": str(i), "pathology": "x", "combined_text": "t"} for i in range(10)]
    train_rows, val_rows, test_rows = split_rows(rows, seed=42, train_ratio=0.7, val_ratio=0.2)
    assert len(train_rows) + len(val_rows) + len(test_rows) == 10


def test_dedupe_rows_uses_text_and_label() -> None:
    rows = [
        {"combined_text": "same", "pathology": "A"},
        {"combined_text": "same", "pathology": "A"},
        {"combined_text": "same", "pathology": "B"},
    ]
    deduped = dedupe_rows(rows)
    assert len(deduped) == 2
