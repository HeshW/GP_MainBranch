from __future__ import annotations

import csv
import shutil
import uuid
from pathlib import Path

from scripts.generate_augmented_ddxplus_data import (
    build_variant_text,
    generate_augmented_rows,
    load_rows,
    split_symptoms,
    write_rows,
)


def test_split_symptoms_preserves_order() -> None:
    assert split_symptoms("chest pain, shortness of breath, palpitations") == [
        "chest pain",
        "shortness of breath",
        "palpitations",
    ]


def test_build_variant_text_supports_mixed_ar() -> None:
    text = build_variant_text("42", "M", ["chest pain", "palpitations"], "mixed_ar")
    assert "42" in text
    assert "chest pain" in text
    assert "palpitations" in text


def test_generate_augmented_rows_adds_variants() -> None:
    rows = [
        {
            "patient_id": "train_1",
            "age": "42",
            "sex": "M",
            "symptoms_text": "chest pain, palpitations",
            "pathology": "Atrial fibrillation",
            "combined_text": "Patient: 42 year old M. Presenting symptoms: chest pain, palpitations",
        }
    ]
    augmented = generate_augmented_rows(rows, styles=["clinical_en", "patient_en"], include_original=True)
    assert len(augmented) == 3
    assert {row["augmentation_style"] for row in augmented} == {"original", "clinical_en", "patient_en"}
    assert all(row["pathology"] == "Atrial fibrillation" for row in augmented)


def test_write_and_load_round_trip() -> None:
    tmp_path = Path("backend/tests/.tmp_augmented") / uuid.uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        source = tmp_path / "source.csv"
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["patient_id", "age", "sex", "symptoms_text", "pathology", "combined_text"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "patient_id": "train_2",
                    "age": "30",
                    "sex": "F",
                    "symptoms_text": "fatigue, thirst",
                    "pathology": "Diabetes",
                    "combined_text": "Patient: 30 year old F. Presenting symptoms: fatigue, thirst",
                }
            )

        rows = load_rows(source)
        augmented = generate_augmented_rows(rows, styles=["brief_en"], include_original=False)
        output = tmp_path / "augmented.csv"
        write_rows(output, augmented)

        written = load_rows(output)
        assert len(written) == 1
        assert written[0]["augmentation_style"] == "brief_en"
        assert written[0]["pathology"] == "Diabetes"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
