"""Generate natural-text augmentation data from DDX-style CSVs.

This script helps improve generalization by expanding structured DDX-like rows
into multiple patient-style text variants that are closer to real user input.

Expected input columns:
- patient_id
- age
- sex
- symptoms_text
- pathology
- combined_text (optional; preserved if present)

Output format:
- keeps the original diagnosis label (`pathology`)
- writes new `combined_text` values suitable for classifier fine-tuning
- adds metadata columns describing the augmentation source/style/language
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_STYLES = ("clinical_en", "patient_en", "mixed_ar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate augmented natural-text DDXPlus rows.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input natural CSV.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV with augmented rows.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=list(DEFAULT_STYLES),
        help=f"Augmentation styles to generate. Default: {', '.join(DEFAULT_STYLES)}",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Include the original input row as one of the output rows.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for quick experiments.",
    )
    return parser.parse_args()


def split_symptoms(symptoms_text: str) -> list[str]:
    if not str(symptoms_text or "").strip():
        return []
    parts = [item.strip() for item in str(symptoms_text).split(",")]
    return [item for item in parts if item]


def normalize_sex(value: str) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"m", "male"}:
        return "male"
    if lowered in {"f", "female"}:
        return "female"
    return lowered or "patient"


def english_join(items: list[str]) -> str:
    if not items:
        return "symptoms"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def arabic_join(items: list[str]) -> str:
    if not items:
        return "أعراض"
    if len(items) == 1:
        return items[0]
    return " و ".join(items)


def select_key_symptoms(symptoms: list[str], limit: int = 4) -> list[str]:
    return symptoms[:limit]


def build_variant_text(age: str, sex: str, symptoms: list[str], style: str) -> str:
    age_text = str(age or "").strip() or "unknown"
    sex_text = normalize_sex(sex)
    key_symptoms = select_key_symptoms(symptoms)
    symptoms_en = english_join(key_symptoms)
    symptoms_ar = arabic_join(key_symptoms)

    if style == "clinical_en":
        return f"Patient is a {age_text}-year-old {sex_text} presenting with {symptoms_en}."
    if style == "patient_en":
        return f"I am a {age_text}-year-old {sex_text} and I have {symptoms_en}."
    if style == "mixed_ar":
        return f"أنا مريض عمري {age_text} سنة وعندي {symptoms_ar}."
    if style == "brief_en":
        return f"Main symptoms: {symptoms_en}."
    raise ValueError(f"Unsupported augmentation style: {style}")


def build_augmented_row(row: dict[str, str], style: str) -> dict[str, str]:
    symptoms = split_symptoms(row.get("symptoms_text", ""))
    text = build_variant_text(
        age=row.get("age", ""),
        sex=row.get("sex", ""),
        symptoms=symptoms,
        style=style,
    )
    augmented = dict(row)
    augmented["combined_text"] = text
    augmented["augmentation_style"] = style
    augmented["augmentation_language"] = "mixed" if style == "mixed_ar" else "en"
    augmented["source_patient_id"] = row.get("patient_id", "")
    return augmented


def load_rows(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            rows.append({key: str(value) for key, value in row.items()})
    return rows


def generate_augmented_rows(
    rows: Iterable[dict[str, str]],
    styles: Iterable[str],
    include_original: bool = False,
) -> list[dict[str, str]]:
    augmented: list[dict[str, str]] = []
    for row in rows:
        if include_original:
            original = dict(row)
            original["augmentation_style"] = "original"
            original["augmentation_language"] = "en"
            original["source_patient_id"] = row.get("patient_id", "")
            augmented.append(original)
        for style in styles:
            augmented.append(build_augmented_row(row, style))
    return augmented


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv, max_rows=args.max_rows)
    augmented = generate_augmented_rows(
        rows,
        styles=args.styles,
        include_original=args.include_original,
    )
    write_rows(args.output_csv, augmented)
    print(
        f"Wrote {len(augmented)} augmented rows from {len(rows)} source rows to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
