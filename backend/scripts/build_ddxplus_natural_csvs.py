"""Build clean natural-language CSVs from a DDXPlus dataset on disk.

This script prepares train/validate/test CSV files for classifier training
without label leakage. The generated `combined_text` contains only patient
demographics and presenting symptoms.

Expected dataset structure:
dataset_dir/
  train/
  test/
  validate/  (or validation/)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build natural-language CSVs from DDXPlus.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to ddxplus_hf dataset directory.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed_ddxplus"),
        help="Output directory for generated CSV files.",
    )
    return parser.parse_args()


class EvidenceMapper:
    @staticmethod
    def get_text(code: Any) -> str:
        cleaned = str(code).replace("_", " ").replace("-", " ")
        cleaned = cleaned.replace("E ", "").replace("e ", "")
        cleaned = " ".join(word.capitalize() for word in cleaned.split())
        return cleaned if cleaned else str(code)


class DDXNaturalPreprocessor:
    def __init__(self) -> None:
        self.evidence_mapper = EvidenceMapper()

    def parse_evidences(self, evidences_data: Any) -> list[str]:
        if evidences_data is None:
            return []
        if isinstance(evidences_data, list):
            symptoms: list[str] = []
            for item in evidences_data:
                if isinstance(item, str) and item.strip():
                    symptoms.append(self.evidence_mapper.get_text(item))
                elif isinstance(item, dict):
                    for key in ("name", "code", "symptom"):
                        if key in item:
                            symptoms.append(self.evidence_mapper.get_text(item[key]))
                            break
            return symptoms
        if isinstance(evidences_data, dict):
            return [
                self.evidence_mapper.get_text(code)
                for code, value in evidences_data.items()
                if value in [1, "Y", True, "yes", "1", 1.0]
            ]
        if isinstance(evidences_data, str) and evidences_data.strip():
            for parser in (ast.literal_eval, json.loads):
                try:
                    parsed = parser(evidences_data)
                    return self.parse_evidences(parsed)
                except Exception:
                    continue
            return [evidences_data.strip()]
        return []

    @staticmethod
    def build_combined_text(age: Any, sex: Any, symptoms_text: str) -> str:
        parts: list[str] = []
        if age not in (None, "", "Unknown") and sex not in (None, "", "Unknown"):
            parts.append(f"Patient: {age} year old {sex}")
        if symptoms_text and symptoms_text != "None reported":
            parts.append(f"Presenting symptoms: {symptoms_text}")
        return ". ".join(parts) if parts else "No information available"

    def process_split(self, split_dataset: Any, split_name: str) -> pd.DataFrame:
        df = pd.DataFrame(split_dataset[:])
        rows: list[dict[str, Any]] = []

        for idx, row in df.iterrows():
            evidences = []
            for col_name in ("EVIDENCES", "evidences", "symptoms", "evidence"):
                if col_name in row and row[col_name] is not None:
                    evidences = self.parse_evidences(row[col_name])
                    if evidences:
                        break

            pathology = "Unknown"
            for col_name in ("PATHOLOGY", "pathology", "diagnosis", "condition"):
                if col_name in row and row[col_name] is not None:
                    pathology = str(row[col_name]).strip()
                    break

            age = "Unknown"
            for col_name in ("AGE", "age"):
                if col_name in row and not pd.isna(row[col_name]):
                    age = int(row[col_name])
                    break

            sex = "Unknown"
            for col_name in ("SEX", "sex", "gender"):
                if col_name in row and row[col_name] is not None:
                    sex = str(row[col_name]).strip()
                    break

            symptoms_text = ", ".join(evidences) if evidences else "None reported"
            rows.append(
                {
                    "patient_id": f"{split_name}_{idx}",
                    "age": age,
                    "sex": sex,
                    "symptoms_text": symptoms_text,
                    "pathology": pathology,
                    "combined_text": self.build_combined_text(age, sex, symptoms_text),
                }
            )

        return pd.DataFrame(rows)


def load_dataset(dataset_dir: Path) -> Any:
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise ImportError("This script requires the 'datasets' package.") from exc
    return load_from_disk(str(dataset_dir))


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_dir)
    preprocessor = DDXNaturalPreprocessor()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}

    for source_split in ("train", "validate", "validation", "test"):
        if source_split not in dataset:
            continue
        normalized_split = "validate" if source_split == "validation" else source_split
        processed = preprocessor.process_split(dataset[source_split], normalized_split)
        out_path = args.out_dir / f"{normalized_split}_natural.csv"
        processed.to_csv(out_path, index=False, encoding="utf-8")
        written[normalized_split] = {
            "path": str(out_path),
            "rows": len(processed),
        }

    print(json.dumps({"out_dir": str(args.out_dir), "written": written}, indent=2))


if __name__ == "__main__":
    main()
