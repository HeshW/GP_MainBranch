"""Rebuild the FAISS medical-case index from a DDXPlus dataset on disk.

This script is the production-style replacement for the notebook flow that
sampled 1,000 rows per split. By default, it uses the FULL dataset.

Expected dataset structure
--------------------------
dataset_dir/
  train/
  test/
  validate/  (or validation/)

Usage
-----
From repo root:

    .\\.venv_rag\\Scripts\\python.exe backend\\scripts\\rebuild_faiss_from_ddx.py ^
        --dataset-dir C:\\path\\to\\DDX\\raw\\ddxplus_hf ^
        --out-dir backend\\faiss_data

Optional sampling:

    --sample-size 1000
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from models.diagnosis.rag import ClinicalBERTEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild FAISS index from DDXPlus data.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to ddxplus_hf dataset directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("backend/faiss_data"), help="Output directory for FAISS files.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size per split. Default uses the full dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validate"],
        help="Dataset splits to include in the FAISS index. Default: train validate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when sampling.")
    return parser.parse_args()


class EvidenceMapper:
    """Notebook-compatible evidence normalizer."""

    @staticmethod
    def get_text(code: Any) -> str:
        cleaned = str(code).replace("_", " ").replace("-", " ")
        cleaned = cleaned.replace("E ", "").replace("e ", "")
        cleaned = " ".join(word.capitalize() for word in cleaned.split())
        return cleaned if cleaned else str(code)


class DDXPreprocessor:
    """Convert raw DDXPlus records into embedding-ready text."""

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
                    for key in ["name", "code", "symptom"]:
                        if key in item:
                            symptoms.append(self.evidence_mapper.get_text(item[key]))
                            break
            return symptoms
        if isinstance(evidences_data, dict):
            symptoms: list[str] = []
            for code, value in evidences_data.items():
                if value in [1, "Y", True, "yes", "1", 1.0]:
                    symptoms.append(self.evidence_mapper.get_text(code))
            return symptoms
        if isinstance(evidences_data, str) and evidences_data.strip():
            try:
                parsed = ast.literal_eval(evidences_data)
                return self.parse_evidences(parsed)
            except Exception:
                try:
                    parsed = json.loads(evidences_data)
                    return self.parse_evidences(parsed)
                except Exception:
                    return [evidences_data.strip()]
        return []

    @staticmethod
    def parse_differential(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "Not available"
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return ", ".join(str(item) for item in value[:3]) if value else "Not available"
        return "Not available"

    @staticmethod
    def create_combined_text(row: dict[str, Any]) -> str:
        parts: list[str] = []
        age = row.get("age")
        sex = row.get("sex")
        if age and sex:
            parts.append(f"Patient: {age} year old {sex}")
        symptoms_text = row.get("symptoms_text")
        if symptoms_text and symptoms_text != "None reported":
            parts.append(f"Presenting symptoms: {symptoms_text}")
        return ". ".join(parts) if parts else "No information available"

    def process_split(self, split_dataset: Any, split_name: str, sample_size: int | None, seed: int) -> pd.DataFrame:
        if sample_size and len(split_dataset) > sample_size:
            indices = random.Random(seed).sample(range(len(split_dataset)), sample_size)
            split_dataset = split_dataset.select(indices)

        df = pd.DataFrame(split_dataset[:])
        processed_rows: list[dict[str, Any]] = []

        for idx, row in df.iterrows():
            symptoms_list: list[str] = []
            for col_name in ["EVIDENCES", "evidences", "symptoms", "evidence"]:
                if col_name in row and row[col_name] is not None:
                    symptoms_list = self.parse_evidences(row[col_name])
                    if symptoms_list:
                        break

            pathology = "Unknown"
            for col_name in ["PATHOLOGY", "pathology", "diagnosis", "condition"]:
                if col_name in row and row[col_name] is not None:
                    pathology = str(row[col_name])
                    break

            age = 0
            for col_name in ["AGE", "age"]:
                if col_name in row and not pd.isna(row[col_name]):
                    age = int(row[col_name])
                    break

            sex = "Unknown"
            for col_name in ["SEX", "sex", "gender"]:
                if col_name in row and row[col_name] is not None:
                    sex = str(row[col_name])
                    break

            differential = "Not available"
            for col_name in ["DIFFERENTIAL_DIAGNOSIS", "differential_diagnosis", "differential"]:
                if col_name in row and row[col_name] is not None:
                    differential = self.parse_differential(row[col_name])
                    break

            record = {
                "patient_id": f"{split_name}_{idx}",
                "age": age,
                "sex": sex,
                "symptoms_text": ", ".join(symptoms_list) if symptoms_list else "None reported",
                "pathology": pathology,
                "differential_diagnosis": differential,
            }
            record["combined_text"] = self.create_combined_text(record)
            processed_rows.append(record)

        return pd.DataFrame(processed_rows)


def load_dataset(dataset_dir: Path) -> Any:
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise ImportError("This script requires the 'datasets' package.") from exc
    return load_from_disk(str(dataset_dir))


def write_metadata_files(out_dir: Path, metadata: dict[str, Any]) -> None:
    json_path = out_dir / "metadata_mapping.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    pickle_path = out_dir / "metadata_mapping.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(metadata, handle)

    digest = hashlib.sha256(pickle_path.read_bytes()).hexdigest()
    (out_dir / "metadata_mapping.pkl.sha256").write_text(
        f"{digest}  metadata_mapping.pkl\n",
        encoding="utf-8",
    )


def build_index(
    dataset_dir: Path,
    out_dir: Path,
    sample_size: int | None,
    splits: list[str],
    seed: int,
) -> dict[str, Any]:
    try:
        import faiss
    except Exception as exc:
        raise ImportError("This script requires 'faiss-cpu'.") from exc

    dataset = load_dataset(dataset_dir)
    preprocessor = DDXPreprocessor()
    embedder = ClinicalBERTEmbedder()

    combined_embeddings: list[np.ndarray] = []
    combined_patient_ids: list[str] = []
    combined_pathologies: list[str] = []
    combined_symptoms: list[str] = []
    combined_texts: list[str] = []
    combined_splits: list[str] = []
    split_info: dict[str, Any] = {}

    requested_splits = [split for split in splits if split in dataset.keys()]
    if not requested_splits:
        raise ValueError(f"None of the requested splits were found in dataset: {splits}")

    for split_name in requested_splits:
        processed_df = preprocessor.process_split(dataset[split_name], split_name, sample_size, seed)
        embeddings = embedder.encode_text
        split_vectors = np.vstack([embeddings(text) for text in processed_df["combined_text"].tolist()])
        combined_embeddings.append(split_vectors)
        combined_patient_ids.extend(processed_df["patient_id"].tolist())
        combined_pathologies.extend(processed_df["pathology"].tolist())
        combined_symptoms.extend(processed_df["symptoms_text"].tolist())
        combined_texts.extend(processed_df["combined_text"].tolist())
        combined_splits.extend([split_name] * len(processed_df))
        split_info[split_name] = {"num_samples": len(processed_df)}

    matrix = np.vstack(combined_embeddings).astype("float32")
    faiss.normalize_L2(matrix)
    dimension = matrix.shape[1]

    nlist = min(100, len(matrix))
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(matrix)
    index.add(matrix)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "medical_cases.index"))

    metadata = {
        "patient_ids": combined_patient_ids,
        "pathologies": combined_pathologies,
        "symptoms": combined_symptoms,
        "combined_text": combined_texts,
        "splits": combined_splits,
        "num_vectors": int(index.ntotal),
        "dimension": int(dimension),
        "split_info": split_info,
        "sample_size_per_split": sample_size,
        "used_full_dataset": sample_size is None,
    }
    write_metadata_files(out_dir, metadata)

    return metadata


def main() -> None:
    args = parse_args()
    metadata = build_index(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        sample_size=args.sample_size,
        splits=args.splits,
        seed=args.seed,
    )
    print(
        {
            "num_vectors": metadata["num_vectors"],
            "dimension": metadata["dimension"],
            "used_full_dataset": metadata["used_full_dataset"],
            "sample_size_per_split": metadata["sample_size_per_split"],
            "split_info": metadata["split_info"],
        }
    )


if __name__ == "__main__":
    main()
