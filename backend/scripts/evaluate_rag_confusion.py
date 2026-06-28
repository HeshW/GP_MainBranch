"""Evaluate diagnosis-retrieval quality and export confusion metrics.

This script evaluates the project's FAISS-backed diagnosis dataset as a
nearest-neighbor multiclass classifier:

1. Load the saved FAISS index and metadata labels.
2. Reconstruct every stored vector from the index.
3. Query the index with each vector, excluding the item itself.
4. Treat the top retrieved pathology as the predicted class.
5. Export confusion matrix, per-class report, and summary metrics.

Usage
-----
From the repository root:

    .\.venv_rag\Scripts\python.exe backend\scripts\evaluate_rag_confusion.py
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the FAISS medical-case index as a multiclass diagnosis retriever."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("backend/faiss_data"),
        help="Directory containing medical_cases.index and metadata_mapping.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Directory where metrics files will be written",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for FAISS search",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k cutoff used for retrieval metrics",
    )
    return parser.parse_args()


def load_assets(index_dir: Path) -> tuple[Any, dict[str, Any]]:
    import faiss

    index_path = index_dir / "medical_cases.index"
    metadata_path = index_dir / "metadata_mapping.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    index = faiss.read_index(str(index_path))
    if hasattr(index, "make_direct_map"):
        index.make_direct_map()

    with metadata_path.open("rb") as fh:
        metadata = pickle.load(fh)

    return index, metadata


def reconstruct_all_vectors(index: Any) -> np.ndarray:
    vectors = [index.reconstruct(i) for i in range(index.ntotal)]
    matrix = np.asarray(vectors, dtype="float32")
    return matrix


def search_without_self(index: Any, vectors: np.ndarray, batch_size: int, top_k: int) -> np.ndarray:
    neighbor_count = min(index.ntotal, top_k + 1)
    predicted_neighbors: list[np.ndarray] = []

    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size].copy()
        scores, indices = index.search(batch, neighbor_count)

        cleaned_rows = []
        for row_offset, row_indices in enumerate(indices):
            self_idx = start + row_offset
            filtered = row_indices[row_indices != self_idx]
            cleaned_rows.append(filtered[:top_k])
        predicted_neighbors.append(np.vstack(cleaned_rows))

    return np.vstack(predicted_neighbors)


def top_k_accuracy(y_true: list[str], neighbor_labels: list[list[str]], k: int) -> float:
    hits = 0
    for true_label, predicted in zip(y_true, neighbor_labels):
        if true_label in predicted[:k]:
            hits += 1
    return hits / len(y_true) if y_true else 0.0


def build_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_idx[true_label], label_to_idx[pred_label]] += 1
    return matrix


def build_classification_report(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    counts = Counter(y_true)
    rows: list[dict[str, Any]] = []

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = counts.get(label, 0)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    total = len(y_true) if y_true else 1
    class_count = len(labels) if labels else 1
    averages = {
        "macro_precision": macro_precision / class_count,
        "macro_recall": macro_recall / class_count,
        "macro_f1": macro_f1 / class_count,
        "weighted_precision": weighted_precision / total,
        "weighted_recall": weighted_recall / total,
        "weighted_f1": weighted_f1 / total,
    }
    return rows, averages


def write_confusion_matrix_csv(output_path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def write_dict_rows_csv(output_path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index, metadata = load_assets(args.index_dir)
    pathologies = list(metadata["pathologies"])

    if len(pathologies) != index.ntotal:
        raise ValueError(
            f"Metadata/index size mismatch: {len(pathologies)} labels vs {index.ntotal} vectors"
        )

    vectors = reconstruct_all_vectors(index)
    neighbor_indices = search_without_self(
        index=index,
        vectors=vectors,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )

    neighbor_labels = [[pathologies[idx] for idx in row] for row in neighbor_indices]
    y_true = pathologies
    y_pred = [predicted_labels[0] for predicted_labels in neighbor_labels]

    labels = sorted(set(y_true) | set(y_pred))
    cm = build_confusion_matrix(y_true, y_pred, labels=labels)
    report_rows, averages = build_classification_report(y_true, y_pred, labels=labels)

    summary = {
        "num_cases": len(y_true),
        "num_classes": len(labels),
        "top_1_accuracy": sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true),
        "top_3_accuracy": top_k_accuracy(y_true, neighbor_labels, k=min(3, args.top_k)),
        "top_5_accuracy": top_k_accuracy(y_true, neighbor_labels, k=min(5, args.top_k)),
        "macro_precision": averages["macro_precision"],
        "macro_recall": averages["macro_recall"],
        "macro_f1": averages["macro_f1"],
        "weighted_precision": averages["weighted_precision"],
        "weighted_recall": averages["weighted_recall"],
        "weighted_f1": averages["weighted_f1"],
        "index_dir": str(args.index_dir),
    }

    write_confusion_matrix_csv(args.output_dir / "rag_confusion_matrix.csv", labels, cm)
    write_dict_rows_csv(
        args.output_dir / "rag_classification_report.csv",
        report_rows,
        ["label", "precision", "recall", "f1_score", "support"],
    )
    write_dict_rows_csv(
        args.output_dir / "rag_predictions.csv",
        [
            {
                "true_pathology": true_label,
                "predicted_pathology": pred_label,
                "top_3_predictions": " | ".join(predicted_labels[:3]),
                "top_5_predictions": " | ".join(predicted_labels[:5]),
            }
            for true_label, pred_label, predicted_labels in zip(y_true, y_pred, neighbor_labels)
        ],
        ["true_pathology", "predicted_pathology", "top_3_predictions", "top_5_predictions"],
    )

    with (args.output_dir / "rag_metrics_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved confusion matrix to: {args.output_dir / 'rag_confusion_matrix.csv'}")
    print(f"Saved classification report to: {args.output_dir / 'rag_classification_report.csv'}")
    print(f"Saved per-case predictions to: {args.output_dir / 'rag_predictions.csv'}")


if __name__ == "__main__":
    main()
