"""Build a graduation-discussion evaluation pack from existing AI artifacts.

This script is intentionally inference-free by default: it reads saved
classifier test outputs, inventories the FAISS/RAG artifacts, and writes a
compact Markdown + JSON report. That makes it useful on machines that do not
have GPU/PyTorch/FAISS installed and avoids retraining.

Typical usage from the repository root:

    python backend/scripts/build_discussion_evaluation.py

For a full RAG retrieval confusion matrix, run ``evaluate_rag_confusion.py`` in
an environment with ``faiss-cpu`` installed, then cite those files beside this
discussion pack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CLASSIFIER_DIR_CANDIDATES = (
    Path("backend/artifacts/artifacts/clinicalbert_classifier_natural"),
    Path("backend/artifacts/clinicalbert_classifier_natural"),
)
DEFAULT_RAG_DIR_CANDIDATES = (
    Path("backend/artifacts/artifacts/faiss_data_natural"),
    Path("backend/artifacts/faiss_data_natural"),
)
DEFAULT_RAG_METRICS_CANDIDATES = (
    Path("data/evaluation/archive/rag_natural/rag_metrics_summary.json"),
    Path("data/evaluation/rag_natural/rag_metrics_summary.json"),
    Path("data/evaluation/rag/rag_metrics_summary.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create discussion-ready evaluation files from existing RAG/classifier artifacts."
    )
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=None,
        help="Directory containing classifier summary/test_predictions/report files.",
    )
    parser.add_argument(
        "--rag-index-dir",
        type=Path,
        default=None,
        help="Directory containing medical_cases.index and metadata_mapping.pkl/json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation/archive/discussion"),
        help="Directory where discussion evaluation files will be written.",
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Allow loading RAG pickle metadata when no matching .sha256 file exists.",
    )
    return parser.parse_args()


def resolve_existing_path(explicit: Path | None, candidates: tuple[Path, ...]) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def build_label_metrics(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    labels = sorted(
        {
            str(row.get("true_label", "")).strip()
            for row in rows
            if str(row.get("true_label", "")).strip()
        }
        | {
            str(row.get("predicted_label", "")).strip()
            for row in rows
            if str(row.get("predicted_label", "")).strip()
        }
    )
    total = len(rows)
    report_rows: list[dict[str, Any]] = []
    macro_precision = macro_recall = macro_f1 = 0.0
    weighted_precision = weighted_recall = weighted_f1 = 0.0

    for label in labels:
        tp = sum(
            1
            for row in rows
            if row.get("true_label") == label and row.get("predicted_label") == label
        )
        fp = sum(
            1
            for row in rows
            if row.get("true_label") != label and row.get("predicted_label") == label
        )
        fn = sum(
            1
            for row in rows
            if row.get("true_label") == label and row.get("predicted_label") != label
        )
        support = sum(1 for row in rows if row.get("true_label") == label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        report_rows.append(
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

    label_count = len(labels)
    averages = {
        "macro_precision": safe_div(macro_precision, label_count),
        "macro_recall": safe_div(macro_recall, label_count),
        "macro_f1": safe_div(macro_f1, label_count),
        "weighted_precision": safe_div(weighted_precision, total),
        "weighted_recall": safe_div(weighted_recall, total),
        "weighted_f1": safe_div(weighted_f1, total),
    }
    return report_rows, averages


def classifier_confusion_pairs(rows: list[dict[str, str]], limit: int = 20) -> list[dict[str, Any]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        true_label = str(row.get("true_label", "")).strip()
        predicted_label = str(row.get("predicted_label", "")).strip()
        if true_label and predicted_label and true_label != predicted_label:
            pair_counts[(true_label, predicted_label)] += 1
    return [
        {"true_label": true_label, "predicted_label": predicted_label, "count": count}
        for (true_label, predicted_label), count in pair_counts.most_common(limit)
    ]


def evaluate_classifier_artifact(classifier_dir: Path | None) -> dict[str, Any]:
    if classifier_dir is None:
        return {"available": False, "reason": "Classifier artifact directory was not found."}

    summary = read_json(classifier_dir / "summary.json")
    label_map = read_json(classifier_dir / "label_map.json")
    predictions = read_csv_dicts(classifier_dir / "test_predictions.csv")
    saved_report = read_csv_dicts(classifier_dir / "test_classification_report.csv")

    if not predictions:
        return {
            "available": False,
            "classifier_dir": str(classifier_dir),
            "reason": "Missing or empty test_predictions.csv.",
        }

    total = len(predictions)
    correct = sum(1 for row in predictions if to_bool(row.get("correct")))
    report_rows, averages = build_label_metrics(predictions)
    worst_classes = sorted(report_rows, key=lambda row: (float(row["f1_score"]), -int(row["support"])))[:10]
    confusion_pairs = classifier_confusion_pairs(predictions)

    return {
        "available": True,
        "classifier_dir": str(classifier_dir),
        "summary_file": summary,
        "num_test_cases": total,
        "num_classes": len(label_map.get("label_to_id") or label_map.get("label2id") or {}),
        "accuracy_from_predictions": safe_div(correct, total),
        "macro_precision_from_predictions": averages["macro_precision"],
        "macro_recall_from_predictions": averages["macro_recall"],
        "macro_f1_from_predictions": averages["macro_f1"],
        "weighted_f1_from_predictions": averages["weighted_f1"],
        "saved_report_rows": len(saved_report),
        "worst_classes_by_f1": worst_classes,
        "top_confusion_pairs": confusion_pairs,
        "recomputed_report_rows": report_rows,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rag_metadata(index_dir: Path, *, allow_unsafe_pickle: bool) -> tuple[dict[str, Any] | None, str | None]:
    json_path = index_dir / "metadata_mapping.json"
    if json_path.exists():
        return read_json(json_path), None

    pickle_path = index_dir / "metadata_mapping.pkl"
    if not pickle_path.exists():
        return None, "Missing metadata_mapping.json or metadata_mapping.pkl."

    hash_path = index_dir / "metadata_mapping.pkl.sha256"
    if hash_path.exists():
        expected = hash_path.read_text(encoding="utf-8").strip().split()[0].lower()
        actual = sha256_file(pickle_path)
        if expected != actual:
            return None, f"metadata_mapping.pkl hash mismatch: expected={expected} actual={actual}"
    elif not allow_unsafe_pickle:
        return None, "metadata_mapping.pkl exists but has no .sha256 file; use --allow-unsafe-pickle only for trusted artifacts."

    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return None, "RAG metadata did not deserialize to a dictionary."
    return payload, None


def evaluate_rag_artifact(index_dir: Path | None, *, allow_unsafe_pickle: bool) -> dict[str, Any]:
    if index_dir is None:
        return {"available": False, "reason": "RAG/FAISS artifact directory was not found."}

    index_path = index_dir / "medical_cases.index"
    index_info = read_json(index_dir / "index_info.json")
    metadata, metadata_error = load_rag_metadata(index_dir, allow_unsafe_pickle=allow_unsafe_pickle)
    pathologies = metadata.get("pathologies", []) if isinstance(metadata, dict) else []
    patient_ids = metadata.get("patient_ids", []) if isinstance(metadata, dict) else []
    labels = [str(item).strip() for item in pathologies if str(item).strip()]
    label_counts = Counter(labels)

    faiss_available = False
    try:
        __import__("faiss")
        faiss_available = True
    except Exception:
        faiss_available = False

    retrieval_metrics: dict[str, Any] = {}
    for metrics_path in DEFAULT_RAG_METRICS_CANDIDATES:
        if metrics_path.exists():
            retrieval_metrics = read_json(metrics_path)
            retrieval_metrics["metrics_path"] = str(metrics_path)
            break

    return {
        "available": index_path.exists(),
        "rag_index_dir": str(index_dir),
        "index_file_present": index_path.exists(),
        "index_file_size_bytes": index_path.stat().st_size if index_path.exists() else 0,
        "index_info": index_info,
        "metadata_loaded": metadata is not None,
        "metadata_error": metadata_error,
        "num_metadata_cases": len(pathologies) or len(patient_ids),
        "num_metadata_classes": len(label_counts),
        "top_10_classes_by_support": [
            {"label": label, "support": count}
            for label, count in label_counts.most_common(10)
        ],
        "faiss_python_available": faiss_available,
        "retrieval_metrics": retrieval_metrics,
        "full_retrieval_eval_command": (
            f"python backend/scripts/evaluate_rag_confusion.py --index-dir {index_dir} "
            "--output-dir data/evaluation/archive/rag_natural"
        ),
    }


def retraining_recommendation(classifier: dict[str, Any], rag: dict[str, Any]) -> str:
    if not classifier.get("available"):
        return "Retrain or restore classifier artifacts before discussion; test predictions are missing."
    accuracy = float(classifier.get("accuracy_from_predictions", 0.0))
    macro_f1 = float(classifier.get("macro_f1_from_predictions", 0.0))
    rag_has_assets = bool(rag.get("available")) and bool(rag.get("metadata_loaded"))
    if accuracy >= 0.95 and macro_f1 >= 0.90 and rag_has_assets:
        return (
            "No retraining is needed for the current discussion. Use the existing classifier "
            "test metrics and run the RAG retrieval evaluation if FAISS is installed."
        )
    return (
        "Avoid retraining first: run the evaluation scripts and inspect weak classes. "
        "Retrain only if the project needs new labels/data, a different dataset split, "
        "or the measured metrics are below the target for the discussion."
    )


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    classifier = summary["classifier"]
    rag = summary["rag"]
    lines = [
        "# Graduation Discussion Evaluation",
        "",
        "## What The System Does",
        "",
        "- RAG uses ClinicalBERT embeddings plus a FAISS medical-case index to retrieve similar historical cases.",
        "- The fine-tuned classifier uses the ClinicalBERT sequence-classification head to predict a diagnosis label from natural symptom text.",
        "- The diagnosis engine fuses rules, RAG candidates, and classifier predictions, then adds clarification/safety metadata.",
        "",
        "## Classifier Artifact Evaluation",
        "",
    ]
    if classifier.get("available"):
        source = classifier.get("summary_file", {})
        lines.extend(
            [
                f"- Artifact: `{classifier['classifier_dir']}`",
                f"- Test cases: {classifier['num_test_cases']}",
                f"- Classes: {classifier['num_classes']}",
                f"- Accuracy from saved predictions: {classifier['accuracy_from_predictions']:.4f}",
                f"- Macro F1 from saved predictions: {classifier['macro_f1_from_predictions']:.4f}",
                f"- Weighted F1 from saved predictions: {classifier['weighted_f1_from_predictions']:.4f}",
                f"- Saved training summary accuracy: {float(source.get('test_accuracy', 0.0)):.4f}",
                f"- Saved training summary macro F1: {float(source.get('test_macro_f1', 0.0)):.4f}",
                "",
                "Weakest classes by F1:",
            ]
        )
        for row in classifier["worst_classes_by_f1"][:5]:
            lines.append(
                f"- {row['label']}: F1={float(row['f1_score']):.4f}, support={int(row['support'])}"
            )
        lines.append("")
        lines.append("Most common classifier confusions:")
        if classifier["top_confusion_pairs"]:
            for row in classifier["top_confusion_pairs"][:5]:
                lines.append(
                    f"- {row['true_label']} -> {row['predicted_label']}: {row['count']}"
                )
        else:
            lines.append("- None in saved predictions.")
    else:
        lines.append(f"- Not available: {classifier.get('reason', 'unknown reason')}")

    lines.extend(
        [
            "",
            "## RAG Artifact Evaluation",
            "",
        ]
    )
    if rag.get("available"):
        index_info = rag.get("index_info", {})
        lines.extend(
            [
                f"- Artifact: `{rag['rag_index_dir']}`",
                f"- Index vectors: {index_info.get('num_vectors', rag.get('num_metadata_cases', 0))}",
                f"- Embedding dimension: {index_info.get('dimension', 'unknown')}",
                f"- Metadata cases: {rag.get('num_metadata_cases', 0)}",
                f"- Metadata classes: {rag.get('num_metadata_classes', 0)}",
                f"- FAISS Python available here: {rag.get('faiss_python_available')}",
            ]
        )
        retrieval_metrics = rag.get("retrieval_metrics") or {}
        if retrieval_metrics:
            lines.extend(
                [
                    f"- Retrieval metrics file: `{retrieval_metrics.get('metrics_path')}`",
                    f"- RAG top-1 accuracy: {float(retrieval_metrics.get('top_1_accuracy', 0.0)):.4f}",
                    f"- RAG top-3 accuracy: {float(retrieval_metrics.get('top_3_accuracy', 0.0)):.4f}",
                    f"- RAG top-5 accuracy: {float(retrieval_metrics.get('top_5_accuracy', 0.0)):.4f}",
                    f"- RAG macro F1: {float(retrieval_metrics.get('macro_f1', 0.0)):.4f}",
                    f"- RAG weighted F1: {float(retrieval_metrics.get('weighted_f1', 0.0)):.4f}",
                ]
            )
        lines.extend(["", "Largest classes in the RAG metadata:"])
        for row in rag.get("top_10_classes_by_support", [])[:5]:
            lines.append(f"- {row['label']}: {row['support']}")
        lines.extend(
            [
                "",
                "Full leave-one-out RAG retrieval metrics require FAISS locally:",
                "",
                f"```bash\n{rag['full_retrieval_eval_command']}\n```",
            ]
        )
        if rag.get("metadata_error"):
            lines.append(f"\nMetadata note: {rag['metadata_error']}")
    else:
        lines.append(f"- Not available: {rag.get('reason', 'unknown reason')}")

    lines.extend(
        [
            "",
            "## Retraining Decision",
            "",
            summary["retraining_recommendation"],
            "",
            "Retrain only when adding new clinical labels/data, changing the dataset split, or after evaluation shows weak target metrics.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for name, output_path in summary["outputs"].items():
        lines.append(f"- {name}: `{output_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    classifier_dir = resolve_existing_path(args.classifier_dir, DEFAULT_CLASSIFIER_DIR_CANDIDATES)
    rag_index_dir = resolve_existing_path(args.rag_index_dir, DEFAULT_RAG_DIR_CANDIDATES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classifier = evaluate_classifier_artifact(classifier_dir)
    rag = evaluate_rag_artifact(rag_index_dir, allow_unsafe_pickle=args.allow_unsafe_pickle)
    outputs = {
        "summary_json": str(args.output_dir / "discussion_evaluation_summary.json"),
        "markdown_report": str(args.output_dir / "discussion_evaluation.md"),
        "classifier_recomputed_report_csv": str(args.output_dir / "classifier_recomputed_report.csv"),
        "classifier_confusion_pairs_csv": str(args.output_dir / "classifier_confusion_pairs.csv"),
    }
    summary = {
        "classifier": {
            key: value
            for key, value in classifier.items()
            if key != "recomputed_report_rows"
        },
        "rag": rag,
        "retraining_recommendation": retraining_recommendation(classifier, rag),
        "outputs": outputs,
    }

    if classifier.get("available"):
        write_csv_dicts(
            Path(outputs["classifier_recomputed_report_csv"]),
            classifier["recomputed_report_rows"],
            ["label", "precision", "recall", "f1_score", "support"],
        )
        write_csv_dicts(
            Path(outputs["classifier_confusion_pairs_csv"]),
            classifier["top_confusion_pairs"],
            ["true_label", "predicted_label", "count"],
        )

    with Path(outputs["summary_json"]).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_markdown_report(Path(outputs["markdown_report"]), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
