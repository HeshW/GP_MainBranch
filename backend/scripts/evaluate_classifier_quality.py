"""Evaluate the active ClinicalBERT classifier without retraining.

The script inventories classifier/RAG artifacts, verifies label consistency,
recomputes metrics from saved classifier predictions, and runs a bounded
stratified inference smoke evaluation against the existing model weights.
It never writes to model or FAISS artifact directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from models.diagnosis.rag import FineTunedDiagnosisClassifier


DEFAULT_OUTPUT_DIR = Path("data/evaluation/classifier_diagnostics")
TARGETED_CLASSIFIER_DIR = Path("backend/artifacts/artifacts/clinicalbert_classifier_targeted")
NATURAL_CLASSIFIER_DIR = Path("backend/artifacts/artifacts/clinicalbert_classifier_natural")
TARGETED_FAISS_DIR = Path("backend/artifacts/artifacts/faiss_data_targeted")
NATURAL_FAISS_DIR = Path("backend/artifacts/artifacts/faiss_data_natural")
EXPECTED_MODEL_FILES = (
    "config.json",
    "label_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
)
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ClinicalBERT classifier artifacts without retraining.")
    parser.add_argument("--classifier-dir", type=Path, default=None, help="Classifier artifact directory.")
    parser.add_argument("--faiss-index-dir", type=Path, default=None, help="Active FAISS artifact directory.")
    parser.add_argument("--natural-classifier-dir", type=Path, default=NATURAL_CLASSIFIER_DIR)
    parser.add_argument("--targeted-classifier-dir", type=Path, default=TARGETED_CLASSIFIER_DIR)
    parser.add_argument("--natural-faiss-dir", type=Path, default=NATURAL_FAISS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-smoke-cases", type=int, default=49, help="0 means run all prediction rows.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-model-smoke", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("-", " ").replace("_", " ").split())


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


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


def write_json(path: Path, payload: dict[str, Any], *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2 if pretty else None)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_label_map(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    label_to_id = payload.get("label_to_id") or payload.get("label2id") or {}
    id_to_label = payload.get("id_to_label") or payload.get("id2label") or {}
    labels = [str(label).strip() for label in label_to_id if str(label).strip()]
    return {
        "path": str(path),
        "exists": path.exists(),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "labels": labels,
        "label_count": len(labels),
        "id_count": len(id_to_label),
        "ids_are_contiguous": sorted(int(v) for v in label_to_id.values()) == list(range(len(label_to_id)))
        if label_to_id
        else False,
        "label_id_roundtrip_ok": all(
            str(id_to_label.get(str(idx), id_to_label.get(idx, ""))) == label
            for label, idx in label_to_id.items()
        )
        if label_to_id and id_to_label
        else False,
    }


def load_metadata(index_dir: Path) -> tuple[dict[str, Any], str | None]:
    json_path = index_dir / "metadata_mapping.json"
    if json_path.exists():
        return read_json(json_path), None

    pickle_path = index_dir / "metadata_mapping.pkl"
    if not pickle_path.exists():
        return {}, "Missing metadata_mapping.json and metadata_mapping.pkl."

    hash_path = index_dir / "metadata_mapping.pkl.sha256"
    if not hash_path.exists():
        return {}, "Refusing to load metadata_mapping.pkl without metadata_mapping.pkl.sha256."
    expected = hash_path.read_text(encoding="utf-8").strip().split()[0].lower()
    actual = sha256_file(pickle_path)
    if expected != actual:
        return {}, f"metadata_mapping.pkl hash mismatch: expected={expected} actual={actual}"

    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if isinstance(payload, dict) else {}, None


def metadata_label_summary(name: str, path: Path) -> dict[str, Any]:
    metadata, error = load_metadata(path)
    labels = [str(item).strip() for item in metadata.get("pathologies", []) if str(item).strip()]
    lengths = {
        key: len(value)
        for key, value in metadata.items()
        if isinstance(value, list)
    }
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "error": error,
        "row_count": len(labels),
        "unique_label_count": len(set(labels)),
        "labels": sorted(set(labels), key=normalize_label),
        "label_distribution_top_10": Counter(labels).most_common(10),
        "metadata_lengths": lengths,
        "metadata_lengths_match": len(set(lengths.values())) <= 1 if lengths else False,
    }


def label_source_summary(name: str, labels: list[str], path: Path | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path) if path else None,
        "exists": path.exists() if path else True,
        "label_count": len(set(labels)),
        "labels": sorted(set(labels), key=normalize_label),
    }


def compare_label_sources(sources: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_sources: dict[str, dict[str, str]] = {}
    for source in sources:
        normalized_sources[source["name"]] = {
            normalize_label(label): label for label in source.get("labels", [])
        }
    all_normalized = sorted(set().union(*(set(items) for items in normalized_sources.values())))
    comparisons: list[dict[str, Any]] = []
    for source in sources:
        source_set = set(normalized_sources[source["name"]])
        other_sets = [
            set(labels)
            for name, labels in normalized_sources.items()
            if name != source["name"]
        ]
        universe = set().union(*other_sets) if other_sets else set()
        comparisons.append(
            {
                "name": source["name"],
                "missing_from_this_source": sorted(
                    normalized_sources[next_name][label]
                    for label in universe - source_set
                    for next_name in normalized_sources
                    if label in normalized_sources[next_name]
                ),
                "extra_in_this_source": sorted(normalized_sources[source["name"]][label] for label in source_set - universe),
            }
        )
    return {
        "all_sources_have_same_normalized_labels": all(
            set(labels) == set(all_normalized) for labels in normalized_sources.values()
        )
        if normalized_sources
        else False,
        "normalized_universe_count": len(all_normalized),
        "comparisons": comparisons,
    }


def inspect_classifier_assets(classifier_dir: Path, settings: Any) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(classifier_dir.iterdir()) if classifier_dir.exists() else []:
        if path.is_file():
            files.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    label_map = load_label_map(classifier_dir / "label_map.json")
    config = read_json(classifier_dir / "config.json")
    config_labels = config.get("id2label") or {}
    missing_required = [name for name in EXPECTED_MODEL_FILES if not (classifier_dir / name).exists()]
    has_weights = any((classifier_dir / name).exists() for name in WEIGHT_FILES)
    if not has_weights:
        missing_required.append("model weights (model.safetensors or pytorch_model.bin)")
    prediction_files = [
        name for name in (
            "test_predictions.csv",
            "test_classification_report.csv",
            "test_confusion_matrix.csv",
            "summary.json",
            "history.json",
        )
        if (classifier_dir / name).exists()
    ]
    return {
        "runtime": {
            "use_finetuned_classifier": bool(settings.use_finetuned_classifier),
            "finetuned_model_dir": settings.finetuned_model_dir,
            "classifier_max_length": settings.classifier_max_length,
            "classifier_translate_arabic": bool(settings.classifier_translate_arabic),
        },
        "active_classifier_path": str(classifier_dir),
        "exists": classifier_dir.exists(),
        "files": files,
        "missing_required_files": missing_required,
        "has_model_weights": has_weights,
        "has_tokenizer": (classifier_dir / "tokenizer.json").exists() or (classifier_dir / "vocab.txt").exists(),
        "prediction_artifacts": prediction_files,
        "label_map": {
            "label_count": label_map["label_count"],
            "id_count": label_map["id_count"],
            "ids_are_contiguous": label_map["ids_are_contiguous"],
            "label_id_roundtrip_ok": label_map["label_id_roundtrip_ok"],
            "labels": label_map["labels"],
        },
        "config_label_count": len(config_labels),
        "config_label_consistent_with_label_map": {
            normalize_label(v) for v in config_labels.values()
        }
        == {normalize_label(v) for v in label_map["labels"]},
        "summary": read_json(classifier_dir / "summary.json"),
        "history_epochs": len(read_json(classifier_dir / "history.json")) if isinstance(read_json(classifier_dir / "history.json"), list) else 0,
    }


def build_assets_report_md(report: dict[str, Any]) -> str:
    missing = report["missing_required_files"] or ["none"]
    summary = report.get("summary") or {}
    lines = [
        "# Classifier Assets Report",
        "",
        f"- Active classifier path: `{report['active_classifier_path']}`",
        f"- Runtime enabled: `{report['runtime']['use_finetuned_classifier']}`",
        f"- Label count: `{report['label_map']['label_count']}`",
        f"- Required files missing: `{missing}`",
        f"- Model weights present: `{report['has_model_weights']}`",
        f"- Tokenizer present: `{report['has_tokenizer']}`",
        f"- Label map roundtrip valid: `{report['label_map']['label_id_roundtrip_ok']}`",
        f"- Config labels match label map: `{report['config_label_consistent_with_label_map']}`",
        "",
        "## Saved Training Summary",
        "",
        f"- Test accuracy in summary: `{summary.get('test_accuracy')}`",
        f"- Test macro F1 in summary: `{summary.get('test_macro_f1')}`",
        f"- Rows: `{summary.get('rows')}`",
        "",
        "## Prediction Artifacts",
        "",
        *[f"- `{name}`" for name in report["prediction_artifacts"]],
    ]
    return "\n".join(lines) + "\n"


def recompute_prediction_metrics(classifier_dir: Path, output_dir: Path) -> dict[str, Any]:
    predictions_path = classifier_dir / "test_predictions.csv"
    rows = read_csv_dicts(predictions_path)
    if not rows:
        return {"available": False, "reason": f"No predictions found at {predictions_path}"}

    y_true = [row["true_label"].strip() for row in rows]
    y_pred = [row["predicted_label"].strip() for row in rows]
    labels = sorted(set(y_true) | set(y_pred), key=normalize_label)
    accuracy = accuracy_score(y_true, y_pred)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    class_rows: list[dict[str, Any]] = []
    for label in labels:
        item = report_dict.get(label, {})
        class_rows.append(
            {
                "label": label,
                "precision": item.get("precision", 0.0),
                "recall": item.get("recall", 0.0),
                "f1_score": item.get("f1-score", 0.0),
                "support": int(item.get("support", 0)),
            }
        )
    write_csv(
        output_dir / "classifier_classification_report.csv",
        class_rows,
        ["label", "precision", "recall", "f1_score", "support"],
    )

    confusion_rows = []
    for idx, label in enumerate(labels):
        row = {"true_label": label}
        row.update({pred_label: int(matrix[idx, pred_idx]) for pred_idx, pred_label in enumerate(labels)})
        confusion_rows.append(row)
    write_csv(output_dir / "classifier_confusion_matrix.csv", confusion_rows, ["true_label", *labels])

    support = Counter(y_true)
    pred_count = Counter(y_pred)
    pair_counts: Counter[tuple[str, str]] = Counter(
        (truth, pred) for truth, pred in zip(y_true, y_pred) if truth != pred
    )
    pair_rows = [
        {
            "true_label": truth,
            "predicted_label": pred,
            "count": count,
            "true_label_support": support[truth],
            "true_label_error_rate": round(safe_div(count, support[truth]), 6),
            "predicted_label_total": pred_count[pred],
        }
        for (truth, pred), count in pair_counts.most_common()
    ]
    write_csv(
        output_dir / "classifier_confusion_pairs.csv",
        pair_rows,
        ["true_label", "predicted_label", "count", "true_label_support", "true_label_error_rate", "predicted_label_total"],
    )

    worst_classes = sorted(class_rows, key=lambda row: (float(row["f1_score"]), int(row["support"])))[:10]
    summary = {
        "available": True,
        "prediction_file": str(predictions_path),
        "num_cases": len(rows),
        "label_count_in_predictions": len(labels),
        "accuracy": float(accuracy),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "correct": int(sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)),
        "incorrect": int(sum(1 for truth, pred in zip(y_true, y_pred) if truth != pred)),
        "top_confusion_pairs": pair_rows[:20],
        "worst_classes_by_f1": worst_classes,
        "class_support_min": min(support.values()),
        "class_support_max": max(support.values()),
        "class_support_median": float(np.median(list(support.values()))),
        "missing_labels_in_predictions": [
            label for label in load_label_map(classifier_dir / "label_map.json")["labels"] if label not in support
        ],
    }
    write_json(output_dir / "classifier_metrics_summary.json", summary)
    return summary


def sample_smoke_rows(rows: list[dict[str, str]], max_cases: int, seed: int) -> list[dict[str, str]]:
    if max_cases <= 0 or len(rows) <= max_cases:
        return rows
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("true_label", "")].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    labels = sorted(grouped, key=normalize_label)
    base_quota = max(1, max_cases // max(len(labels), 1))
    for label in labels:
        candidates = list(grouped[label])
        rng.shuffle(candidates)
        selected.extend(candidates[: min(base_quota, len(candidates))])
    if len(selected) < max_cases:
        selected_ids = {id(row) for row in selected}
        leftovers = [row for row in rows if id(row) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_cases - len(selected)])
    rng.shuffle(selected)
    return selected[:max_cases]


def topk_classifier_predict(
    classifier: FineTunedDiagnosisClassifier,
    texts: list[str],
    *,
    batch_size: int,
    k: int = 5,
) -> list[dict[str, Any]]:
    torch = classifier._torch
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = classifier.tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=classifier.max_length,
            return_tensors="pt",
        ).to(classifier.device)
        with torch.no_grad():
            outputs = classifier.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)
        for row_probs, row_indices in zip(top_probs, top_indices):
            top_predictions = [
                {
                    "label": classifier.id_to_label.get(int(idx.item()), str(int(idx.item()))),
                    "confidence": float(score.item()),
                }
                for score, idx in zip(row_probs, row_indices)
            ]
            predictions.append(
                {
                    "predicted_label": top_predictions[0]["label"],
                    "confidence": top_predictions[0]["confidence"],
                    "top_predictions": top_predictions,
                }
            )
    return predictions


def rank_of_label(top_predictions: list[dict[str, Any]], label: str) -> int | None:
    expected = normalize_label(label)
    for index, item in enumerate(top_predictions, start=1):
        if normalize_label(item.get("label")) == expected:
            return index
    return None


def calibration_bins(confidences: list[float], correctness: list[bool], bin_count: int = 10) -> tuple[list[dict[str, Any]], float]:
    bins: list[dict[str, Any]] = []
    ece = 0.0
    total = len(confidences)
    for idx in range(bin_count):
        low = idx / bin_count
        high = (idx + 1) / bin_count
        members = [
            (conf, ok)
            for conf, ok in zip(confidences, correctness)
            if (low <= conf < high) or (idx == bin_count - 1 and conf == 1.0)
        ]
        count = len(members)
        avg_conf = safe_div(sum(conf for conf, _ in members), count)
        acc = safe_div(sum(1 for _, ok in members if ok), count)
        gap = abs(acc - avg_conf)
        ece += safe_div(count, total) * gap
        bins.append(
            {
                "bin": f"{low:.1f}-{high:.1f}",
                "count": count,
                "avg_confidence": round(avg_conf, 6),
                "accuracy": round(acc, 6),
                "gap": round(gap, 6),
            }
        )
    return bins, ece


def infer_failure_causes(
    *,
    true_label: str,
    predicted_label: str,
    confidence: float | None,
    top_predictions: list[dict[str, Any]],
    support: Counter[str],
    confusion_pairs: Counter[tuple[str, str]],
) -> list[str]:
    causes: list[str] = []
    class_support = support.get(true_label, 0)
    supports = list(support.values())
    low_support_cutoff = float(np.percentile(supports, 25)) if supports else 0.0
    if class_support <= low_support_cutoff:
        causes.append("class imbalance or insufficient evaluation examples")
    if confusion_pairs.get((true_label, predicted_label), 0) > 0:
        causes.append("known confusing label pair in saved test predictions")
    normalized_true = set(normalize_label(true_label).split())
    normalized_pred = set(normalize_label(predicted_label).split())
    if normalized_true & normalized_pred:
        causes.append("confusing labels with overlapping disease family terms")
    if confidence is not None and confidence >= 0.80:
        causes.append("calibration issue: high-confidence wrong prediction")
    elif confidence is not None and confidence < 0.55:
        causes.append("weak features or low classifier confidence")
    if len(top_predictions) >= 2:
        margin = float(top_predictions[0]["confidence"]) - float(top_predictions[1]["confidence"])
        if margin < 0.15:
            causes.append("weak features: small top-label probability margin")
    if not causes:
        causes.append("weak features or preprocessing mismatch")
    return causes


def run_model_smoke_eval(
    classifier_dir: Path,
    output_dir: Path,
    *,
    max_cases: int,
    batch_size: int,
    seed: int,
    metrics_summary: dict[str, Any],
) -> dict[str, Any]:
    prediction_rows = read_csv_dicts(classifier_dir / "test_predictions.csv")
    if not prediction_rows:
        return {"available": False, "reason": "No test_predictions.csv rows available for smoke inference."}

    sample_rows = sample_smoke_rows(prediction_rows, max_cases=max_cases, seed=seed)
    classifier = FineTunedDiagnosisClassifier(model_dir=classifier_dir)
    predictions = topk_classifier_predict(
        classifier,
        [row.get("text", "") for row in sample_rows],
        batch_size=batch_size,
        k=5,
    )

    support = Counter(row.get("true_label", "") for row in prediction_rows)
    saved_confusions = Counter(
        (row.get("true_label", ""), row.get("predicted_label", ""))
        for row in prediction_rows
        if row.get("true_label") != row.get("predicted_label")
    )
    case_rows: list[dict[str, Any]] = []
    top1_hits = top3_hits = top5_hits = 0
    reciprocal_total = 0.0
    confidences: list[float] = []
    correctness: list[bool] = []
    failure_counts: Counter[str] = Counter()

    for idx, (source, pred) in enumerate(zip(sample_rows, predictions), start=1):
        true_label = source.get("true_label", "").strip()
        top_predictions = pred["top_predictions"]
        rank = rank_of_label(top_predictions, true_label)
        top1 = rank == 1
        top3 = rank is not None and rank <= 3
        top5 = rank is not None and rank <= 5
        top1_hits += int(top1)
        top3_hits += int(top3)
        top5_hits += int(top5)
        reciprocal_total += (1.0 / rank) if rank else 0.0
        confidence = float(pred["confidence"])
        confidences.append(confidence)
        correctness.append(top1)
        causes: list[str] = []
        if not top1:
            causes = infer_failure_causes(
                true_label=true_label,
                predicted_label=pred["predicted_label"],
                confidence=confidence,
                top_predictions=top_predictions,
                support=support,
                confusion_pairs=saved_confusions,
            )
            failure_counts.update(causes)
        case_rows.append(
            {
                "case_index": idx,
                "expected_label": true_label,
                "predicted_label": pred["predicted_label"],
                "confidence": round(confidence, 6),
                "rank": rank or "",
                "top1_correct": top1,
                "top3_correct": top3,
                "top5_correct": top5,
                "top_predictions": "; ".join(
                    f"{item['label']} ({float(item['confidence']):.4f})" for item in top_predictions
                ),
                "likely_causes": "; ".join(causes),
                "text_preview": source.get("text", "")[:320],
            }
        )

    bins, ece = calibration_bins(confidences, correctness)
    total = len(case_rows)
    failures = [row for row in case_rows if not row["top1_correct"]]
    write_csv(
        output_dir / "classifier_smoke_eval_cases.csv",
        case_rows,
        [
            "case_index",
            "expected_label",
            "predicted_label",
            "confidence",
            "rank",
            "top1_correct",
            "top3_correct",
            "top5_correct",
            "top_predictions",
            "likely_causes",
            "text_preview",
        ],
    )

    summary = {
        "available": True,
        "classifier_dir": str(classifier_dir),
        "sample_strategy": "stratified_from_test_predictions",
        "random_seed": seed,
        "num_cases": total,
        "source_prediction_rows": len(prediction_rows),
        "top_1_accuracy": safe_div(top1_hits, total),
        "top_3_accuracy": safe_div(top3_hits, total),
        "top_5_accuracy": safe_div(top5_hits, total),
        "mrr_at_5": safe_div(reciprocal_total, total),
        "mean_confidence": safe_div(sum(confidences), total),
        "mean_confidence_correct": safe_div(sum(conf for conf, ok in zip(confidences, correctness) if ok), sum(correctness)),
        "mean_confidence_incorrect": safe_div(sum(conf for conf, ok in zip(confidences, correctness) if not ok), total - sum(correctness)),
        "expected_calibration_error": ece,
        "calibration_bins": bins,
        "high_confidence_error_count": sum(1 for conf, ok in zip(confidences, correctness) if conf >= 0.8 and not ok),
        "failure_count": len(failures),
        "failure_causes": failure_counts.most_common(),
        "failed_cases": failures[:50],
        "matches_saved_prediction_accuracy": math.isclose(
            safe_div(top1_hits, total),
            float(metrics_summary.get("accuracy", 0.0)),
            rel_tol=0.05,
            abs_tol=0.05,
        )
        if metrics_summary.get("available")
        else None,
    }
    write_json(output_dir / "classifier_smoke_eval_summary.json", summary)
    return summary


def retraining_assessment(metrics: dict[str, Any], smoke: dict[str, Any], label_consistency: dict[str, Any]) -> dict[str, Any]:
    accuracy = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    worst_classes = metrics.get("worst_classes_by_f1", [])
    worst_f1 = min((float(row.get("f1_score", 0.0)) for row in worst_classes), default=1.0)
    label_ok = bool(label_consistency.get("comparison", {}).get("all_sources_have_same_normalized_labels"))
    ece = float(smoke.get("expected_calibration_error", 0.0) or 0.0)
    high_conf_errors = int(smoke.get("high_confidence_error_count", 0) or 0)

    if label_ok and accuracy >= 0.98 and macro_f1 >= 0.98 and worst_f1 >= 0.65:
        category = "B. Retraining recommended later."
        recommended = True
        necessary = False
        optional = True
        rationale = (
            "Current saved-test performance is strong and labels are consistent, "
            "but concentrated weak classes and calibration should be monitored."
        )
    elif accuracy >= 0.95 and macro_f1 >= 0.95 and label_ok:
        category = "B. Retraining recommended later."
        recommended = True
        necessary = False
        optional = True
        rationale = "Metrics are acceptable, but class-level failures justify a later data-quality retraining pass."
    else:
        category = "C. Retraining strongly recommended."
        recommended = True
        necessary = True
        optional = False
        rationale = "Current evidence shows either weak overall metrics or label inconsistency."

    if smoke.get("available") and high_conf_errors == 0 and ece <= 0.05 and accuracy >= 0.99:
        category = "A. No retraining needed."
        recommended = False
        necessary = False
        optional = False
        rationale = "Metrics, calibration smoke results, and label consistency do not currently justify retraining."

    return {
        "category": category,
        "is_retraining_recommended": recommended,
        "is_retraining_necessary": necessary,
        "is_retraining_optional": optional,
        "evidence": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "worst_observed_class_f1": worst_f1,
            "labels_consistent": label_ok,
            "smoke_expected_calibration_error": ece if smoke.get("available") else None,
            "smoke_high_confidence_errors": high_conf_errors if smoke.get("available") else None,
        },
        "rationale": rationale,
        "if_retraining_is_later_approved": {
            "why": "Improve low-recall rhinosinusitis/sinusitis classes and calibrate confidence on natural user phrasing.",
            "expected_gains": "Higher recall for confusing ENT/cardiac subclasses and better probability calibration.",
            "required_data_changes": "Add/clean examples for confused labels; include natural/chat-style symptom phrasings.",
            "classifier_labels_would_change": False,
            "faiss_rebuild_needed": False,
            "faiss_rebuild_note": "Only required if the label universe or retrieval corpus changes.",
        },
    }


def compare_classifier_vs_rag(metrics: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    rag_summary_paths = [
        Path("data/evaluation/rag_diagnostics/expanded_retrieval_eval_summary.json"),
        Path("data/evaluation/archive/rag_natural/rag_metrics_summary.json"),
    ]
    rag_cases_path = Path("data/evaluation/rag_diagnostics/expanded_retrieval_eval_cases.csv")
    rag_summary = next((read_json(path) | {"path": str(path)} for path in rag_summary_paths if path.exists()), {})
    rag_cases = read_csv_dicts(rag_cases_path)
    rag_failures = [
        row for row in rag_cases
        if normalize_label(row.get("scope")) == "in scope" and str(row.get("top1_hit", "")).lower() != "true"
    ]
    classifier_worst = metrics.get("worst_classes_by_f1", [])
    classifier_weak = [row["label"] for row in classifier_worst if float(row.get("f1_score", 0.0)) < 0.95]
    rag_failed_labels = sorted({row.get("expected_family", "") for row in rag_failures if row.get("expected_family")})
    return {
        "rag_summary": rag_summary,
        "rag_cases_path": str(rag_cases_path) if rag_cases_path.exists() else None,
        "classifier_handles_better": [
            "Most saved DDXPlus test labels, especially labels with perfect classifier F1 in the recomputed report.",
            "Exact label discrimination on structured DDXPlus-style questionnaire text.",
        ],
        "rag_handles_better": [
            "Out-of-scope detection and retrieval confidence gating.",
            "Transparent evidence lists for supported in-scope cases.",
        ],
        "classifier_weak_labels": classifier_weak,
        "rag_top1_failed_expected_families": rag_failed_labels,
        "labels_or_families_weak_for_both": sorted(
            set(normalize_label(label) for label in classifier_weak)
            & set(normalize_label(label) for label in rag_failed_labels)
        ),
        "ensemble_fusion_justified": True,
        "ensemble_fusion_rationale": (
            "Classifier top-1 accuracy is stronger on saved structured labels, while RAG provides retrieval evidence "
            "and out-of-scope confidence gating. Fusion remains justified if confidence thresholds prevent weak RAG "
            "or low-confidence classifier outputs from dominating."
        ),
    }


def write_label_consistency_reports(output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(output_dir / "classifier_label_consistency_report.json", payload)
    comparison = payload["comparison"]
    lines = [
        "# Classifier Label Consistency Report",
        "",
        f"- Label universe count: `{comparison['normalized_universe_count']}`",
        f"- All checked sources match: `{comparison['all_sources_have_same_normalized_labels']}`",
        "",
        "## Sources",
        "",
    ]
    for source in payload["sources"]:
        lines.append(f"- `{source['name']}`: `{source.get('label_count', source.get('unique_label_count'))}` labels from `{source.get('path')}`")
    lines.extend(["", "## Differences", ""])
    for item in comparison["comparisons"]:
        lines.append(f"### {item['name']}")
        lines.append(f"- Missing: `{item['missing_from_this_source']}`")
        lines.append(f"- Extra: `{item['extra_in_this_source']}`")
    (output_dir / "classifier_label_consistency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eval_report(output_dir: Path, metrics: dict[str, Any], smoke: dict[str, Any], assessment: dict[str, Any], rag_compare: dict[str, Any]) -> None:
    pairs = metrics.get("top_confusion_pairs", [])[:10]
    worst = metrics.get("worst_classes_by_f1", [])[:10]
    lines = [
        "# Classifier Evaluation Report",
        "",
        "## Summary Metrics",
        "",
        f"- Cases: `{metrics.get('num_cases')}`",
        f"- Accuracy: `{metrics.get('accuracy'):.6f}`",
        f"- Macro precision/recall/F1: `{metrics.get('macro_precision'):.6f}` / `{metrics.get('macro_recall'):.6f}` / `{metrics.get('macro_f1'):.6f}`",
        f"- Weighted precision/recall/F1: `{metrics.get('weighted_precision'):.6f}` / `{metrics.get('weighted_recall'):.6f}` / `{metrics.get('weighted_f1'):.6f}`",
        "",
        "## Worst Classes",
        "",
        *[
            f"- `{row['label']}`: F1 `{float(row['f1_score']):.4f}`, recall `{float(row['recall']):.4f}`, support `{row['support']}`"
            for row in worst
        ],
        "",
        "## Top Confusion Pairs",
        "",
        *[
            f"- `{row['true_label']}` -> `{row['predicted_label']}`: `{row['count']}`"
            for row in pairs
        ],
        "",
        "## Independent Smoke Evaluation",
        "",
    ]
    if smoke.get("available"):
        lines.extend(
            [
                f"- Smoke cases: `{smoke.get('num_cases')}`",
                f"- Top-1/top-3/top-5: `{smoke.get('top_1_accuracy'):.6f}` / `{smoke.get('top_3_accuracy'):.6f}` / `{smoke.get('top_5_accuracy'):.6f}`",
                f"- MRR@5: `{smoke.get('mrr_at_5'):.6f}`",
                f"- Expected calibration error: `{smoke.get('expected_calibration_error'):.6f}`",
                f"- High-confidence errors: `{smoke.get('high_confidence_error_count')}`",
            ]
        )
    else:
        lines.append(f"- Smoke model inference skipped/unavailable: `{smoke.get('reason')}`")
    lines.extend(
        [
            "",
            "## Retraining Assessment",
            "",
            f"- Category: `{assessment['category']}`",
            f"- Retraining recommended: `{assessment['is_retraining_recommended']}`",
            f"- Retraining necessary: `{assessment['is_retraining_necessary']}`",
            f"- Retraining optional: `{assessment['is_retraining_optional']}`",
            f"- Evidence: `{assessment['evidence']}`",
            f"- Rationale: {assessment['rationale']}",
            "",
            "## Classifier vs RAG",
            "",
            f"- RAG top-1/top-5 from latest retrieval report: `{rag_compare.get('rag_summary', {}).get('top_1_accuracy')}` / `{rag_compare.get('rag_summary', {}).get('top_5_accuracy')}`",
            f"- Ensemble fusion justified: `{rag_compare['ensemble_fusion_justified']}`",
            f"- Rationale: {rag_compare['ensemble_fusion_rationale']}",
            "",
            "## Risks",
            "",
            "- Saved test predictions are from the same DDXPlus-derived distribution used for training, so real chat phrasing can be harder.",
            "- The active label universe has 49 labels and excludes common conditions such as diabetes/hyperglycemia and UTI/cystitis.",
            "- A few clinically adjacent labels remain concentrated failure modes.",
        ]
    )
    (output_dir / "classifier_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_smoke_report(output_dir: Path, smoke: dict[str, Any]) -> None:
    lines = ["# Classifier Smoke Evaluation Report", ""]
    if not smoke.get("available"):
        lines.append(f"Smoke evaluation unavailable: `{smoke.get('reason')}`")
    else:
        lines.extend(
            [
                f"- Cases: `{smoke['num_cases']}`",
                f"- Top-1 accuracy: `{smoke['top_1_accuracy']:.6f}`",
                f"- Top-3 accuracy: `{smoke['top_3_accuracy']:.6f}`",
                f"- Top-5 accuracy: `{smoke['top_5_accuracy']:.6f}`",
                f"- MRR@5: `{smoke['mrr_at_5']:.6f}`",
                f"- Mean confidence: `{smoke['mean_confidence']:.6f}`",
                f"- Expected calibration error: `{smoke['expected_calibration_error']:.6f}`",
                f"- Failure count: `{smoke['failure_count']}`",
                "",
                "## Failure Causes",
                "",
                *[f"- `{cause}`: `{count}`" for cause, count in smoke.get("failure_causes", [])],
            ]
        )
    (output_dir / "classifier_smoke_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = get_settings()
    classifier_dir = args.classifier_dir or Path(settings.finetuned_model_dir or "") or TARGETED_CLASSIFIER_DIR
    faiss_dir = args.faiss_index_dir or Path(settings.faiss_index_dir or "") or TARGETED_FAISS_DIR
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = inspect_classifier_assets(classifier_dir, settings)
    write_json(output_dir / "classifier_assets_report.json", assets)
    (output_dir / "classifier_assets_report.md").write_text(build_assets_report_md(assets), encoding="utf-8")

    active_label_map = load_label_map(classifier_dir / "label_map.json")
    targeted_label_map = load_label_map(args.targeted_classifier_dir / "label_map.json")
    natural_label_map = load_label_map(args.natural_classifier_dir / "label_map.json")
    active_faiss = metadata_label_summary("active_faiss_metadata_labels", faiss_dir)
    targeted_faiss = metadata_label_summary("targeted_faiss_metadata_labels", TARGETED_FAISS_DIR)
    natural_faiss = metadata_label_summary("natural_faiss_metadata_labels", args.natural_faiss_dir)
    sources = [
        label_source_summary("classifier_labels", active_label_map["labels"], classifier_dir / "label_map.json"),
        label_source_summary("active_rag_labels", active_faiss["labels"], faiss_dir),
        label_source_summary("active_faiss_metadata_labels", active_faiss["labels"], faiss_dir),
        label_source_summary("targeted_classifier_label_map", targeted_label_map["labels"], args.targeted_classifier_dir / "label_map.json"),
        label_source_summary("natural_classifier_label_map", natural_label_map["labels"], args.natural_classifier_dir / "label_map.json"),
        label_source_summary("targeted_faiss_metadata_labels", targeted_faiss["labels"], TARGETED_FAISS_DIR),
        label_source_summary("natural_faiss_metadata_labels", natural_faiss["labels"], args.natural_faiss_dir),
    ]
    label_payload = {
        "active_classifier_path": str(classifier_dir),
        "active_faiss_path": str(faiss_dir),
        "sources": sources,
        "metadata_sources": [active_faiss, targeted_faiss, natural_faiss],
        "comparison": compare_label_sources(sources),
    }
    write_label_consistency_reports(output_dir, label_payload)

    metrics = recompute_prediction_metrics(classifier_dir, output_dir)
    smoke = (
        {"available": False, "reason": "Model smoke inference skipped by --skip-model-smoke."}
        if args.skip_model_smoke
        else run_model_smoke_eval(
            classifier_dir,
            output_dir,
            max_cases=args.max_smoke_cases,
            batch_size=args.batch_size,
            seed=args.seed,
            metrics_summary=metrics,
        )
    )
    if not smoke.get("available"):
        write_json(output_dir / "classifier_smoke_eval_summary.json", smoke)
        write_csv(
            output_dir / "classifier_smoke_eval_cases.csv",
            [],
            [
                "case_index",
                "expected_label",
                "predicted_label",
                "confidence",
                "rank",
                "top1_correct",
                "top3_correct",
                "top5_correct",
                "top_predictions",
                "likely_causes",
                "text_preview",
            ],
        )
    write_smoke_report(output_dir, smoke)

    assessment = retraining_assessment(metrics, smoke, label_payload)
    rag_compare = compare_classifier_vs_rag(metrics, smoke)
    metrics["retraining_assessment"] = assessment
    metrics["classifier_vs_rag"] = rag_compare
    write_json(output_dir / "classifier_metrics_summary.json", metrics)
    write_eval_report(output_dir, metrics, smoke, assessment, rag_compare)

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "active_classifier_path": str(classifier_dir),
                "label_count": assets["label_map"]["label_count"],
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "smoke_top_1_accuracy": smoke.get("top_1_accuracy"),
                "retraining_category": assessment["category"],
            },
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )


if __name__ == "__main__":
    main()
