"""Evaluate the diagnosis pipeline end-to-end from JSON cases.

Supported case types
--------------------
1. Labs-only:
   {"labs": {"glucose": 130}, "expected_conditions": ["Diabetes Mellitus (suspected)"]}

2. OCR-text simulation:
   {"ocr_text": "Glucose: 130 mg/dL", "expected_conditions": ["Diabetes Mellitus (suspected)"]}

This script exports:
- top-1 final diagnosis accuracy
- top-3 diagnosis accuracy
- macro precision / recall / F1
- legacy diagnosis_hit_rate
- therapy / safety / fusion metadata rates
- per-case predictions
- confusion matrix
- classification report
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from manager.chat_manager import ChatManager
from manager.symptom_parser import parse_symptoms

FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "anemia": ("anemia",),
    "glycemic_disorder": ("diabetes", "hyperglycemia", "prediabetes"),
    "gerd": ("gerd", "gastroesophageal reflux", "reflux", "heartburn"),
    "cardiopulmonary_red_flag": ("cardiopulmonary", "chest pain", "shortness of breath", "dyspnea"),
    "respiratory_infection": ("pneumonia", "bronchitis", "urti", "viral pharyngitis", "lower respiratory infection"),
}


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate the AI pipeline end-to-end.")
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="JSON file containing evaluation cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/pipeline_end_to_end_summary.json"),
        help="Where to write the summary JSON report.",
    )
    parser.add_argument("--use-rag", action="store_true", default=settings.use_rag)
    parser.add_argument(
        "--faiss-index-dir",
        type=Path,
        default=Path(settings.faiss_index_dir) if settings.faiss_index_dir else Path("backend/faiss_data"),
    )
    parser.add_argument(
        "--clinicalbert-model-dir",
        type=Path,
        default=Path(settings.clinicalbert_model_dir) if settings.clinicalbert_model_dir else Path("backend/artifacts/bio_clinicalbert"),
    )
    parser.add_argument(
        "--gemini-api-key",
        default=settings.gemini_api_key,
    )
    parser.add_argument("--rag-top-k", type=int, default=settings.rag_top_k)
    parser.add_argument("--rag-translate-arabic", action="store_true", default=settings.rag_translate_arabic)
    parser.add_argument("--use-finetuned-classifier", action="store_true", default=settings.use_finetuned_classifier)
    parser.add_argument(
        "--finetuned-model-dir",
        type=Path,
        default=(
            Path(settings.finetuned_model_dir)
            if settings.finetuned_model_dir
            else Path("backend/artifacts/bio_clinicalbert")
        ),
    )
    parser.add_argument("--classifier-max-length", type=int, default=settings.classifier_max_length)
    parser.add_argument("--classifier-translate-arabic", action="store_true", default=settings.classifier_translate_arabic)
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cases file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("Cases file must contain a JSON list.")
    return payload


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def label_families(value: str) -> set[str]:
    normalized = normalize_label(value)
    families: set[str] = set()
    for family, keywords in FAMILY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            families.add(family)
    return families


def clinically_matches(expected_label: str, predicted_label: str) -> bool:
    expected = normalize_label(expected_label)
    predicted = normalize_label(predicted_label)
    if not expected or not predicted:
        return False
    if expected == predicted:
        return True
    if len(expected) >= 5 and expected in predicted:
        return True
    if len(predicted) >= 5 and predicted in expected:
        return True
    return bool(label_families(expected_label) & label_families(predicted_label))


def any_clinical_match(expected_label: str, predicted_labels: list[str]) -> bool:
    return any(clinically_matches(expected_label, label) for label in predicted_labels if str(label).strip())


def load_supported_labels(*paths: Path) -> set[str]:
    supported: set[str] = set()
    for path in paths:
        if not path or not path.exists():
            continue
        if path.name == "label_map.json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            supported.update(str(label).strip() for label in (payload.get("label2id") or {}).keys() if str(label).strip())
            continue
        if path.name == "metadata_mapping.pkl":
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            items = payload.values() if isinstance(payload, dict) else payload
            for item in items:
                if isinstance(item, dict):
                    label = item.get("pathology") or item.get("label")
                else:
                    label = None
                if str(label or "").strip():
                    supported.add(str(label).strip())
    return supported


def is_in_supported_scope(label: str, supported_labels: set[str]) -> bool:
    normalized_supported = {normalize_label(item) for item in supported_labels if str(item).strip()}
    return normalize_label(label) in normalized_supported


def extract_conditions(result: dict[str, Any]) -> list[str]:
    findings = result.get("diagnosis", {}).get("findings", [])
    return [str(item.get("condition", "")).strip() for item in findings if str(item.get("condition", "")).strip()]


def extract_final_diagnosis_label(result: dict[str, Any]) -> str:
    diagnosis = result.get("diagnosis", {})
    final_diagnosis = diagnosis.get("final_diagnosis", {}) or {}
    label = str(final_diagnosis.get("diagnosis", "")).strip()
    if label:
        return label

    conditions = extract_conditions(result)
    return conditions[0] if conditions else "NO_DIAGNOSIS"


def extract_top_k_predictions(result: dict[str, Any], k: int = 3) -> list[str]:
    diagnosis = result.get("diagnosis", {})
    ranked: list[str] = []

    final_label = str((diagnosis.get("final_diagnosis", {}) or {}).get("diagnosis", "")).strip()
    if final_label:
        ranked.append(final_label)

    classifier_prediction = diagnosis.get("classifier_prediction", {}) or {}
    for item in classifier_prediction.get("top_predictions", []) or []:
        label = str(item.get("label", "")).strip()
        if label and label not in ranked:
            ranked.append(label)

    for label in extract_conditions(result):
        if label not in ranked:
            ranked.append(label)

    return ranked[:k]


def top_k_accuracy(y_true: list[str], top_k_predictions: list[list[str]], k: int) -> float:
    if not y_true:
        return 0.0
    hits = 0
    for true_label, predicted_labels in zip(y_true, top_k_predictions):
        if true_label in predicted_labels[:k]:
            hits += 1
    return hits / len(y_true)


def build_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_idx[true_label]][label_to_idx[pred_label]] += 1
    return matrix


def build_classification_report(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
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


def write_confusion_matrix_csv(output_path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def write_rows_csv(output_path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_output_paths(summary_path: Path) -> dict[str, Path]:
    stem = summary_path.stem
    if stem.endswith("_summary"):
        prefix = stem[: -len("_summary")]
    else:
        prefix = stem
    return {
        "summary_json": summary_path,
        "predictions_csv": summary_path.with_name(f"{prefix}_predictions.csv"),
        "classification_report_csv": summary_path.with_name(f"{prefix}_classification_report.csv"),
        "confusion_matrix_csv": summary_path.with_name(f"{prefix}_confusion_matrix.csv"),
    }


async def evaluate_case(
    manager: ChatManager,
    case: dict[str, Any],
    index: int,
    *,
    supported_labels: set[str],
) -> dict[str, Any]:
    if "raw_text" in case:
        result = await manager.run_from_symptoms(case["raw_text"])
        mode = "raw_text"
    elif "ocr_text" in case:
        parsed = parse_symptoms(case["ocr_text"])
        result = await manager.run_pipeline(manual_input={"labs": parsed.get("labs", {})})
        mode = "ocr_text"
    else:
        result = await manager.run_pipeline(labs=case.get("labs", {}))
        mode = "labs"

    predicted_conditions = extract_conditions(result)
    expected_conditions = [str(item).strip() for item in case.get("expected_conditions", []) if str(item).strip()]
    diagnosis_hit = all(item in predicted_conditions for item in expected_conditions)

    top_1_prediction = extract_final_diagnosis_label(result)
    top_3_predictions = extract_top_k_predictions(result, k=3)
    primary_expected = expected_conditions[0] if expected_conditions else "NO_EXPECTATION"
    top_1_clinical_match = clinically_matches(primary_expected, top_1_prediction)
    top_3_clinical_match = any_clinical_match(primary_expected, top_3_predictions)
    in_ai_scope = is_in_supported_scope(primary_expected, supported_labels)

    therapy = result.get("therapy", {})
    therapy_present = bool(therapy.get("therapy_plan"))
    safety_present = "safety" in result.get("diagnosis", {})
    fusion_present = "decision_fusion" in result.get("diagnosis", {})
    diagnosis = result.get("diagnosis", {})
    final_diagnosis = diagnosis.get("final_diagnosis", {}) or {}
    primary_source = str(final_diagnosis.get("source", "")).strip() or "none"
    rule_covered = (
        primary_source.startswith("rules")
        or any(str(item.get("source", "")).endswith("rules") for item in diagnosis.get("findings", []) or [])
    )

    return {
        "case_id": case.get("id", f"case_{index:03d}"),
        "mode": mode,
        "expected_conditions": expected_conditions,
        "primary_expected_condition": primary_expected,
        "predicted_conditions": predicted_conditions,
        "top_1_prediction": top_1_prediction,
        "top_3_predictions": top_3_predictions,
        "top_1_correct": top_1_prediction == primary_expected,
        "top_3_correct": primary_expected in top_3_predictions,
        "top_1_clinical_match": top_1_clinical_match,
        "top_3_clinical_match": top_3_clinical_match,
        "diagnosis_hit": diagnosis_hit,
        "diagnosis_hit_clinical": any_clinical_match(primary_expected, predicted_conditions),
        "in_ai_scope": in_ai_scope,
        "primary_source": primary_source,
        "rule_covered": rule_covered,
        "therapy_present": therapy_present,
        "safety_present": safety_present,
        "fusion_present": fusion_present,
        "result": result,
    }


async def main_async() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    output_paths = build_output_paths(args.output)
    finetuned_model_dir: Optional[Path] = args.finetuned_model_dir
    if finetuned_model_dir and not (finetuned_model_dir / "label_map.json").exists():
        bio_clinicalbert_dir = Path("backend/artifacts/bio_clinicalbert")
        if (bio_clinicalbert_dir / "label_map.json").exists():
            finetuned_model_dir = bio_clinicalbert_dir
    supported_labels = load_supported_labels(
        (finetuned_model_dir / "label_map.json") if finetuned_model_dir else Path(),
        args.faiss_index_dir / "metadata_mapping.pkl",
    )

    manager = ChatManager(
        use_rag=args.use_rag,
        faiss_index_dir=args.faiss_index_dir,
        clinicalbert_model_dir=args.clinicalbert_model_dir,
        gemini_api_key=args.gemini_api_key,
        rag_top_k=args.rag_top_k,
        rag_translate_arabic=args.rag_translate_arabic,
        use_finetuned_classifier=args.use_finetuned_classifier,
        finetuned_model_dir=finetuned_model_dir,
        classifier_max_length=args.classifier_max_length,
        classifier_translate_arabic=args.classifier_translate_arabic,
    )

    details: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        details.append(await evaluate_case(manager, case, index, supported_labels=supported_labels))

    y_true = [item["primary_expected_condition"] for item in details]
    y_pred = [item["top_1_prediction"] for item in details]
    top_3_predictions = [item["top_3_predictions"] for item in details]
    labels = sorted(set(y_true) | set(y_pred))
    confusion_matrix = build_confusion_matrix(y_true, y_pred, labels)
    report_rows, averages = build_classification_report(y_true, y_pred, labels)
    in_scope_details = [item for item in details if item["in_ai_scope"]]
    out_of_scope_details = [item for item in details if not item["in_ai_scope"]]
    source_distribution = dict(Counter(item["primary_source"] for item in details))

    total = len(details)
    summary = {
        "num_cases": total,
        "top_1_accuracy": (sum(int(item["top_1_correct"]) for item in details) / total if total else 0.0),
        "top_3_accuracy": top_k_accuracy(y_true, top_3_predictions, k=3),
        "clinical_top_1_accuracy": (
            sum(int(item["top_1_clinical_match"]) for item in details) / total if total else 0.0
        ),
        "clinical_top_3_accuracy": (
            sum(int(item["top_3_clinical_match"]) for item in details) / total if total else 0.0
        ),
        "macro_precision": averages["macro_precision"],
        "macro_recall": averages["macro_recall"],
        "macro_f1": averages["macro_f1"],
        "weighted_precision": averages["weighted_precision"],
        "weighted_recall": averages["weighted_recall"],
        "weighted_f1": averages["weighted_f1"],
        "diagnosis_hit_rate": (
            sum(int(item["diagnosis_hit"]) for item in details) / total if total else 0.0
        ),
        "diagnosis_hit_rate_clinical": (
            sum(int(item["diagnosis_hit_clinical"]) for item in details) / total if total else 0.0
        ),
        "ai_scope_num_cases": len(in_scope_details),
        "ai_scope_top_1_accuracy": (
            sum(int(item["top_1_correct"]) for item in in_scope_details) / len(in_scope_details)
            if in_scope_details else 0.0
        ),
        "ai_scope_top_3_accuracy": (
            sum(int(item["top_3_correct"]) for item in in_scope_details) / len(in_scope_details)
            if in_scope_details else 0.0
        ),
        "out_of_scope_num_cases": len(out_of_scope_details),
        "out_of_scope_rule_coverage_rate": (
            sum(int(item["rule_covered"]) for item in out_of_scope_details) / len(out_of_scope_details)
            if out_of_scope_details else 0.0
        ),
        "rules_primary_rate": (
            sum(int(str(item["primary_source"]).startswith("rules")) for item in details) / total if total else 0.0
        ),
        "primary_source_distribution": source_distribution,
        "therapy_presence_rate": (
            sum(int(item["therapy_present"]) for item in details) / total if total else 0.0
        ),
        "safety_metadata_rate": (
            sum(int(item["safety_present"]) for item in details) / total if total else 0.0
        ),
        "fusion_metadata_rate": (
            sum(int(item["fusion_present"]) for item in details) / total if total else 0.0
        ),
        "cases_file": str(args.cases),
        "runtime_config": {
            "use_rag": args.use_rag,
            "faiss_index_dir": str(args.faiss_index_dir),
            "clinicalbert_model_dir": str(args.clinicalbert_model_dir),
            "use_finetuned_classifier": args.use_finetuned_classifier,
            "finetuned_model_dir": str(finetuned_model_dir) if finetuned_model_dir else None,
            "rag_top_k": args.rag_top_k,
            "gemini_key_present": bool(args.gemini_api_key),
            "supported_label_space_size": len(supported_labels),
        },
        "outputs": {
            "predictions_csv": str(output_paths["predictions_csv"]),
            "classification_report_csv": str(output_paths["classification_report_csv"]),
            "confusion_matrix_csv": str(output_paths["confusion_matrix_csv"]),
            "summary_json": str(output_paths["summary_json"]),
        },
        "details": details,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with output_paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    write_rows_csv(
        output_paths["predictions_csv"],
        [
            {
                "case_id": item["case_id"],
                "mode": item["mode"],
                "expected_condition": item["primary_expected_condition"],
                "top_1_prediction": item["top_1_prediction"],
                "top_3_predictions": " | ".join(item["top_3_predictions"]),
                "top_1_correct": item["top_1_correct"],
                "top_3_correct": item["top_3_correct"],
                "diagnosis_hit": item["diagnosis_hit"],
            }
            for item in details
        ],
        [
            "case_id",
            "mode",
            "expected_condition",
            "top_1_prediction",
            "top_3_predictions",
            "top_1_correct",
            "top_3_correct",
            "diagnosis_hit",
        ],
    )
    write_rows_csv(
        output_paths["classification_report_csv"],
        report_rows,
        ["label", "precision", "recall", "f1_score", "support"],
    )
    write_confusion_matrix_csv(
        output_paths["confusion_matrix_csv"],
        labels,
        confusion_matrix,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
