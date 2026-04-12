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
- clarification-aware one-shot vs post-clarification metrics when follow-up answers are provided
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
from datetime import datetime, timezone
import json
import pickle
import re
import subprocess
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

if hasattr(sys.stdout, "reconfigure"):
    # Avoid cp1252 crashes when benchmark details contain non-ASCII labels/text.
    sys.stdout.reconfigure(encoding="utf-8")

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
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier for reproducible benchmark tracking.",
    )
    parser.add_argument(
        "--bundle-id",
        default=None,
        help="Optional promoted artifact bundle identifier.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-text notes stored in summary metadata.",
    )
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


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_git_commit(repo_root: Path) -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    commit = output.strip()
    return commit or None


def file_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def load_json_if_exists(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if not path or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def summarize_by_ambiguity_group(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in details:
        group = str(item.get("ambiguity_group") or "").strip()
        if not group:
            continue
        bucket = grouped.setdefault(
            group,
            {
                "num_cases": 0,
                "top_1_correct_count": 0,
                "top_3_correct_count": 0,
                "clarification_applied_count": 0,
                "clarification_top_1_correct_count": 0,
                "clarification_improved_count": 0,
                "average_clarification_questions": 0.0,
                "_question_sum": 0,
            },
        )
        bucket["num_cases"] += 1
        bucket["top_1_correct_count"] += int(item.get("top_1_correct", False))
        bucket["top_3_correct_count"] += int(item.get("top_3_correct", False))
        if item.get("clarification_applied"):
            bucket["clarification_applied_count"] += 1
            bucket["clarification_top_1_correct_count"] += int(item.get("clarification_top_1_correct", False))
            bucket["clarification_improved_count"] += int(
                (not item.get("top_1_correct", False)) and item.get("clarification_top_1_correct", False)
            )
            bucket["_question_sum"] += int(item.get("clarification_question_count") or 0)

    for bucket in grouped.values():
        total = max(bucket["num_cases"], 1)
        clar_total = bucket["clarification_applied_count"]
        bucket["top_1_accuracy"] = bucket.pop("top_1_correct_count") / total
        bucket["top_3_accuracy"] = bucket.pop("top_3_correct_count") / total
        if clar_total:
            bucket["post_clarification_top_1_accuracy"] = bucket.pop("clarification_top_1_correct_count") / clar_total
            bucket["clarification_improved_rate"] = bucket.pop("clarification_improved_count") / clar_total
            bucket["average_clarification_questions"] = bucket.pop("_question_sum") / clar_total
        else:
            # If clarification was not needed for a group, use one-shot top-1 as the effective post-clar quality.
            bucket["post_clarification_top_1_accuracy"] = bucket["top_1_accuracy"]
            bucket["clarification_improved_rate"] = 0.0
            bucket["average_clarification_questions"] = 0.0
            bucket.pop("clarification_top_1_correct_count")
            bucket.pop("clarification_improved_count")
            bucket.pop("_question_sum")

    return grouped


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
    clarification = diagnosis.get("clarification", {}) or {}
    clarification_needed = bool(clarification.get("needed"))
    clarification_questions = clarification.get("questions", []) or []
    clarification_question_count = len(clarification_questions)
    clarification_total_targets = 0
    clarification_multi_target_question_count = 0
    for question in clarification_questions:
        targets = [
            str(item).strip()
            for item in question.get("target_conditions", []) or []
            if str(item).strip()
        ]
        clarification_total_targets += len(targets)
        if len(targets) >= 2:
            clarification_multi_target_question_count += 1
    clarification_avg_targets_per_question = (
        clarification_total_targets / clarification_question_count
        if clarification_question_count else 0.0
    )
    final_diagnosis = diagnosis.get("final_diagnosis", {}) or {}
    primary_source = str(final_diagnosis.get("source", "")).strip() or "none"
    rule_covered = (
        primary_source.startswith("rules")
        or any(str(item.get("source", "")).endswith("rules") for item in diagnosis.get("findings", []) or [])
    )

    follow_up_answers = [
        str(item).strip() for item in case.get("follow_up_answers", []) if str(item).strip()
    ]
    clarification_result: Optional[dict[str, Any]] = None
    clarification_top_1_prediction = ""
    clarification_top_3_predictions: list[str] = []
    clarification_top_1_correct = False
    clarification_top_3_correct = False
    clarification_top_1_clinical_match = False
    clarification_top_3_clinical_match = False
    clarification_applied = False

    if clarification_needed and follow_up_answers and result.get("report"):
        clarification_result = await manager.run_clarification(
            result["report"],
            follow_up_answers,
            prior_diagnosis=result.get("diagnosis", {}),
        )
        clarification_applied = True
        clarification_top_1_prediction = extract_final_diagnosis_label(clarification_result)
        clarification_top_3_predictions = extract_top_k_predictions(clarification_result, k=3)
        clarification_top_1_correct = clarification_top_1_prediction == primary_expected
        clarification_top_3_correct = primary_expected in clarification_top_3_predictions
        clarification_top_1_clinical_match = clinically_matches(primary_expected, clarification_top_1_prediction)
        clarification_top_3_clinical_match = any_clinical_match(primary_expected, clarification_top_3_predictions)

    return {
        "case_id": case.get("id", f"case_{index:03d}"),
        "ambiguity_group": str(case.get("ambiguity_group", "")).strip(),
        "language": str(case.get("language", "")).strip(),
        "difficulty": str(case.get("difficulty", "")).strip(),
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
        "clarification_needed": clarification_needed,
        "clarification_candidate_diseases": [
            str(item.get("label", "")).strip()
            for item in clarification.get("candidate_diseases", []) or []
            if str(item.get("label", "")).strip()
        ],
        "follow_up_answers_provided": bool(follow_up_answers),
        "clarification_applied": clarification_applied,
        "clarification_question_count": clarification_question_count,
        "clarification_multi_target_question_count": clarification_multi_target_question_count,
        "clarification_total_targets": clarification_total_targets,
        "clarification_avg_targets_per_question": clarification_avg_targets_per_question,
        "clarification_top_1_prediction": clarification_top_1_prediction,
        "clarification_top_3_predictions": clarification_top_3_predictions,
        "clarification_top_1_correct": clarification_top_1_correct,
        "clarification_top_3_correct": clarification_top_3_correct,
        "clarification_top_1_clinical_match": clarification_top_1_clinical_match,
        "clarification_top_3_clinical_match": clarification_top_3_clinical_match,
        "therapy_present": therapy_present,
        "safety_present": safety_present,
        "fusion_present": fusion_present,
        "latency_ms": float(result.get("elapsed_ms", 0.0) or 0.0),
        "clarification_latency_ms": float((clarification_result or {}).get("elapsed_ms", 0.0) or 0.0),
        "result": result,
        "clarification_result": clarification_result,
    }


async def main_async() -> dict[str, Any]:
    args = parse_args()
    run_id = str(args.run_id).strip() if args.run_id else default_run_id()
    bundle_id = str(args.bundle_id).strip() if args.bundle_id else run_id
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
    clarification_needed_details = [item for item in details if item["clarification_needed"]]
    clarification_applied_details = [item for item in details if item["clarification_applied"]]
    clarification_total_questions = sum(int(item.get("clarification_question_count") or 0) for item in clarification_needed_details)
    clarification_total_targets = sum(int(item.get("clarification_total_targets") or 0) for item in clarification_needed_details)
    clarification_cases_with_multi_target_question = sum(
        int((item.get("clarification_multi_target_question_count") or 0) > 0)
        for item in clarification_needed_details
    )
    clarification_improved_count = sum(
        int((not item["top_1_correct"]) and item["clarification_top_1_correct"])
        for item in clarification_applied_details
    )
    clarification_changed_top_1_count = sum(
        int(bool(item["clarification_top_1_prediction"]) and item["clarification_top_1_prediction"] != item["top_1_prediction"])
        for item in clarification_applied_details
    )
    low_information_clarification_count = sum(
        int(
            item["clarification_top_1_prediction"] == item["top_1_prediction"]
            and item["clarification_top_3_predictions"] == item["top_3_predictions"]
        )
        for item in clarification_applied_details
    )

    classifier_summary_path = (finetuned_model_dir / "summary.json") if finetuned_model_dir else None
    artifact_metadata = {
        "bundle_id": bundle_id,
        "faiss_index": file_metadata(args.faiss_index_dir / "medical_cases.index"),
        "faiss_mapping": file_metadata(args.faiss_index_dir / "metadata_mapping.pkl"),
        "faiss_index_info": load_json_if_exists(args.faiss_index_dir / "index_info.json"),
        "classifier_model_dir": str(finetuned_model_dir) if finetuned_model_dir else None,
        "classifier_summary": load_json_if_exists(classifier_summary_path),
    }

    total = len(details)
    summary = {
        "run_metadata": {
            "run_id": run_id,
            "bundle_id": bundle_id,
            "evaluated_at_utc": utc_now_iso(),
            "script": "evaluate_pipeline_end_to_end.py",
            "git_commit": get_git_commit(ROOT.parent),
            "notes": str(args.notes or "").strip(),
        },
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
        "clarification_rate": (
            len(clarification_needed_details) / total if total else 0.0
        ),
        "follow_up_answers_available_rate": (
            sum(int(item["follow_up_answers_provided"]) for item in details) / total if total else 0.0
        ),
        "clarification_applied_rate": (
            len(clarification_applied_details) / total if total else 0.0
        ),
        "average_clarification_questions": (
            sum(item["clarification_question_count"] for item in clarification_needed_details) / len(clarification_needed_details)
            if clarification_needed_details else 0.0
        ),
        "average_latency_ms": (
            sum(item["latency_ms"] for item in details) / total if total else 0.0
        ),
        "average_post_clarification_latency_ms": (
            sum(item["clarification_latency_ms"] for item in clarification_applied_details) / len(clarification_applied_details)
            if clarification_applied_details else 0.0
        ),
        "post_clarification_top_1_accuracy": (
            sum(int(item["clarification_top_1_correct"]) for item in clarification_applied_details) / len(clarification_applied_details)
            if clarification_applied_details else 0.0
        ),
        "post_clarification_top_3_accuracy": (
            sum(int(item["clarification_top_3_correct"]) for item in clarification_applied_details) / len(clarification_applied_details)
            if clarification_applied_details else 0.0
        ),
        "post_clarification_clinical_top_1_accuracy": (
            sum(int(item["clarification_top_1_clinical_match"]) for item in clarification_applied_details) / len(clarification_applied_details)
            if clarification_applied_details else 0.0
        ),
        "post_clarification_clinical_top_3_accuracy": (
            sum(int(item["clarification_top_3_clinical_match"]) for item in clarification_applied_details) / len(clarification_applied_details)
            if clarification_applied_details else 0.0
        ),
        "clarification_accuracy_gain_top_1": (
            (
                sum(int(item["clarification_top_1_correct"]) for item in clarification_applied_details)
                - sum(int(item["top_1_correct"]) for item in clarification_applied_details)
            ) / len(clarification_applied_details)
            if clarification_applied_details else 0.0
        ),
        "clarification_multi_target_question_rate": (
            clarification_cases_with_multi_target_question / len(clarification_needed_details)
            if clarification_needed_details else 0.0
        ),
        "average_target_conditions_per_clarification_question": (
            clarification_total_targets / clarification_total_questions
            if clarification_total_questions else 0.0
        ),
        "clarification_utility_rate": (
            clarification_improved_count / len(clarification_applied_details)
            if clarification_applied_details else 0.0
        ),
        "clarification_changed_top_1_rate": (
            clarification_changed_top_1_count / len(clarification_applied_details)
            if clarification_applied_details else 0.0
        ),
        "low_information_clarification_rate": (
            low_information_clarification_count / len(clarification_applied_details)
            if clarification_applied_details else 0.0
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
        "artifact_metadata": artifact_metadata,
        "ambiguity_group_metrics": summarize_by_ambiguity_group(details),
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
                "ambiguity_group": item.get("ambiguity_group", ""),
                "language": item.get("language", ""),
                "difficulty": item.get("difficulty", ""),
                "mode": item["mode"],
                "expected_condition": item["primary_expected_condition"],
                "top_1_prediction": item["top_1_prediction"],
                "top_3_predictions": " | ".join(item["top_3_predictions"]),
                "top_1_correct": item["top_1_correct"],
                "top_3_correct": item["top_3_correct"],
                "diagnosis_hit": item["diagnosis_hit"],
                "clarification_needed": item["clarification_needed"],
                "clarification_applied": item["clarification_applied"],
                "clarification_multi_target_question_count": item["clarification_multi_target_question_count"],
                "clarification_avg_targets_per_question": round(item["clarification_avg_targets_per_question"], 3),
                "clarification_top_1_prediction": item["clarification_top_1_prediction"],
                "clarification_top_3_predictions": " | ".join(item["clarification_top_3_predictions"]),
                "clarification_top_1_correct": item["clarification_top_1_correct"],
                "clarification_top_3_correct": item["clarification_top_3_correct"],
                "clarification_improved_top_1": (
                    (not item["top_1_correct"]) and item["clarification_top_1_correct"]
                ),
            }
            for item in details
        ],
        [
            "case_id",
            "ambiguity_group",
            "language",
            "difficulty",
            "mode",
            "expected_condition",
            "top_1_prediction",
            "top_3_predictions",
            "top_1_correct",
            "top_3_correct",
            "diagnosis_hit",
            "clarification_needed",
            "clarification_applied",
            "clarification_multi_target_question_count",
            "clarification_avg_targets_per_question",
            "clarification_top_1_prediction",
            "clarification_top_3_predictions",
            "clarification_top_1_correct",
            "clarification_top_3_correct",
            "clarification_improved_top_1",
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
    return summary


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
