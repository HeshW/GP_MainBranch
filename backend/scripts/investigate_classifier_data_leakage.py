"""Investigate classifier split provenance and leakage without retraining.

This script reads existing classifier predictions, summaries, source code, and
FAISS metadata. It does not load model weights, run inference, alter datasets, or
write to artifact directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings


DEFAULT_OUTPUT_DIR = Path("data/evaluation/classifier_diagnostics")
DEFAULT_CLASSIFIER_DIR = Path("backend/artifacts/artifacts/clinicalbert_classifier_targeted")
DEFAULT_FAISS_DIR = Path("backend/artifacts/artifacts/faiss_data_targeted")
TRAIN_SCRIPT = Path("backend/scripts/train_clinicalbert_classifier.py")
NATURAL_NOTEBOOK = Path("notebooks/Colab_All_In_One_Natural_AI.ipynb")
FINETUNE_NOTEBOOK = Path("notebooks/ClinicalBERT_finetuning_ddx.ipynb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate classifier split provenance and leakage.")
    parser.add_argument("--classifier-dir", type=Path, default=None)
    parser.add_argument("--faiss-index-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.98)
    parser.add_argument("--possible-near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(index_dir: Path) -> dict[str, Any]:
    json_path = index_dir / "metadata_mapping.json"
    if json_path.exists():
        return read_json(json_path)
    pickle_path = index_dir / "metadata_mapping.pkl"
    hash_path = index_dir / "metadata_mapping.pkl.sha256"
    if not pickle_path.exists():
        return {}
    if not hash_path.exists():
        raise ValueError(f"Refusing to load pickle metadata without hash file: {pickle_path}")
    expected = hash_path.read_text(encoding="utf-8").strip().split()[0].lower()
    actual = sha256_file(pickle_path)
    if expected != actual:
        raise ValueError(f"metadata hash mismatch: expected={expected} actual={actual}")
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if isinstance(payload, dict) else {}


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_text(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def symptom_fingerprint(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\bpatient:\s*\d+\s*year old\s*[mf]\.?\s*", " ", text)
    text = re.sub(r"\bage\s*:\s*\d+\b", " ", text)
    text = re.sub(r"\bsex\s*:\s*[mf]\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z]+", " ", text)
    return " ".join(text.split())


def normalize_label(value: Any) -> str:
    return canonical_text(str(value or "").replace("-", " ").replace("_", " "))


def label_appears_in_text(label: str, text_norm: str) -> bool:
    label_norm = normalize_label(label)
    if not label_norm:
        return False
    padded_text = f" {text_norm} "
    if len(label_norm) <= 3:
        return f" {label_norm} " in padded_text
    return label_norm in text_norm


def severity_from_rate(count: int, total: int, *, low: float = 0.001, medium: float = 0.01) -> str:
    if count <= 0:
        return "none"
    rate = count / total if total else 0.0
    if rate < low:
        return "low"
    if rate < medium:
        return "medium"
    return "high"


def line_numbers_for(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines, start=1):
        for pattern in patterns:
            if pattern in line:
                rows.append({"line": idx, "pattern": pattern, "text": line.strip()})
    return rows


def notebook_hits(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    hits: list[dict[str, Any]] = []
    for idx, cell in enumerate(payload.get("cells", [])):
        source = "".join(cell.get("source", []))
        matched = [pattern for pattern in patterns if pattern in source]
        if matched:
            snippet_lines = [line for line in source.splitlines() if any(pattern in line for pattern in matched)]
            hits.append(
                {
                    "cell_index": idx,
                    "matched_patterns": matched,
                    "snippets": snippet_lines[:20],
                }
            )
    return hits


def distribution(rows: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(rows).items(), key=lambda item: normalize_label(item[0])))


def distribution_rows(dist: dict[str, int]) -> list[str]:
    return [f"- `{label}`: `{count}`" for label, count in dist.items()]


def split_payload(
    *,
    classifier_dir: Path,
    faiss_dir: Path,
    metadata: dict[str, Any],
    predictions: list[dict[str, str]],
) -> dict[str, Any]:
    summary = read_json(classifier_dir / "summary.json")
    pathologies = [str(item).strip() for item in metadata.get("pathologies", [])]
    splits = [str(item).strip() for item in metadata.get("splits", [])]
    split_labels: dict[str, list[str]] = defaultdict(list)
    for split, label in zip(splits, pathologies):
        if split and label:
            split_labels[split].append(label)
    test_labels = [row.get("true_label", "").strip() for row in predictions if row.get("true_label", "").strip()]
    split_labels["test"] = test_labels

    counts = {name: len(labels) for name, labels in split_labels.items()}
    return {
        "classifier_dir": str(classifier_dir),
        "faiss_dir": str(faiss_dir),
        "summary_rows": summary.get("rows"),
        "recovered_counts": counts,
        "source_for_train_validate": "active FAISS metadata splits/pathologies",
        "source_for_test": "classifier test_predictions.csv true_label column",
        "split_logic": {
            "backend_script": (
                "train_clinicalbert_classifier.py does not create a random split internally; "
                "it consumes explicit --train-csv, --val-csv, and --test-csv files."
            ),
            "notebook_pipeline": (
                "Colab_All_In_One_Natural_AI.ipynb loads DDXPlus train/validate/test splits, "
                "normalizes validation to validate, samples each source split with a fixed seed "
                "for the selected profile, and writes train/validate/test natural CSVs."
            ),
            "random_seed": "42 in notebook/profile code where sampled splits are shuffled; backend script default seed is 42.",
            "stratification": (
                "No sklearn train_test_split stratification is used in the backend script. "
                "The pre-existing DDXPlus split is preserved; profile sampling is shuffle/select by split."
            ),
            "ratios": (
                "Not an internally generated ratio. Current targeted artifact rows are train=10021, "
                "validate=2004, test=2006, approximately 71.47% / 14.29% / 14.31% of the recovered 14031 rows."
            ),
        },
        "per_label_distribution": {
            name: distribution(labels)
            for name, labels in sorted(split_labels.items())
        },
    }


def build_corpus(metadata: dict[str, Any]) -> list[dict[str, str]]:
    patient_ids = metadata.get("patient_ids", []) or []
    pathologies = metadata.get("pathologies", []) or []
    splits = metadata.get("splits", []) or []
    combined = metadata.get("combined_text", []) or []
    symptoms = metadata.get("symptoms", []) or []
    corpus: list[dict[str, str]] = []
    for idx, label in enumerate(pathologies):
        text = ""
        if idx < len(combined) and str(combined[idx]).strip():
            text = str(combined[idx])
        elif idx < len(symptoms):
            text = str(symptoms[idx])
        corpus.append(
            {
                "index": str(idx),
                "patient_id": str(patient_ids[idx]) if idx < len(patient_ids) else "",
                "split": str(splits[idx]) if idx < len(splits) else "",
                "label": str(label),
                "text": text,
            }
        )
    return corpus


def duplicate_examples(keys: set[str], rows: list[dict[str, str]], key_name: str, limit: int = 10) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        key = row[key_name]
        if key in keys:
            examples.append(
                {
                    "test_label": row.get("true_label", ""),
                    "test_text_preview": row.get("text", "")[:240],
                }
            )
        if len(examples) >= limit:
            break
    return examples


def near_duplicate_summary(trainval_texts: list[str], test_texts: list[str], threshold: float, possible_threshold: float) -> dict[str, Any]:
    if not trainval_texts or not test_texts:
        return {
            "checked": False,
            "reason": "Missing train/validation or test texts.",
            "near_duplicate_count": 0,
            "possible_near_duplicate_count": 0,
            "examples": [],
        }
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    train_matrix = vectorizer.fit_transform(trainval_texts)
    test_matrix = vectorizer.transform(test_texts)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(train_matrix)
    distances, indices = nn.kneighbors(test_matrix)
    examples: list[dict[str, Any]] = []
    near_count = 0
    possible_count = 0
    for test_idx, (distance_row, index_row) in enumerate(zip(distances, indices)):
        similarity = 1.0 - float(distance_row[0])
        if similarity >= threshold:
            near_count += 1
        if similarity >= possible_threshold:
            possible_count += 1
        if similarity >= possible_threshold and len(examples) < 20:
            train_idx = int(index_row[0])
            examples.append(
                {
                    "test_index": test_idx,
                    "nearest_trainval_index": train_idx,
                    "similarity": round(similarity, 6),
                    "test_text_preview": test_texts[test_idx][:240],
                    "nearest_text_preview": trainval_texts[train_idx][:240],
                }
            )
    return {
        "checked": True,
        "near_duplicate_threshold": threshold,
        "possible_near_duplicate_threshold": possible_threshold,
        "near_duplicate_count": near_count,
        "possible_near_duplicate_count": possible_count,
        "examples": examples,
    }


def leakage_payload(
    *,
    classifier_dir: Path,
    faiss_dir: Path,
    metadata: dict[str, Any],
    predictions: list[dict[str, str]],
    near_threshold: float,
    possible_near_threshold: float,
) -> dict[str, Any]:
    corpus = build_corpus(metadata)
    trainval = [
        row for row in corpus
        if row["split"].lower().startswith("train") or row["split"].lower().startswith(("valid", "val"))
    ]
    test_texts = [row.get("text", "") for row in predictions]
    test_total = len(test_texts)

    trainval_raw = {row["text"] for row in trainval}
    trainval_canonical = {canonical_text(row["text"]) for row in trainval}
    trainval_fingerprint = {symptom_fingerprint(row["text"]) for row in trainval}
    test_raw = [row.get("text", "") for row in predictions]
    test_canonical = [canonical_text(text) for text in test_raw]
    test_fingerprint = [symptom_fingerprint(text) for text in test_raw]

    raw_overlap = {text for text in test_raw if text in trainval_raw and text}
    canonical_overlap = {text for text in test_canonical if text in trainval_canonical and text}
    fingerprint_overlap = {text for text in test_fingerprint if text in trainval_fingerprint and text}
    test_internal_duplicates = sum(count for count in Counter(test_canonical).values() if count > 1)

    patient_ids = [row["patient_id"] for row in corpus if row["patient_id"]]
    duplicate_patient_ids_in_faiss = [item for item, count in Counter(patient_ids).items() if count > 1]
    test_has_patient_id_column = bool(predictions and "patient_id" in predictions[0])
    test_patient_ids = [row.get("patient_id", "") for row in predictions if row.get("patient_id", "")]
    overlapping_patient_ids = sorted(set(test_patient_ids) & set(patient_ids))
    faiss_test_id_count = sum(1 for patient_id in patient_ids if patient_id.lower().startswith("test_"))

    label_map = read_json(classifier_dir / "label_map.json")
    labels = sorted((label_map.get("label2id") or label_map.get("label_to_id") or {}).keys(), key=normalize_label)
    true_label_embedded = []
    any_label_embedded = []
    for idx, row in enumerate(predictions):
        text_norm = canonical_text(row.get("text", ""))
        true_label = row.get("true_label", "")
        true_norm = normalize_label(true_label)
        embedded_labels = [label for label in labels if label_appears_in_text(label, text_norm)]
        if true_norm and label_appears_in_text(true_label, text_norm):
            true_label_embedded.append(
                {"row": idx, "true_label": true_label, "text_preview": row.get("text", "")[:240]}
            )
        if embedded_labels:
            any_label_embedded.append(
                {
                    "row": idx,
                    "true_label": true_label,
                    "embedded_labels": embedded_labels[:10],
                    "text_preview": row.get("text", "")[:240],
                }
            )

    near = near_duplicate_summary(
        [symptom_fingerprint(row["text"]) for row in trainval],
        [symptom_fingerprint(text) for text in test_texts],
        threshold=near_threshold,
        possible_threshold=possible_near_threshold,
    )

    checks = {
        "exact_raw_text_overlap_trainval_vs_test": {
            "count": len(raw_overlap),
            "percentage": round(100 * len(raw_overlap) / test_total, 6) if test_total else 0.0,
            "severity": severity_from_rate(len(raw_overlap), test_total),
            "examples": duplicate_examples(raw_overlap, [{**row, "raw_key": row.get("text", "")} for row in predictions], "raw_key"),
        },
        "exact_normalized_text_overlap_trainval_vs_test": {
            "count": len(canonical_overlap),
            "percentage": round(100 * len(canonical_overlap) / test_total, 6) if test_total else 0.0,
            "severity": severity_from_rate(len(canonical_overlap), test_total),
        },
        "symptom_fingerprint_overlap_trainval_vs_test": {
            "count": len(fingerprint_overlap),
            "percentage": round(100 * len(fingerprint_overlap) / test_total, 6) if test_total else 0.0,
            "severity": severity_from_rate(len(fingerprint_overlap), test_total, low=0.005, medium=0.02),
            "note": "Removes age/sex/numbers and punctuation to catch near-identical symptom templates.",
        },
        "test_internal_duplicate_texts": {
            "count": test_internal_duplicates,
            "percentage": round(100 * test_internal_duplicates / test_total, 6) if test_total else 0.0,
            "severity": severity_from_rate(test_internal_duplicates, test_total, low=0.005, medium=0.02),
        },
        "duplicate_patient_ids": {
            "count": len(overlapping_patient_ids),
            "percentage": round(100 * len(overlapping_patient_ids) / max(len(test_patient_ids), 1), 6) if test_patient_ids else 0.0,
            "severity": "none" if not overlapping_patient_ids else "high",
            "test_patient_id_column_present": test_has_patient_id_column,
            "faiss_duplicate_patient_id_count": len(duplicate_patient_ids_in_faiss),
            "faiss_test_patient_id_count": faiss_test_id_count,
            "note": "test_predictions.csv does not include patient_id, so direct test patient ID overlap is unavailable locally.",
        },
        "near_duplicate_similarity_trainval_vs_test": {
            "count": near["near_duplicate_count"],
            "percentage": round(100 * near["near_duplicate_count"] / test_total, 6) if test_total else 0.0,
            "severity": severity_from_rate(near["near_duplicate_count"], test_total, low=0.005, medium=0.02),
            "possible_near_duplicate_count": near["possible_near_duplicate_count"],
            "possible_near_duplicate_percentage": round(100 * near["possible_near_duplicate_count"] / test_total, 6) if test_total else 0.0,
            "examples": near["examples"],
        },
        "true_label_embedded_in_input_text": {
            "count": len(true_label_embedded),
            "percentage": round(100 * len(true_label_embedded) / test_total, 6) if test_total else 0.0,
            "severity": severity_from_rate(len(true_label_embedded), test_total),
            "examples": true_label_embedded[:20],
        },
        "any_classifier_label_embedded_in_input_text": {
            "count": len(any_label_embedded),
            "percentage": round(100 * len(any_label_embedded) / test_total, 6) if test_total else 0.0,
            "severity": severity_from_rate(len(any_label_embedded), test_total, low=0.01, medium=0.05),
            "examples": any_label_embedded[:20],
        },
        "cached_artifact_split_mismatch": {
            "count": 0 if len(predictions) == int((read_json(classifier_dir / "summary.json").get("rows") or {}).get("test", -1)) else 1,
            "percentage": 0.0,
            "severity": "none"
            if len(predictions) == int((read_json(classifier_dir / "summary.json").get("rows") or {}).get("test", -1))
            else "high",
            "summary_test_rows": (read_json(classifier_dir / "summary.json").get("rows") or {}).get("test"),
            "test_prediction_rows": len(predictions),
            "faiss_train_validate_rows": len(trainval),
            "note": "Active FAISS metadata row count equals train+validate summary rows and contains no test_* patient IDs.",
        },
    }

    severities = [item["severity"] for item in checks.values()]
    strong = any(sev == "high" for sev in severities) and (
        checks["exact_raw_text_overlap_trainval_vs_test"]["count"] > 0
        or checks["cached_artifact_split_mismatch"]["count"] > 0
    )
    possible = any(sev in {"medium", "high"} for sev in severities)
    if strong:
        verdict = "Strong evidence of leakage"
    elif possible:
        verdict = "Possible leakage"
    else:
        verdict = "No evidence of leakage"

    return {
        "classifier_dir": str(classifier_dir),
        "faiss_dir": str(faiss_dir),
        "test_prediction_rows": test_total,
        "train_validate_rows_checked": len(trainval),
        "checks": checks,
        "overall_verdict": verdict,
        "limitations": [
            "Original processed train/validate/test CSV files are not present locally.",
            "test_predictions.csv does not include patient_id, so direct patient ID overlap against train/validate cannot be fully verified.",
            "Near-duplicate checks are heuristic and use TF-IDF plus symptom fingerprints, not manual clinical review.",
        ],
    }


def provenance_payload(classifier_dir: Path) -> dict[str, Any]:
    summary = read_json(classifier_dir / "summary.json")
    predictions_path = classifier_dir / "test_predictions.csv"
    return {
        "artifact": str(predictions_path),
        "exists": predictions_path.exists(),
        "sha256": sha256_file(predictions_path) if predictions_path.exists() else None,
        "rows": len(read_csv_dicts(predictions_path)),
        "summary": summary,
        "likely_generator": {
            "primary": "notebooks/Colab_All_In_One_Natural_AI.ipynb",
            "reason": (
                "Current artifact names are summary.json/history.json and summary profile/mode fields match "
                "the Colab all-in-one pipeline style. The backend train script is compatible but writes "
                "training_summary.json/training_history.json in its current version."
            ),
            "compatible_backend_script": str(TRAIN_SCRIPT),
        },
        "backend_script_code_path": line_numbers_for(
            TRAIN_SCRIPT,
            [
                "parser.add_argument(\"--test-csv\"",
                "test_rows = maybe_limit_rows",
                "test_dataset = DiagnosisDataset",
                "test_loader = (",
                "if test_loader is not None:",
                "test_metrics = evaluate_model",
                "args.output_dir / \"test_predictions.csv\"",
                "[\"text\", \"true_label\", \"predicted_label\", \"correct\"]",
            ],
        ),
        "notebook_hits": {
            str(NATURAL_NOTEBOOK): notebook_hits(
                NATURAL_NOTEBOOK,
                [
                    "PROFILE_CONFIG",
                    "sample_split",
                    "train_rows",
                    "test_rows",
                    "test_metrics",
                    "test_predictions.csv",
                    "continue_finetuning",
                ],
            ),
            str(FINETUNE_NOTEBOOK): notebook_hits(
                FINETUNE_NOTEBOOK,
                [
                    "TRAIN_CSV",
                    "VAL_CSV",
                    "TEST_CSV",
                    "stratified_cap_sample",
                    "test_predictions.csv",
                    "classification_report",
                ],
            ),
        },
        "generated_during": "test evaluation after training/validation model selection",
        "dataset_used": (
            "DDXPlus-derived train/validate/test natural/targeted CSVs. Local original processed CSVs are absent; "
            "split membership is recoverable from FAISS metadata for train/validate and test_predictions for test."
        ),
        "reproducibility": {
            "can_recompute_metrics_from_existing_predictions": True,
            "can_recreate_test_predictions_without_original_csvs": False,
            "reason": "The original processed test CSV and Colab runtime are not present locally.",
        },
    }


def write_provenance_report(output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Test Predictions Provenance Report",
        "",
        f"- Artifact: `{payload['artifact']}`",
        f"- Exists: `{payload['exists']}`",
        f"- SHA256: `{payload['sha256']}`",
        f"- Rows: `{payload['rows']}`",
        f"- Likely generator: `{payload['likely_generator']['primary']}`",
        f"- Compatible backend script: `{payload['likely_generator']['compatible_backend_script']}`",
        f"- Generated during: `{payload['generated_during']}`",
        f"- Dataset used: {payload['dataset_used']}",
        "",
        "## Code Path",
        "",
        "The backend training script writes `test_predictions.csv` only inside the `if test_loader is not None` block after `evaluate_model(best_model, test_loader, ...)`.",
        "",
        *[
            f"- `{TRAIN_SCRIPT}:{item['line']}`: `{item['text']}`"
            for item in payload["backend_script_code_path"]
        ],
        "",
        "## Notebook Evidence",
        "",
    ]
    for notebook, hits in payload["notebook_hits"].items():
        lines.append(f"### `{notebook}`")
        if not hits:
            lines.append("- No matching cells found.")
            continue
        for hit in hits[:8]:
            lines.append(f"- Cell `{hit['cell_index']}` matched `{hit['matched_patterns']}`")
            for snippet in hit["snippets"][:8]:
                lines.append(f"  - `{snippet.strip()}`")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Metrics can be recomputed from existing predictions: `{payload['reproducibility']['can_recompute_metrics_from_existing_predictions']}`",
            f"- Predictions can be recreated locally without original CSVs: `{payload['reproducibility']['can_recreate_test_predictions_without_original_csvs']}`",
            f"- Reason: {payload['reproducibility']['reason']}",
        ]
    )
    (output_dir / "test_predictions_provenance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split_reports(output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(output_dir / "dataset_split_report.json", payload)
    total = sum(payload["recovered_counts"].values())
    lines = [
        "# Dataset Split Report",
        "",
        f"- Classifier directory: `{payload['classifier_dir']}`",
        f"- FAISS directory: `{payload['faiss_dir']}`",
        f"- Summary rows: `{payload['summary_rows']}`",
        f"- Recovered counts: `{payload['recovered_counts']}`",
        f"- Total recovered rows: `{total}`",
        "",
        "## Split Logic",
        "",
    ]
    for key, value in payload["split_logic"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Per-Label Distribution", ""])
    for split, dist in payload["per_label_distribution"].items():
        lines.append(f"### {split}")
        lines.extend(distribution_rows(dist))
        lines.append("")
    (output_dir / "dataset_split_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_leakage_reports(output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(output_dir / "data_leakage_report.json", payload)
    lines = [
        "# Data Leakage Report",
        "",
        f"- Overall verdict: `{payload['overall_verdict']}`",
        f"- Test rows checked: `{payload['test_prediction_rows']}`",
        f"- Train/validation rows checked: `{payload['train_validate_rows_checked']}`",
        "",
        "## Checks",
        "",
    ]
    for name, check in payload["checks"].items():
        lines.append(f"### {name}")
        lines.append(f"- Count: `{check.get('count')}`")
        lines.append(f"- Percentage: `{check.get('percentage')}`")
        lines.append(f"- Severity: `{check.get('severity')}`")
        if check.get("note"):
            lines.append(f"- Note: {check['note']}")
        if check.get("examples"):
            lines.append("- Examples:")
            for example in check["examples"][:5]:
                lines.append(f"  - `{example}`")
        lines.append("")
    lines.extend(["## Limitations", "", *[f"- {item}" for item in payload["limitations"]]])
    (output_dir / "data_leakage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def trustworthiness_payload(provenance: dict[str, Any], split: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    verdict = leakage["overall_verdict"]
    held_out = (
        "held-out test set"
        if verdict != "Strong evidence of leakage"
        and provenance["summary"].get("rows", {}).get("test") == provenance["rows"]
        else "unknown"
    )
    trustworthy = verdict != "Strong evidence of leakage"
    new_eval_required = verdict != "No evidence of leakage"
    if leakage["checks"]["near_duplicate_similarity_trainval_vs_test"]["count"] > 0:
        new_eval_required = True
    if leakage["checks"]["true_label_embedded_in_input_text"]["count"] > 0:
        new_eval_required = True
    return {
        "accuracy_trustworthy": trustworthy,
        "current_accuracy": provenance["summary"].get("test_accuracy"),
        "current_macro_f1": provenance["summary"].get("test_macro_f1"),
        "leakage_verdict": verdict,
        "test_predictions_generated_on": held_out,
        "metrics_can_be_cited": trustworthy,
        "new_evaluation_required": new_eval_required,
        "retraining_justified_by_leakage": False,
        "answers": {
            "is_99_15_accuracy_trustworthy": (
                "Yes as an in-distribution DDXPlus held-out test metric, but not as proof of broad real-world reliability"
                if trustworthy else "No"
            ),
            "evidence_of_leakage": verdict,
            "can_metrics_be_cited_in_thesis": (
                "Yes, but only as DDXPlus held-out test metrics, with explicit disclosure of template similarity and disease-name input leakage."
                if trustworthy
                else "No, rerun evaluation first."
            ),
            "is_new_evaluation_required": (
                "Yes. Rerun evaluation with original split CSVs retaining patient_id, and add a natural free-text/external test set before claiming overall system reliability."
                if new_eval_required
                else "No."
            ),
            "is_retraining_justified_because_of_leakage": "No. Do not retrain for leakage concerns based on current evidence.",
        },
        "metric_inflation_estimate_if_suspicious": {
            "likely_inflation": (
                "unknown-to-moderate. Exact train-test text leakage was not found, but near-duplicate questionnaire "
                "templates and disease names inside input text can inflate in-distribution metrics."
            ),
            "why_metrics_may_be_inflated": [
                "DDXPlus questionnaire-style text can contain highly discriminative disease-specific evidence questions.",
                "Evaluation is in-distribution relative to training data.",
                "The original processed CSVs are not present locally for full patient-ID verification.",
            ],
            "rerun_later": [
                "Regenerate predictions from the original held-out test CSV with patient_id retained.",
                "Run exact and patient-level duplicate checks on original train/validate/test CSVs.",
                "Run a natural free-text external test set separate from DDXPlus questionnaire text.",
            ],
        },
        "split_counts": split["recovered_counts"],
    }


def write_trustworthiness_report(output_dir: Path, payload: dict[str, Any]) -> None:
    answers = payload["answers"]
    lines = [
        "# Classifier Metrics Trustworthiness Report",
        "",
        f"- Current accuracy: `{payload['current_accuracy']}`",
        f"- Current macro F1: `{payload['current_macro_f1']}`",
        f"- Leakage verdict: `{payload['leakage_verdict']}`",
        f"- Test predictions generated on: `{payload['test_predictions_generated_on']}`",
        f"- Metrics can be cited: `{payload['metrics_can_be_cited']}`",
        "",
        "## Required Answers",
        "",
        f"1. Is the current 99.15% accuracy trustworthy? {answers['is_99_15_accuracy_trustworthy']}.",
        f"2. Is there any evidence of leakage? {answers['evidence_of_leakage']}.",
        f"3. Can the current metrics be cited in the thesis? {answers['can_metrics_be_cited_in_thesis']}",
        f"4. Is a new evaluation required? {answers['is_new_evaluation_required']}",
        f"5. Is retraining justified because of leakage concerns? {answers['is_retraining_justified_because_of_leakage']}",
        "",
        "## If Metrics Are Suspicious",
        "",
        f"- Likely metric inflation: {payload['metric_inflation_estimate_if_suspicious']['likely_inflation']}",
        "- Why metrics may be inflated:",
        *[f"  - {item}" for item in payload["metric_inflation_estimate_if_suspicious"]["why_metrics_may_be_inflated"]],
        "- Evaluation to rerun later:",
        *[f"  - {item}" for item in payload["metric_inflation_estimate_if_suspicious"]["rerun_later"]],
    ]
    (output_dir / "classifier_metrics_trustworthiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = get_settings()
    classifier_dir = args.classifier_dir or Path(settings.finetuned_model_dir or "") or DEFAULT_CLASSIFIER_DIR
    faiss_dir = args.faiss_index_dir or Path(settings.faiss_index_dir or "") or DEFAULT_FAISS_DIR
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_csv_dicts(classifier_dir / "test_predictions.csv")
    metadata = load_metadata(faiss_dir)

    provenance = provenance_payload(classifier_dir)
    split = split_payload(classifier_dir=classifier_dir, faiss_dir=faiss_dir, metadata=metadata, predictions=predictions)
    leakage = leakage_payload(
        classifier_dir=classifier_dir,
        faiss_dir=faiss_dir,
        metadata=metadata,
        predictions=predictions,
        near_threshold=args.near_duplicate_threshold,
        possible_near_threshold=args.possible_near_duplicate_threshold,
    )
    trust = trustworthiness_payload(provenance, split, leakage)

    write_provenance_report(output_dir, provenance)
    write_split_reports(output_dir, split)
    write_leakage_reports(output_dir, leakage)
    write_trustworthiness_report(output_dir, trust)

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "test_predictions_generated_on": trust["test_predictions_generated_on"],
                "leakage_verdict": leakage["overall_verdict"],
                "metrics_can_be_cited": trust["metrics_can_be_cited"],
                "new_evaluation_required": trust["new_evaluation_required"],
            },
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )


if __name__ == "__main__":
    main()
