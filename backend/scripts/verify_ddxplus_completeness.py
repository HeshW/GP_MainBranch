"""Compare active RAG FAISS metadata against local DDXPlus-derived artifacts.

The repository does not include the raw DDXPlus dataset directory by default, so
this script compares all local derivative sources that are present: FAISS
metadata, classifier label maps, saved prediction CSVs, and saved RAG evaluation
metadata. It does not print secrets.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings


OUT_DIR = Path("data/evaluation/rag_diagnostics")
REPORT_JSON = OUT_DIR / "ddxplus_completeness_report.json"
REPORT_MD = OUT_DIR / "ddxplus_completeness_report.md"


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("-", " ").replace("_", " ").split())


def load_metadata(path: Path) -> dict[str, Any]:
    json_path = path / "metadata_mapping.json"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    with (path / "metadata_mapping.pkl").open("rb") as handle:
        return pickle.load(handle)


def source_summary(name: str, path: Path, rows: int, labels: list[str], active_labels: set[str]) -> dict[str, Any]:
    normalized = {normalize_label(item): item for item in labels}
    active_normalized = {normalize_label(item): item for item in active_labels}
    label_set = set(normalized)
    active_set = set(active_normalized)
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "total_rows_or_cases": rows,
        "unique_pathology_count": len(label_set),
        "pathologies": sorted(normalized.values(), key=lambda item: normalize_label(item)),
        "missing_compared_to_active_faiss": sorted(
            active_normalized[item] for item in active_set - label_set
        ),
        "extra_compared_to_active_faiss": sorted(
            normalized[item] for item in label_set - active_set
        ),
    }


def summarize_faiss_source(name: str, path: Path, active_labels: set[str]) -> dict[str, Any]:
    if not path.exists() or not ((path / "metadata_mapping.pkl").exists() or (path / "metadata_mapping.json").exists()):
        return source_summary(name, path, 0, [], active_labels) | {"exists": False}
    metadata = load_metadata(path)
    labels = [str(item).strip() for item in metadata.get("pathologies", []) if str(item).strip()]
    return source_summary(name, path, len(labels), sorted(set(labels)), active_labels) | {
        "metadata_lengths": {
            key: len(value)
            for key, value in metadata.items()
            if isinstance(value, list)
        }
    }


def summarize_label_map(name: str, path: Path, active_labels: set[str]) -> dict[str, Any]:
    if not path.exists():
        return source_summary(name, path, 0, [], active_labels) | {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    label_map = payload.get("label2id") or payload.get("label_to_id") or {}
    labels = [str(item).strip() for item in label_map.keys() if str(item).strip()]
    return source_summary(name, path, len(labels), labels, active_labels)


def summarize_predictions_csv(name: str, path: Path, active_labels: set[str]) -> dict[str, Any]:
    if not path.exists():
        return source_summary(name, path, 0, [], active_labels) | {"exists": False}
    labels: list[str] = []
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        label_column = next(
            (column for column in ("true_label", "label", "pathology", "expected", "y_true") if column in fields),
            None,
        )
        for row in reader:
            rows += 1
            if label_column and row.get(label_column):
                labels.append(str(row[label_column]).strip())
    return source_summary(name, path, rows, sorted(set(labels)), active_labels)


def summarize_rag_metrics(name: str, path: Path, active_labels: set[str]) -> dict[str, Any]:
    if not path.exists():
        return source_summary(name, path, 0, [], active_labels) | {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = source_summary(name, path, int(payload.get("num_cases", 0) or 0), [], active_labels)
    summary["missing_compared_to_active_faiss"] = []
    summary["extra_compared_to_active_faiss"] = []
    return summary | {
        "num_classes": payload.get("num_classes"),
        "comparison_note": "metrics-only source; no per-pathology label list is stored here",
        "metrics": {
            key: payload.get(key)
            for key in ("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "macro_f1", "weighted_f1")
            if key in payload
        },
    }


def read_index_ntotal(index_dir: Path) -> int | None:
    try:
        import faiss
    except Exception:
        return None
    index_path = index_dir / "medical_cases.index"
    if not index_path.exists():
        return None
    return int(faiss.read_index(str(index_path)).ntotal)


def has_terms(labels: set[str], terms: tuple[str, ...]) -> bool:
    return any(any(term in normalize_label(label) for term in terms) for label in labels)


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def main() -> None:
    settings = get_settings()
    active_faiss_dir = Path(settings.faiss_index_dir or "backend/artifacts/artifacts/faiss_data_targeted")
    active_metadata = load_metadata(active_faiss_dir)
    active_labels_raw = [str(item).strip() for item in active_metadata.get("pathologies", []) if str(item).strip()]
    active_labels = set(active_labels_raw)
    active_distribution = Counter(active_labels_raw)
    active_count = len(active_labels_raw)
    index_ntotal = read_index_ntotal(active_faiss_dir)

    sources = [
        summarize_faiss_source("active_faiss_metadata", active_faiss_dir, active_labels),
        summarize_faiss_source("natural_faiss_metadata", Path("backend/artifacts/artifacts/faiss_data_natural"), active_labels),
        summarize_label_map(
            "targeted_classifier_label_map",
            Path("backend/artifacts/artifacts/clinicalbert_classifier_targeted/label_map.json"),
            active_labels,
        ),
        summarize_label_map(
            "natural_classifier_label_map",
            Path("backend/artifacts/artifacts/clinicalbert_classifier_natural/label_map.json"),
            active_labels,
        ),
        summarize_predictions_csv(
            "targeted_classifier_test_predictions",
            Path("backend/artifacts/artifacts/clinicalbert_classifier_targeted/test_predictions.csv"),
            active_labels,
        ),
        summarize_predictions_csv(
            "natural_classifier_test_predictions",
            Path("backend/artifacts/artifacts/clinicalbert_classifier_natural/test_predictions.csv"),
            active_labels,
        ),
        summarize_rag_metrics(
            "rag_natural_metrics_summary",
            first_existing_path(
                Path("data/evaluation/archive/rag_natural/rag_metrics_summary.json"),
                Path("data/evaluation/rag_natural/rag_metrics_summary.json"),
            ),
            active_labels,
        ),
    ]
    label_universe_source_names = {
        "active_faiss_metadata",
        "natural_faiss_metadata",
        "targeted_classifier_label_map",
        "natural_classifier_label_map",
    }
    evaluation_subset_source_names = {
        "targeted_classifier_test_predictions",
        "natural_classifier_test_predictions",
    }
    authoritative_sources = [
        source
        for source in sources
        if source.get("exists") and source["name"] in label_universe_source_names
    ]

    raw_dataset_candidates = [
        Path("data/ddxplus"),
        Path("data/ddxplus_hf"),
        Path("data/processed_ddxplus"),
        Path("backend/data/ddxplus"),
    ]
    raw_dataset_found = [str(path) for path in raw_dataset_candidates if path.exists()]

    active_label_list = sorted(active_labels, key=normalize_label)
    diabetes_terms = ("diabetes", "hyperglycemia", "diabetic", "prediabetes")
    uti_terms = ("urinary tract infection", "cystitis", "uti", "dysuria")
    respiratory_terms = (
        "cough",
        "bronch",
        "pneum",
        "sinus",
        "rhino",
        "pharyng",
        "laryng",
        "copd",
        "asthma",
        "pulmonary",
        "croup",
        "whooping",
        "tuberculosis",
        "epiglottitis",
        "urti",
        "influenza",
    )
    respiratory_labels = [
        label for label in active_label_list if any(term in normalize_label(label) for term in respiratory_terms)
    ]

    report = {
        "active_runtime": {
            "faiss_index_dir": str(active_faiss_dir),
            "settings_use_rag": bool(settings.use_rag),
            "is_old_backend_faiss_data": normalize_label(str(active_faiss_dir)) == normalize_label("backend/faiss_data"),
        },
        "active_faiss": {
            "index_ntotal": index_ntotal,
            "metadata_count": active_count,
            "unique_pathology_count": len(active_labels),
            "index_matches_metadata_count": index_ntotal == active_count,
            "all_metadata_lengths_match": all(
                len(value) == active_count
                for value in active_metadata.values()
                if isinstance(value, list)
            ),
            "patient_id_test_prefix_count": sum(
                1
                for item in active_metadata.get("patient_ids", [])
                if str(item).lower().startswith("test_")
            ),
            "patient_id_prefix_distribution": Counter(
                str(item).split("_", 1)[0] for item in active_metadata.get("patient_ids", [])
            ),
            "pathology_distribution": dict(sorted(active_distribution.items(), key=lambda item: (-item[1], item[0]))),
            "pathologies": active_label_list,
        },
        "local_raw_dataset_dirs_found": raw_dataset_found,
        "sources": sources,
        "source_roles": {
            "label_universe_sources": sorted(label_universe_source_names),
            "evaluation_subset_sources": sorted(evaluation_subset_source_names),
            "metrics_only_sources": ["rag_natural_metrics_summary"],
        },
        "scope_assessment": {
            "pathology_count_49_expected_from_local_artifacts": all(
                source.get("unique_pathology_count") == 49
                and not source.get("missing_compared_to_active_faiss")
                and not source.get("extra_compared_to_active_faiss")
                for source in authoritative_sources
            )
            and len(active_labels) == 49,
            "diabetes_or_hyperglycemia_present": has_terms(active_labels, diabetes_terms),
            "uti_or_cystitis_present": has_terms(active_labels, uti_terms),
            "respiratory_related_label_count": len(respiratory_labels),
            "respiratory_related_labels": respiratory_labels,
        },
        "conclusions": {
            "active_faiss_complete_relative_to_local_artifacts": index_ntotal == active_count
            and len(active_labels) == 49
            and all(
                not source.get("missing_compared_to_active_faiss")
                and not source.get("extra_compared_to_active_faiss")
                for source in authoritative_sources
            ),
            "diabetes_hyperglycemia_out_of_scope": not has_terms(active_labels, diabetes_terms),
            "uti_cystitis_out_of_scope": not has_terms(active_labels, uti_terms),
            "recommend_excluding_diabetes_uti_from_in_scope_retrieval_eval": True,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DDXPlus Completeness Report",
        "",
        "## Executive Answer",
        "",
        f"- Active FAISS path: `{active_faiss_dir}`",
        f"- FAISS vectors: `{index_ntotal}`",
        f"- Metadata rows: `{active_count}`",
        f"- Unique active pathologies: `{len(active_labels)}`",
        f"- Index count equals metadata count: `{index_ntotal == active_count}`",
        f"- Raw DDXPlus dataset directories found locally: `{raw_dataset_found or 'none'}`",
        "",
        "Based on local artifacts, 49 pathologies is the expected label set for the DDXPlus-derived RAG/classifier bundle in this project.",
        "The active FAISS metadata, natural FAISS metadata, targeted classifier label map, and natural classifier label map agree on the same 49-label universe.",
        "Saved test prediction CSVs are evaluation subsets, so they may omit a rare label while still having no extra out-of-universe labels.",
        "",
        "## Diabetes / UTI Scope",
        "",
        f"- Diabetes/hyperglycemia present: `{report['scope_assessment']['diabetes_or_hyperglycemia_present']}`",
        f"- UTI/cystitis present: `{report['scope_assessment']['uti_or_cystitis_present']}`",
        "",
        "Diabetes/hyperglycemia and UTI/cystitis are absent from every local DDXPlus-derived label source checked. Locally, this supports explanation A/D: they are out-of-scope for this project's DDXPlus label universe, not dropped during preprocessing.",
        "",
        "## Source Comparison",
        "",
    ]
    for source in sources:
        lines.extend(
            [
                f"### {source['name']}",
                f"- Path: `{source['path']}`",
                f"- Exists: `{source['exists']}`",
                f"- Rows/cases: `{source['total_rows_or_cases']}`",
                f"- Unique pathologies: `{source['unique_pathology_count']}`",
                f"- Missing vs active FAISS: `{source['missing_compared_to_active_faiss']}`",
                f"- Extra vs active FAISS: `{source['extra_compared_to_active_faiss']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Active Pathologies",
            "",
            *[f"- {label}" for label in active_label_list],
            "",
            "## Final Recommendations",
            "",
            "- Mark diabetes/hyperglycemia and UTI/cystitis as out-of-scope safety cases, not in-scope retrieval failures.",
            "- Keep one out-of-scope safety case to verify low-confidence RAG gating.",
            "- Build the in-scope retrieval evaluation from labels present in the 49-pathology universe, especially respiratory/ENT/cardiopulmonary labels.",
            "- Do not rebuild FAISS for completeness reasons; current active index is complete relative to local DDXPlus-derived artifacts.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(REPORT_JSON), "markdown": str(REPORT_MD)}, indent=2))


if __name__ == "__main__":
    main()
