from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.diagnosis.rag import MedicalCaseSearcher
from scripts import evaluate_pipeline_end_to_end as e2e
from scripts.check_release_gates import GateThresholds, evaluate_summary


@dataclass
class AblationConfig:
    name: str
    rag_top_k: int
    search_multiplier: int
    search_min: int
    embedding: float
    symptom_overlap: float
    lexical: float
    feature_alignment: float
    mismatch_penalty: float
    pathology_penalty: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG rerank/search-depth ablation frontier.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/evaluation/targeted_cases_v1.json"),
        help="Benchmark cases JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation/ablation"),
        help="Directory for per-run summaries and frontier report.",
    )
    parser.add_argument(
        "--faiss-index-dir",
        type=Path,
        default=Path("backend/artifacts/faiss_data_natural"),
    )
    parser.add_argument(
        "--clinicalbert-model-dir",
        type=Path,
        default=Path("backend/artifacts/bio_clinicalbert"),
    )
    parser.add_argument(
        "--finetuned-model-dir",
        type=Path,
        default=Path("backend/artifacts/clinicalbert_classifier_natural"),
    )
    parser.add_argument(
        "--bundle-id",
        default="clinicalbert_natural_faiss_natural_v1",
    )
    parser.add_argument(
        "--prefix",
        default="track2_phase4_ablation",
        help="Prefix for output run IDs and files.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional limit on number of predefined configs to run (for quick iteration).",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_default_configs() -> list[AblationConfig]:
    return [
        AblationConfig(
            name="baseline_k5",
            rag_top_k=5,
            search_multiplier=80,
            search_min=400,
            embedding=0.55,
            symptom_overlap=0.28,
            lexical=0.15,
            feature_alignment=0.22,
            mismatch_penalty=0.24,
            pathology_penalty=0.28,
        ),
        AblationConfig(
            name="symptom_feature_k5",
            rag_top_k=5,
            search_multiplier=80,
            search_min=400,
            embedding=0.48,
            symptom_overlap=0.32,
            lexical=0.17,
            feature_alignment=0.26,
            mismatch_penalty=0.23,
            pathology_penalty=0.30,
        ),
        AblationConfig(
            name="lexical_guard_k5",
            rag_top_k=5,
            search_multiplier=90,
            search_min=420,
            embedding=0.50,
            symptom_overlap=0.27,
            lexical=0.22,
            feature_alignment=0.25,
            mismatch_penalty=0.24,
            pathology_penalty=0.31,
        ),
        AblationConfig(
            name="deeper_search_k7",
            rag_top_k=7,
            search_multiplier=100,
            search_min=500,
            embedding=0.50,
            symptom_overlap=0.28,
            lexical=0.18,
            feature_alignment=0.24,
            mismatch_penalty=0.23,
            pathology_penalty=0.30,
        ),
    ]


async def run_one_config(
    cfg: AblationConfig,
    *,
    cases: Path,
    output_dir: Path,
    faiss_index_dir: Path,
    clinicalbert_model_dir: Path,
    finetuned_model_dir: Path,
    bundle_id: str,
    prefix: str,
) -> dict[str, Any]:
    MedicalCaseSearcher.configure_rerank_weights(
        embedding=cfg.embedding,
        symptom_overlap=cfg.symptom_overlap,
        lexical=cfg.lexical,
        feature_alignment=cfg.feature_alignment,
        mismatch_penalty=cfg.mismatch_penalty,
        pathology_penalty=cfg.pathology_penalty,
    )
    MedicalCaseSearcher.configure_search_expansion(
        multiplier=cfg.search_multiplier,
        minimum=cfg.search_min,
    )

    run_id = f"{prefix}_{cfg.name}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    summary_path = output_dir / f"{prefix}_{cfg.name}_summary.json"
    summary_payload: dict[str, Any] | None = None

    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "evaluate_pipeline_end_to_end.py",
            "--cases",
            str(cases),
            "--output",
            str(summary_path),
            "--use-rag",
            "--faiss-index-dir",
            str(faiss_index_dir),
            "--clinicalbert-model-dir",
            str(clinicalbert_model_dir),
            "--gemini-api-key",
            "",
            "--use-finetuned-classifier",
            "--finetuned-model-dir",
            str(finetuned_model_dir),
            "--rag-top-k",
            str(cfg.rag_top_k),
            "--run-id",
            run_id,
            "--bundle-id",
            bundle_id,
            "--notes",
            f"phase4_ablation:{cfg.name}",
        ]

        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
            payload = await e2e.main_async()
            if isinstance(payload, dict):
                summary_payload = payload
    finally:
        sys.argv = argv_backup

    if summary_payload is None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = summary_payload
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    thresholds = GateThresholds(
        min_cases=16,
        min_clinical_top_1_accuracy=0.25,
        min_ai_scope_top_1_accuracy=0.2,
        min_post_clarification_clinical_top_1_accuracy=0.7,
        min_clarification_utility_rate=0.25,
        max_low_information_clarification_rate=0.2,
        min_ambiguity_group_top_1_accuracy=0.25,
        ambiguity_group_metric_key="post_clarification_top_1_accuracy",
    )
    gate_report = evaluate_summary(
        summary,
        thresholds,
        required_ambiguity_groups=["atrial_fibrillation_vs_psvt", "pneumonia_vs_bronchospasm"],
    )

    gate_path = output_dir / f"{prefix}_{cfg.name}_gate_report.json"
    gate_path.write_text(json.dumps(gate_report, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "config": cfg.__dict__,
        "summary_path": str(summary_path),
        "gate_report_path": str(gate_path),
        "gate_decision": gate_report.get("decision"),
        "clinical_top_1_accuracy": summary.get("clinical_top_1_accuracy", 0.0),
        "clinical_top_3_accuracy": summary.get("clinical_top_3_accuracy", 0.0),
        "post_clarification_clinical_top_1_accuracy": summary.get("post_clarification_clinical_top_1_accuracy", 0.0),
        "clarification_utility_rate": summary.get("clarification_utility_rate", 0.0),
        "low_information_clarification_rate": summary.get("low_information_clarification_rate", 1.0),
        "average_latency_ms": summary.get("average_latency_ms", 0.0),
        "average_post_clarification_latency_ms": summary.get("average_post_clarification_latency_ms", 0.0),
    }


def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    # Higher is better for quality metrics; lower is better for low-info and latencies.
    gate_score = 1.0 if str(row.get("gate_decision", "")).lower() == "go" else 0.0
    low_info_rate = float(row.get("low_information_clarification_rate") or 1.0)
    post_latency = float(row.get("average_post_clarification_latency_ms") or 1_000_000.0)
    latency = float(row.get("average_latency_ms") or 1_000_000.0)
    return (
        gate_score,
        float(row.get("clinical_top_1_accuracy") or 0.0),
        float(row.get("post_clarification_clinical_top_1_accuracy") or 0.0),
        float(row.get("clarification_utility_rate") or 0.0),
        -low_info_rate,
        -post_latency,
        -latency,
    )


async def main_async() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_default_configs()
    if args.max_configs is not None:
        configs = configs[: max(1, int(args.max_configs))]
    results: list[dict[str, Any]] = []

    for cfg in configs:
        result = await run_one_config(
            cfg,
            cases=args.cases,
            output_dir=args.output_dir,
            faiss_index_dir=args.faiss_index_dir,
            clinicalbert_model_dir=args.clinicalbert_model_dir,
            finetuned_model_dir=args.finetuned_model_dir,
            bundle_id=args.bundle_id,
            prefix=args.prefix,
        )
        results.append(result)

    ranked = sorted(results, key=sort_key, reverse=True)

    report = {
        "generated_at_utc": utc_now_iso(),
        "cases": str(args.cases),
        "bundle_id": args.bundle_id,
        "configs_tested": len(configs),
        "best_config": ranked[0] if ranked else None,
        "results": ranked,
    }

    report_path = args.output_dir / f"{args.prefix}_frontier_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.output_dir / f"{args.prefix}_frontier_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config_name",
                "gate_decision",
                "clinical_top_1_accuracy",
                "clinical_top_3_accuracy",
                "post_clarification_clinical_top_1_accuracy",
                "clarification_utility_rate",
                "low_information_clarification_rate",
                "average_latency_ms",
                "average_post_clarification_latency_ms",
                "summary_path",
                "gate_report_path",
            ],
        )
        writer.writeheader()
        for row in ranked:
            writer.writerow(
                {
                    "config_name": row["config"]["name"],
                    "gate_decision": row["gate_decision"],
                    "clinical_top_1_accuracy": row["clinical_top_1_accuracy"],
                    "clinical_top_3_accuracy": row["clinical_top_3_accuracy"],
                    "post_clarification_clinical_top_1_accuracy": row["post_clarification_clinical_top_1_accuracy"],
                    "clarification_utility_rate": row["clarification_utility_rate"],
                    "low_information_clarification_rate": row["low_information_clarification_rate"],
                    "average_latency_ms": row["average_latency_ms"],
                    "average_post_clarification_latency_ms": row["average_post_clarification_latency_ms"],
                    "summary_path": row["summary_path"],
                    "gate_report_path": row["gate_report_path"],
                }
            )

    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
