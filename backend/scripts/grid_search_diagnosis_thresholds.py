from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.diagnosis.diagnosisengine import DiagnosisEngine
from scripts import evaluate_pipeline_end_to_end as e2e
from scripts.check_release_gates import GateThresholds, evaluate_summary


@dataclass(frozen=True)
class ThresholdConfig:
    name: str
    classifier_primary_threshold: float
    classifier_support_threshold: float
    rule_gating_ai_confidence_threshold: float
    clarification_confidence_threshold: float
    clarification_margin_threshold: float
    classifier_override_margin: float
    clarification_override_margin: float
    clarification_override_gain_threshold: float
    clarification_leader_margin: float


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for token in str(raw or "").split(","):
        item = token.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("At least one float value must be provided.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnosis threshold grid search.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/evaluation/targeted_cases_v1.json"),
        help="Benchmark cases JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation/threshold_grid"),
        help="Directory for per-config summaries and grid report.",
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
    parser.add_argument("--rag-top-k", type=int, default=7)
    parser.add_argument("--bundle-id", default="clinicalbert_natural_faiss_natural_v1")
    parser.add_argument(
        "--prefix",
        default="track2_phase3_threshold_grid",
        help="Prefix for output run IDs and files.",
    )
    parser.add_argument(
        "--classifier-primary-values",
        default="0.52,0.55",
        help="Comma-separated values for CLASSIFIER_PRIMARY_THRESHOLD.",
    )
    parser.add_argument(
        "--classifier-support-values",
        default="0.32,0.35",
        help="Comma-separated values for CLASSIFIER_SUPPORT_THRESHOLD.",
    )
    parser.add_argument(
        "--rule-gating-values",
        default="0.80",
        help="Comma-separated values for RULE_GATING_AI_CONFIDENCE_THRESHOLD.",
    )
    parser.add_argument(
        "--clarification-confidence-values",
        default="0.70,0.72",
        help="Comma-separated values for CLARIFICATION_CONFIDENCE_THRESHOLD.",
    )
    parser.add_argument(
        "--clarification-margin-values",
        default="0.10,0.12",
        help="Comma-separated values for CLARIFICATION_MARGIN_THRESHOLD.",
    )
    parser.add_argument(
        "--classifier-override-values",
        default="0.08",
        help="Comma-separated values for CLASSIFIER_OVERRIDE_MARGIN.",
    )
    parser.add_argument(
        "--clarification-override-margin-values",
        default="0.05",
        help="Comma-separated values for CLARIFICATION_OVERRIDE_MARGIN.",
    )
    parser.add_argument(
        "--clarification-override-gain-values",
        default="0.12",
        help="Comma-separated values for CLARIFICATION_OVERRIDE_GAIN_THRESHOLD.",
    )
    parser.add_argument(
        "--clarification-leader-margin-values",
        default="0.05",
        help="Comma-separated values for CLARIFICATION_LEADER_MARGIN.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the current in-code baseline thresholds.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on total configs (after baseline deduplication).",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def snapshot_current_thresholds() -> dict[str, float]:
    return {
        "classifier_primary_threshold": float(DiagnosisEngine.CLASSIFIER_PRIMARY_THRESHOLD),
        "classifier_support_threshold": float(DiagnosisEngine.CLASSIFIER_SUPPORT_THRESHOLD),
        "rule_gating_ai_confidence_threshold": float(DiagnosisEngine.RULE_GATING_AI_CONFIDENCE_THRESHOLD),
        "clarification_confidence_threshold": float(DiagnosisEngine.CLARIFICATION_CONFIDENCE_THRESHOLD),
        "clarification_margin_threshold": float(DiagnosisEngine.CLARIFICATION_MARGIN_THRESHOLD),
        "classifier_override_margin": float(DiagnosisEngine.CLASSIFIER_OVERRIDE_MARGIN),
        "clarification_override_margin": float(DiagnosisEngine.CLARIFICATION_OVERRIDE_MARGIN),
        "clarification_override_gain_threshold": float(DiagnosisEngine.CLARIFICATION_OVERRIDE_GAIN_THRESHOLD),
        "clarification_leader_margin": float(DiagnosisEngine.CLARIFICATION_LEADER_MARGIN),
    }


def build_config(name: str, values: dict[str, float]) -> ThresholdConfig:
    return ThresholdConfig(
        name=name,
        classifier_primary_threshold=float(values["classifier_primary_threshold"]),
        classifier_support_threshold=float(values["classifier_support_threshold"]),
        rule_gating_ai_confidence_threshold=float(values["rule_gating_ai_confidence_threshold"]),
        clarification_confidence_threshold=float(values["clarification_confidence_threshold"]),
        clarification_margin_threshold=float(values["clarification_margin_threshold"]),
        classifier_override_margin=float(values["classifier_override_margin"]),
        clarification_override_margin=float(values["clarification_override_margin"]),
        clarification_override_gain_threshold=float(values["clarification_override_gain_threshold"]),
        clarification_leader_margin=float(values["clarification_leader_margin"]),
    )


def config_key(cfg: ThresholdConfig) -> tuple[float, ...]:
    return (
        float(cfg.classifier_primary_threshold),
        float(cfg.classifier_support_threshold),
        float(cfg.rule_gating_ai_confidence_threshold),
        float(cfg.clarification_confidence_threshold),
        float(cfg.clarification_margin_threshold),
        float(cfg.classifier_override_margin),
        float(cfg.clarification_override_margin),
        float(cfg.clarification_override_gain_threshold),
        float(cfg.clarification_leader_margin),
    )


def build_grid_configs(args: argparse.Namespace) -> list[ThresholdConfig]:
    baseline_values = snapshot_current_thresholds()
    configs: list[ThresholdConfig] = []
    seen: set[tuple[float, ...]] = set()

    if not args.skip_baseline:
        baseline = build_config("baseline_current", baseline_values)
        configs.append(baseline)
        seen.add(config_key(baseline))

    classifier_primary_values = parse_float_list(args.classifier_primary_values)
    classifier_support_values = parse_float_list(args.classifier_support_values)
    rule_gating_values = parse_float_list(args.rule_gating_values)
    clarification_confidence_values = parse_float_list(args.clarification_confidence_values)
    clarification_margin_values = parse_float_list(args.clarification_margin_values)
    classifier_override_values = parse_float_list(args.classifier_override_values)
    clarification_override_margin_values = parse_float_list(args.clarification_override_margin_values)
    clarification_override_gain_values = parse_float_list(args.clarification_override_gain_values)
    clarification_leader_margin_values = parse_float_list(args.clarification_leader_margin_values)

    index = 0
    for values in itertools.product(
        classifier_primary_values,
        classifier_support_values,
        rule_gating_values,
        clarification_confidence_values,
        clarification_margin_values,
        classifier_override_values,
        clarification_override_margin_values,
        clarification_override_gain_values,
        clarification_leader_margin_values,
    ):
        index += 1
        config = ThresholdConfig(
            name=f"cfg_{index:03d}",
            classifier_primary_threshold=values[0],
            classifier_support_threshold=values[1],
            rule_gating_ai_confidence_threshold=values[2],
            clarification_confidence_threshold=values[3],
            clarification_margin_threshold=values[4],
            classifier_override_margin=values[5],
            clarification_override_margin=values[6],
            clarification_override_gain_threshold=values[7],
            clarification_leader_margin=values[8],
        )
        key = config_key(config)
        if key in seen:
            continue
        seen.add(key)
        configs.append(config)

    if args.max_configs is not None:
        configs = configs[: max(1, int(args.max_configs))]

    return configs


def apply_threshold_config(cfg: ThresholdConfig) -> None:
    DiagnosisEngine.CLASSIFIER_PRIMARY_THRESHOLD = float(cfg.classifier_primary_threshold)
    DiagnosisEngine.CLASSIFIER_SUPPORT_THRESHOLD = float(cfg.classifier_support_threshold)
    DiagnosisEngine.RULE_GATING_AI_CONFIDENCE_THRESHOLD = float(cfg.rule_gating_ai_confidence_threshold)
    DiagnosisEngine.CLARIFICATION_CONFIDENCE_THRESHOLD = float(cfg.clarification_confidence_threshold)
    DiagnosisEngine.CLARIFICATION_MARGIN_THRESHOLD = float(cfg.clarification_margin_threshold)
    DiagnosisEngine.CLASSIFIER_OVERRIDE_MARGIN = float(cfg.classifier_override_margin)
    DiagnosisEngine.CLARIFICATION_OVERRIDE_MARGIN = float(cfg.clarification_override_margin)
    DiagnosisEngine.CLARIFICATION_OVERRIDE_GAIN_THRESHOLD = float(cfg.clarification_override_gain_threshold)
    DiagnosisEngine.CLARIFICATION_LEADER_MARGIN = float(cfg.clarification_leader_margin)


def restore_thresholds(snapshot: dict[str, float]) -> None:
    apply_threshold_config(build_config("restore", snapshot))


async def run_one_config(
    cfg: ThresholdConfig,
    *,
    cases: Path,
    output_dir: Path,
    faiss_index_dir: Path,
    clinicalbert_model_dir: Path,
    finetuned_model_dir: Path,
    bundle_id: str,
    prefix: str,
    rag_top_k: int,
) -> dict[str, Any]:
    apply_threshold_config(cfg)

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
            str(rag_top_k),
            "--run-id",
            run_id,
            "--bundle-id",
            bundle_id,
            "--notes",
            f"phase3_threshold_grid:{cfg.name}",
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
        "config": asdict(cfg),
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


def add_delta_metrics(row: dict[str, Any], baseline_row: dict[str, Any] | None) -> dict[str, Any]:
    if baseline_row is None:
        row["delta_clinical_top_1_accuracy_vs_baseline"] = None
        row["delta_post_clarification_clinical_top_1_accuracy_vs_baseline"] = None
        row["delta_clarification_utility_rate_vs_baseline"] = None
        row["delta_low_information_clarification_rate_vs_baseline"] = None
        row["delta_average_latency_ms_vs_baseline"] = None
        row["delta_average_post_clarification_latency_ms_vs_baseline"] = None
        return row

    row["delta_clinical_top_1_accuracy_vs_baseline"] = (
        float(row.get("clinical_top_1_accuracy") or 0.0)
        - float(baseline_row.get("clinical_top_1_accuracy") or 0.0)
    )
    row["delta_post_clarification_clinical_top_1_accuracy_vs_baseline"] = (
        float(row.get("post_clarification_clinical_top_1_accuracy") or 0.0)
        - float(baseline_row.get("post_clarification_clinical_top_1_accuracy") or 0.0)
    )
    row["delta_clarification_utility_rate_vs_baseline"] = (
        float(row.get("clarification_utility_rate") or 0.0)
        - float(baseline_row.get("clarification_utility_rate") or 0.0)
    )
    row["delta_low_information_clarification_rate_vs_baseline"] = (
        float(row.get("low_information_clarification_rate") or 0.0)
        - float(baseline_row.get("low_information_clarification_rate") or 0.0)
    )
    row["delta_average_latency_ms_vs_baseline"] = (
        float(row.get("average_latency_ms") or 0.0)
        - float(baseline_row.get("average_latency_ms") or 0.0)
    )
    row["delta_average_post_clarification_latency_ms_vs_baseline"] = (
        float(row.get("average_post_clarification_latency_ms") or 0.0)
        - float(baseline_row.get("average_post_clarification_latency_ms") or 0.0)
    )
    return row


def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
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

    snapshot = snapshot_current_thresholds()
    configs = build_grid_configs(args)
    results: list[dict[str, Any]] = []

    try:
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
                rag_top_k=args.rag_top_k,
            )
            results.append(result)
    finally:
        restore_thresholds(snapshot)

    baseline_row = next((row for row in results if row.get("config", {}).get("name") == "baseline_current"), None)
    enriched = [add_delta_metrics(dict(row), baseline_row) for row in results]
    ranked = sorted(enriched, key=sort_key, reverse=True)

    report = {
        "generated_at_utc": utc_now_iso(),
        "cases": str(args.cases),
        "bundle_id": args.bundle_id,
        "configs_tested": len(configs),
        "baseline_config": baseline_row,
        "best_config": ranked[0] if ranked else None,
        "results": ranked,
    }

    report_path = args.output_dir / f"{args.prefix}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.output_dir / f"{args.prefix}_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config_name",
                "gate_decision",
                "classifier_primary_threshold",
                "classifier_support_threshold",
                "rule_gating_ai_confidence_threshold",
                "clarification_confidence_threshold",
                "clarification_margin_threshold",
                "classifier_override_margin",
                "clarification_override_margin",
                "clarification_override_gain_threshold",
                "clarification_leader_margin",
                "clinical_top_1_accuracy",
                "clinical_top_3_accuracy",
                "post_clarification_clinical_top_1_accuracy",
                "clarification_utility_rate",
                "low_information_clarification_rate",
                "average_latency_ms",
                "average_post_clarification_latency_ms",
                "delta_clinical_top_1_accuracy_vs_baseline",
                "delta_post_clarification_clinical_top_1_accuracy_vs_baseline",
                "delta_clarification_utility_rate_vs_baseline",
                "delta_low_information_clarification_rate_vs_baseline",
                "delta_average_latency_ms_vs_baseline",
                "delta_average_post_clarification_latency_ms_vs_baseline",
                "summary_path",
                "gate_report_path",
            ],
        )
        writer.writeheader()
        for row in ranked:
            cfg = row["config"]
            writer.writerow(
                {
                    "config_name": cfg["name"],
                    "gate_decision": row["gate_decision"],
                    "classifier_primary_threshold": cfg["classifier_primary_threshold"],
                    "classifier_support_threshold": cfg["classifier_support_threshold"],
                    "rule_gating_ai_confidence_threshold": cfg["rule_gating_ai_confidence_threshold"],
                    "clarification_confidence_threshold": cfg["clarification_confidence_threshold"],
                    "clarification_margin_threshold": cfg["clarification_margin_threshold"],
                    "classifier_override_margin": cfg["classifier_override_margin"],
                    "clarification_override_margin": cfg["clarification_override_margin"],
                    "clarification_override_gain_threshold": cfg["clarification_override_gain_threshold"],
                    "clarification_leader_margin": cfg["clarification_leader_margin"],
                    "clinical_top_1_accuracy": row["clinical_top_1_accuracy"],
                    "clinical_top_3_accuracy": row["clinical_top_3_accuracy"],
                    "post_clarification_clinical_top_1_accuracy": row["post_clarification_clinical_top_1_accuracy"],
                    "clarification_utility_rate": row["clarification_utility_rate"],
                    "low_information_clarification_rate": row["low_information_clarification_rate"],
                    "average_latency_ms": row["average_latency_ms"],
                    "average_post_clarification_latency_ms": row["average_post_clarification_latency_ms"],
                    "delta_clinical_top_1_accuracy_vs_baseline": row["delta_clinical_top_1_accuracy_vs_baseline"],
                    "delta_post_clarification_clinical_top_1_accuracy_vs_baseline": row[
                        "delta_post_clarification_clinical_top_1_accuracy_vs_baseline"
                    ],
                    "delta_clarification_utility_rate_vs_baseline": row["delta_clarification_utility_rate_vs_baseline"],
                    "delta_low_information_clarification_rate_vs_baseline": row[
                        "delta_low_information_clarification_rate_vs_baseline"
                    ],
                    "delta_average_latency_ms_vs_baseline": row["delta_average_latency_ms_vs_baseline"],
                    "delta_average_post_clarification_latency_ms_vs_baseline": row[
                        "delta_average_post_clarification_latency_ms_vs_baseline"
                    ],
                    "summary_path": row["summary_path"],
                    "gate_report_path": row["gate_report_path"],
                }
            )

    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
