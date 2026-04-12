from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class GateThresholds:
    min_cases: int = 100
    min_clinical_top_1_accuracy: float = 0.60
    min_ai_scope_top_1_accuracy: float = 0.58
    min_post_clarification_clinical_top_1_accuracy: float = 0.72
    min_clarification_utility_rate: float = 0.20
    max_low_information_clarification_rate: float = 0.45
    min_ambiguity_group_top_1_accuracy: float = 0.45
    min_clinical_top_1_delta_vs_baseline: float = 0.0
    min_post_clarification_top_1_delta_vs_baseline: float = 0.0
    ambiguity_group_metric_key: str = "top_1_accuracy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate benchmark summary against release gates.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to evaluation summary JSON.")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="Optional baseline summary JSON for regression gates.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON report path.")
    parser.add_argument(
        "--required-ambiguity-group",
        action="append",
        default=[],
        help="Ambiguity group that must exist and pass threshold. Can be repeated.",
    )
    parser.add_argument("--min-cases", type=int, default=100)
    parser.add_argument("--min-clinical-top1", type=float, default=0.60)
    parser.add_argument("--min-ai-scope-top1", type=float, default=0.58)
    parser.add_argument("--min-post-clarification-clinical-top1", type=float, default=0.72)
    parser.add_argument("--min-clarification-utility-rate", type=float, default=0.20)
    parser.add_argument("--max-low-information-clarification-rate", type=float, default=0.45)
    parser.add_argument("--min-ambiguity-group-top1", type=float, default=0.45)
    parser.add_argument(
        "--ambiguity-group-metric-key",
        type=str,
        default="top_1_accuracy",
        choices=["top_1_accuracy", "post_clarification_top_1_accuracy", "clarification_improved_rate"],
        help="Metric key to use for required ambiguity-group gates.",
    )
    parser.add_argument("--min-clinical-top1-delta-vs-baseline", type=float, default=0.0)
    parser.add_argument("--min-post-clarification-top1-delta-vs-baseline", type=float, default=0.0)
    return parser.parse_args()


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _evaluate_min_gate(name: str, actual: Optional[float], threshold: float, *, message: str) -> dict[str, Any]:
    passed = actual is not None and actual >= threshold
    return {
        "name": name,
        "passed": passed,
        "actual": actual,
        "threshold": threshold,
        "operator": ">=",
        "message": message,
    }


def _evaluate_max_gate(name: str, actual: Optional[float], threshold: float, *, message: str) -> dict[str, Any]:
    passed = actual is not None and actual <= threshold
    return {
        "name": name,
        "passed": passed,
        "actual": actual,
        "threshold": threshold,
        "operator": "<=",
        "message": message,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def evaluate_summary(
    summary: dict[str, Any],
    thresholds: GateThresholds,
    *,
    baseline_summary: Optional[dict[str, Any]] = None,
    required_ambiguity_groups: Optional[list[str]] = None,
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []

    num_cases = int(summary.get("num_cases") or 0)
    gates.append(
        _evaluate_min_gate(
            "num_cases",
            float(num_cases),
            float(thresholds.min_cases),
            message="Benchmark must include enough cases for stable decision making.",
        )
    )

    gates.append(
        _evaluate_min_gate(
            "clinical_top_1_accuracy",
            _as_float(summary.get("clinical_top_1_accuracy")),
            thresholds.min_clinical_top_1_accuracy,
            message="Strict in-scope clinical Top-1 must meet minimum quality gate.",
        )
    )
    gates.append(
        _evaluate_min_gate(
            "ai_scope_top_1_accuracy",
            _as_float(summary.get("ai_scope_top_1_accuracy")),
            thresholds.min_ai_scope_top_1_accuracy,
            message="AI-supported scope Top-1 must stay above release threshold.",
        )
    )
    gates.append(
        _evaluate_min_gate(
            "post_clarification_clinical_top_1_accuracy",
            _as_float(summary.get("post_clarification_clinical_top_1_accuracy")),
            thresholds.min_post_clarification_clinical_top_1_accuracy,
            message="Clarified strict Top-1 must demonstrate useful follow-up value.",
        )
    )
    gates.append(
        _evaluate_min_gate(
            "clarification_utility_rate",
            _as_float(summary.get("clarification_utility_rate")),
            thresholds.min_clarification_utility_rate,
            message="Follow-up must improve Top-1 often enough to justify asking questions.",
        )
    )
    gates.append(
        _evaluate_max_gate(
            "low_information_clarification_rate",
            _as_float(summary.get("low_information_clarification_rate")),
            thresholds.max_low_information_clarification_rate,
            message="Too many low-information follow-ups indicate poor question quality.",
        )
    )

    ambiguity_metrics = summary.get("ambiguity_group_metrics")
    if not isinstance(ambiguity_metrics, dict):
        ambiguity_metrics = {}

    for group in required_ambiguity_groups or []:
        group_payload = ambiguity_metrics.get(group)
        if not isinstance(group_payload, dict):
            gates.append(
                {
                    "name": f"ambiguity_group:{group}",
                    "passed": False,
                    "actual": None,
                    "threshold": thresholds.min_ambiguity_group_top_1_accuracy,
                    "operator": ">=",
                    "message": "Required ambiguity group is missing from the benchmark summary.",
                }
            )
            continue

        gates.append(
            _evaluate_min_gate(
                f"ambiguity_group:{group}",
                _as_float(group_payload.get(thresholds.ambiguity_group_metric_key)),
                thresholds.min_ambiguity_group_top_1_accuracy,
                message=(
                    "Each critical ambiguity group must keep acceptable performance on "
                    f"'{thresholds.ambiguity_group_metric_key}'."
                ),
            )
        )

    if baseline_summary:
        current_clinical = _as_float(summary.get("clinical_top_1_accuracy"))
        baseline_clinical = _as_float(baseline_summary.get("clinical_top_1_accuracy"))
        clinical_delta = (
            current_clinical - baseline_clinical
            if current_clinical is not None and baseline_clinical is not None
            else None
        )
        gates.append(
            _evaluate_min_gate(
                "clinical_top_1_delta_vs_baseline",
                clinical_delta,
                thresholds.min_clinical_top_1_delta_vs_baseline,
                message="Strict Top-1 should not regress against baseline bundle.",
            )
        )

        current_post = _as_float(summary.get("post_clarification_clinical_top_1_accuracy"))
        baseline_post = _as_float(baseline_summary.get("post_clarification_clinical_top_1_accuracy"))
        post_delta = current_post - baseline_post if current_post is not None and baseline_post is not None else None
        gates.append(
            _evaluate_min_gate(
                "post_clarification_top_1_delta_vs_baseline",
                post_delta,
                thresholds.min_post_clarification_top_1_delta_vs_baseline,
                message="Clarified Top-1 should not regress against baseline bundle.",
            )
        )

    decision = "go" if all(gate["passed"] for gate in gates) else "no-go"
    return {
        "decision": decision,
        "all_gates_passed": decision == "go",
        "checked_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "thresholds": asdict(thresholds),
        "gates": gates,
    }


def _thresholds_from_args(args: argparse.Namespace) -> GateThresholds:
    return GateThresholds(
        min_cases=args.min_cases,
        min_clinical_top_1_accuracy=args.min_clinical_top1,
        min_ai_scope_top_1_accuracy=args.min_ai_scope_top1,
        min_post_clarification_clinical_top_1_accuracy=args.min_post_clarification_clinical_top1,
        min_clarification_utility_rate=args.min_clarification_utility_rate,
        max_low_information_clarification_rate=args.max_low_information_clarification_rate,
        min_ambiguity_group_top_1_accuracy=args.min_ambiguity_group_top1,
        min_clinical_top_1_delta_vs_baseline=args.min_clinical_top1_delta_vs_baseline,
        min_post_clarification_top_1_delta_vs_baseline=args.min_post_clarification_top1_delta_vs_baseline,
        ambiguity_group_metric_key=args.ambiguity_group_metric_key,
    )


def main() -> int:
    args = parse_args()
    summary = _load_json(args.summary)
    baseline_summary = _load_json(args.baseline_summary) if args.baseline_summary else None
    thresholds = _thresholds_from_args(args)
    report = evaluate_summary(
        summary,
        thresholds,
        baseline_summary=baseline_summary,
        required_ambiguity_groups=list(dict.fromkeys(args.required_ambiguity_group or [])),
    )

    print(f"Decision: {report['decision'].upper()}")
    for gate in report["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        print(
            f"[{status}] {gate['name']}: actual={gate['actual']} "
            f"{gate['operator']} threshold={gate['threshold']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
