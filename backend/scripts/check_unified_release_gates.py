from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_release_gates import GateThresholds, evaluate_summary


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_unified_report(
    *,
    discussion_summary: dict[str, Any],
    targeted_summary: dict[str, Any],
    discussion_thresholds: GateThresholds,
    targeted_thresholds: GateThresholds,
    discussion_baseline_summary: Optional[dict[str, Any]] = None,
    targeted_baseline_summary: Optional[dict[str, Any]] = None,
    required_discussion_ambiguity_groups: Optional[list[str]] = None,
    required_targeted_ambiguity_groups: Optional[list[str]] = None,
) -> dict[str, Any]:
    discussion_report = evaluate_summary(
        discussion_summary,
        discussion_thresholds,
        baseline_summary=discussion_baseline_summary,
        required_ambiguity_groups=required_discussion_ambiguity_groups or [],
    )
    targeted_report = evaluate_summary(
        targeted_summary,
        targeted_thresholds,
        baseline_summary=targeted_baseline_summary,
        required_ambiguity_groups=required_targeted_ambiguity_groups or [],
    )

    overall_go = bool(discussion_report.get("all_gates_passed")) and bool(targeted_report.get("all_gates_passed"))
    return {
        "checked_at_utc": _utc_now_iso(),
        "decision": "go" if overall_go else "no-go",
        "all_gates_passed": overall_go,
        "components": {
            "discussion": discussion_report,
            "targeted": targeted_report,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply unified release gates across discussion and targeted benchmark summaries. "
            "Decision is GO only if both components pass."
        )
    )
    parser.add_argument("--discussion-summary", type=Path, required=True)
    parser.add_argument("--targeted-summary", type=Path, required=True)
    parser.add_argument("--discussion-baseline-summary", type=Path, default=None)
    parser.add_argument("--targeted-baseline-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument("--required-discussion-ambiguity-group", action="append", default=[])
    parser.add_argument("--required-targeted-ambiguity-group", action="append", default=[])

    parser.add_argument("--discussion-min-cases", type=int, default=16)
    parser.add_argument("--discussion-min-clinical-top1", type=float, default=0.20)
    parser.add_argument("--discussion-min-ai-scope-top1", type=float, default=0.20)
    parser.add_argument("--discussion-min-post-clarification-clinical-top1", type=float, default=0.35)
    parser.add_argument("--discussion-min-clarification-utility-rate", type=float, default=0.20)
    parser.add_argument("--discussion-max-low-information-clarification-rate", type=float, default=0.45)
    parser.add_argument("--discussion-min-ambiguity-group-top1", type=float, default=0.25)
    parser.add_argument(
        "--discussion-ambiguity-group-metric-key",
        type=str,
        default="top_1_accuracy",
        choices=["top_1_accuracy", "post_clarification_top_1_accuracy", "clarification_improved_rate"],
    )

    parser.add_argument("--targeted-min-cases", type=int, default=16)
    parser.add_argument("--targeted-min-clinical-top1", type=float, default=0.25)
    parser.add_argument("--targeted-min-ai-scope-top1", type=float, default=0.20)
    parser.add_argument("--targeted-min-post-clarification-clinical-top1", type=float, default=0.70)
    parser.add_argument("--targeted-min-clarification-utility-rate", type=float, default=0.25)
    parser.add_argument("--targeted-max-low-information-clarification-rate", type=float, default=0.20)
    parser.add_argument("--targeted-min-ambiguity-group-top1", type=float, default=0.25)
    parser.add_argument(
        "--targeted-ambiguity-group-metric-key",
        type=str,
        default="post_clarification_top_1_accuracy",
        choices=["top_1_accuracy", "post_clarification_top_1_accuracy", "clarification_improved_rate"],
    )

    return parser.parse_args()


def _thresholds_from_args(args: argparse.Namespace, prefix: str) -> GateThresholds:
    return GateThresholds(
        min_cases=getattr(args, f"{prefix}_min_cases"),
        min_clinical_top_1_accuracy=getattr(args, f"{prefix}_min_clinical_top1"),
        min_ai_scope_top_1_accuracy=getattr(args, f"{prefix}_min_ai_scope_top1"),
        min_post_clarification_clinical_top_1_accuracy=getattr(
            args, f"{prefix}_min_post_clarification_clinical_top1"
        ),
        min_clarification_utility_rate=getattr(args, f"{prefix}_min_clarification_utility_rate"),
        max_low_information_clarification_rate=getattr(
            args, f"{prefix}_max_low_information_clarification_rate"
        ),
        min_ambiguity_group_top_1_accuracy=getattr(args, f"{prefix}_min_ambiguity_group_top1"),
        ambiguity_group_metric_key=getattr(args, f"{prefix}_ambiguity_group_metric_key"),
    )


def main() -> int:
    args = parse_args()

    discussion_summary = _load_json(args.discussion_summary)
    targeted_summary = _load_json(args.targeted_summary)

    discussion_baseline_summary = _load_json(args.discussion_baseline_summary) if args.discussion_baseline_summary else None
    targeted_baseline_summary = _load_json(args.targeted_baseline_summary) if args.targeted_baseline_summary else None

    discussion_thresholds = _thresholds_from_args(args, "discussion")
    targeted_thresholds = _thresholds_from_args(args, "targeted")

    required_targeted = list(args.required_targeted_ambiguity_group)
    if not required_targeted:
        required_targeted = ["atrial_fibrillation_vs_psvt", "pneumonia_vs_bronchospasm"]

    report = build_unified_report(
        discussion_summary=discussion_summary,
        targeted_summary=targeted_summary,
        discussion_thresholds=discussion_thresholds,
        targeted_thresholds=targeted_thresholds,
        discussion_baseline_summary=discussion_baseline_summary,
        targeted_baseline_summary=targeted_baseline_summary,
        required_discussion_ambiguity_groups=list(args.required_discussion_ambiguity_group),
        required_targeted_ambiguity_groups=required_targeted,
    )

    discussion_decision = report["components"]["discussion"]["decision"]
    targeted_decision = report["components"]["targeted"]["decision"]
    print(f"Discussion decision: {discussion_decision.upper()}")
    print(f"Targeted decision: {targeted_decision.upper()}")
    print(f"Unified decision: {report['decision'].upper()}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
