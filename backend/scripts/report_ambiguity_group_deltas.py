from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATTERN = "data/evaluation/targeted_cases_v1_summary_track2_canonicalized_v*.json"
DEFAULT_OUTPUT = Path("data/evaluation/ambiguity_group_delta_report_latest.json")
DEFAULT_GROUP_METRICS = [
    "top_1_accuracy",
    "post_clarification_top_1_accuracy",
    "clarification_improved_rate",
]
DEFAULT_TOP_LEVEL_METRICS = [
    "clinical_top_1_accuracy",
    "post_clarification_clinical_top_1_accuracy",
    "clarification_utility_rate",
    "low_information_clarification_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a weekly ambiguity-group delta report between a current and baseline summary. "
            "If summary paths are omitted, the script auto-discovers the latest two versioned summaries."
        )
    )
    parser.add_argument("--current-summary", type=Path, default=None)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--group-metric",
        action="append",
        default=[],
        help="Ambiguity-group metric key to compare. Can be repeated.",
    )
    parser.add_argument(
        "--top-level-metric",
        action="append",
        default=[],
        help="Top-level summary metric key to compare. Can be repeated.",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=-0.05,
        help="A metric is flagged as regressed when delta is below this threshold.",
    )
    parser.add_argument(
        "--required-group",
        action="append",
        default=[],
        help="Group that must exist in current summary and should not regress.",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_version(path: Path) -> Optional[int]:
    match = re.search(r"_v(\d+)\.json$", path.name)
    if not match:
        return None
    return int(match.group(1))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve_pattern_candidates(pattern: str) -> list[Path]:
    pattern_path = Path(pattern)
    candidates: list[Path] = []

    if pattern_path.is_absolute():
        for item in pattern_path.parent.glob(pattern_path.name):
            if item.is_file():
                candidates.append(item)
        return sorted(set(path.resolve() for path in candidates))

    search_roots = [Path.cwd(), REPO_ROOT]
    for root in search_roots:
        for item in root.glob(pattern):
            if item.is_file():
                candidates.append(item.resolve())

    return sorted(set(candidates))


def discover_latest_summaries(pattern: str) -> tuple[Path, Path]:
    candidates = _resolve_pattern_candidates(pattern)
    if len(candidates) < 2:
        raise ValueError(f"Need at least two summary files for comparison. Found {len(candidates)} using pattern: {pattern}")

    versioned = [(version, path) for path in candidates if (version := _extract_version(path)) is not None]
    if len(versioned) >= 2:
        versioned.sort(key=lambda item: item[0])
        baseline = versioned[-2][1]
        current = versioned[-1][1]
        return baseline, current

    by_mtime = sorted(candidates, key=lambda item: item.stat().st_mtime)
    return by_mtime[-2], by_mtime[-1]


def _metric_delta(current: dict[str, Any], baseline: dict[str, Any], key: str) -> dict[str, Any]:
    current_value = _as_float(current.get(key))
    baseline_value = _as_float(baseline.get(key))
    delta = current_value - baseline_value if current_value is not None and baseline_value is not None else None
    return {
        "current": current_value,
        "baseline": baseline_value,
        "delta": delta,
    }


def _ambiguity_groups(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = summary.get("ambiguity_group_metrics")
    if not isinstance(payload, dict):
        return {}

    groups: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            groups[str(key)] = value
    return groups


def build_ambiguity_delta_report(
    *,
    current_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    current_summary_path: Optional[Path] = None,
    baseline_summary_path: Optional[Path] = None,
    group_metric_keys: Optional[list[str]] = None,
    top_level_metric_keys: Optional[list[str]] = None,
    regression_threshold: float = -0.05,
    required_groups: Optional[list[str]] = None,
) -> dict[str, Any]:
    group_metric_keys = group_metric_keys or list(DEFAULT_GROUP_METRICS)
    top_level_metric_keys = top_level_metric_keys or list(DEFAULT_TOP_LEVEL_METRICS)

    current_groups = _ambiguity_groups(current_summary)
    baseline_groups = _ambiguity_groups(baseline_summary)
    shared_groups = sorted(set(current_groups).intersection(baseline_groups))
    new_groups = sorted(set(current_groups) - set(baseline_groups))
    dropped_groups = sorted(set(baseline_groups) - set(current_groups))

    top_level_deltas = {
        metric: _metric_delta(current_summary, baseline_summary, metric)
        for metric in top_level_metric_keys
    }

    regression_groups_by_metric: dict[str, list[str]] = {metric: [] for metric in group_metric_keys}
    group_rows: list[dict[str, Any]] = []

    for group in sorted(set(current_groups).union(baseline_groups)):
        current_payload = current_groups.get(group)
        baseline_payload = baseline_groups.get(group)

        state = "shared"
        if baseline_payload is None:
            state = "new"
        elif current_payload is None:
            state = "dropped"

        metrics: dict[str, dict[str, Any]] = {}
        for metric in group_metric_keys:
            current_value = _as_float(current_payload.get(metric)) if isinstance(current_payload, dict) else None
            baseline_value = _as_float(baseline_payload.get(metric)) if isinstance(baseline_payload, dict) else None
            delta = (
                current_value - baseline_value
                if current_value is not None and baseline_value is not None
                else None
            )
            regressed = state == "shared" and delta is not None and delta < regression_threshold
            metrics[metric] = {
                "current": current_value,
                "baseline": baseline_value,
                "delta": delta,
                "regressed": regressed,
            }
            if regressed:
                regression_groups_by_metric[metric].append(group)

        group_rows.append(
            {
                "group": group,
                "state": state,
                "num_cases": {
                    "current": (
                        int(current_payload.get("num_cases"))
                        if isinstance(current_payload, dict) and current_payload.get("num_cases") is not None
                        else None
                    ),
                    "baseline": (
                        int(baseline_payload.get("num_cases"))
                        if isinstance(baseline_payload, dict) and baseline_payload.get("num_cases") is not None
                        else None
                    ),
                },
                "metrics": metrics,
                "has_regression": any(item["regressed"] for item in metrics.values()),
            }
        )

    required_checks: list[dict[str, Any]] = []
    required_unique = list(dict.fromkeys(required_groups or []))
    for group in required_unique:
        row = next((item for item in group_rows if item["group"] == group), None)
        has_regression = bool(row and row.get("has_regression"))
        in_current = group in current_groups
        in_baseline = group in baseline_groups
        required_checks.append(
            {
                "group": group,
                "in_current": in_current,
                "in_baseline": in_baseline,
                "present_in_both": in_current and in_baseline,
                "has_regression": has_regression,
                "passed": in_current and not has_regression,
            }
        )

    regressed_groups = sorted(
        {
            group
            for groups in regression_groups_by_metric.values()
            for group in groups
        }
    )

    required_failures = [item for item in required_checks if not item["passed"]]
    decision = "regression-detected" if regressed_groups or required_failures else "stable-or-improved"

    return {
        "generated_at_utc": _utc_now_iso(),
        "decision": decision,
        "regression_threshold": regression_threshold,
        "baseline_summary": {
            "path": str(baseline_summary_path) if baseline_summary_path else None,
            "version": _extract_version(baseline_summary_path) if baseline_summary_path else None,
        },
        "current_summary": {
            "path": str(current_summary_path) if current_summary_path else None,
            "version": _extract_version(current_summary_path) if current_summary_path else None,
        },
        "top_level_deltas": top_level_deltas,
        "group_summary": {
            "baseline_group_count": len(baseline_groups),
            "current_group_count": len(current_groups),
            "shared_group_count": len(shared_groups),
            "new_groups": new_groups,
            "dropped_groups": dropped_groups,
            "regressed_groups": regressed_groups,
        },
        "regression_groups_by_metric": regression_groups_by_metric,
        "required_group_checks": required_checks,
        "groups": group_rows,
    }


def main() -> int:
    args = parse_args()

    if (args.current_summary is None) != (args.baseline_summary is None):
        raise SystemExit("Provide both --current-summary and --baseline-summary, or provide neither.")

    if args.current_summary and args.baseline_summary:
        baseline_path = args.baseline_summary
        current_path = args.current_summary
    else:
        baseline_path, current_path = discover_latest_summaries(args.pattern)

    baseline_summary = _load_json(baseline_path)
    current_summary = _load_json(current_path)

    report = build_ambiguity_delta_report(
        current_summary=current_summary,
        baseline_summary=baseline_summary,
        current_summary_path=current_path,
        baseline_summary_path=baseline_path,
        group_metric_keys=list(dict.fromkeys(args.group_metric)) or None,
        top_level_metric_keys=list(dict.fromkeys(args.top_level_metric)) or None,
        regression_threshold=args.regression_threshold,
        required_groups=list(dict.fromkeys(args.required_group)),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Baseline: {baseline_path}")
    print(f"Current:  {current_path}")
    print(f"Decision: {report['decision'].upper()}")
    print(
        "Shared/New/Dropped groups: "
        f"{report['group_summary']['shared_group_count']}/"
        f"{len(report['group_summary']['new_groups'])}/"
        f"{len(report['group_summary']['dropped_groups'])}"
    )
    print(
        "Regressed groups: "
        f"{len(report['group_summary']['regressed_groups'])}"
    )

    return 0 if report["decision"] == "stable-or-improved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
