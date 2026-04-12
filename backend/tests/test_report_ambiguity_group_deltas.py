from __future__ import annotations

import json
from pathlib import Path

from scripts.report_ambiguity_group_deltas import (
    build_ambiguity_delta_report,
    discover_latest_summaries,
)


def _summary(**overrides):
    payload = {
        "num_cases": 16,
        "clinical_top_1_accuracy": 0.5,
        "post_clarification_clinical_top_1_accuracy": 0.75,
        "clarification_utility_rate": 0.45,
        "low_information_clarification_rate": 0.15,
        "ambiguity_group_metrics": {
            "atrial_fibrillation_vs_psvt": {
                "num_cases": 1,
                "top_1_accuracy": 0.5,
                "post_clarification_top_1_accuracy": 1.0,
                "clarification_improved_rate": 0.5,
            },
            "pneumonia_vs_bronchospasm": {
                "num_cases": 1,
                "top_1_accuracy": 0.5,
                "post_clarification_top_1_accuracy": 1.0,
                "clarification_improved_rate": 0.5,
            },
        },
    }
    payload.update(overrides)
    return payload


def test_build_delta_report_flags_regression_and_required_group_failure():
    baseline = _summary()
    current = _summary(
        ambiguity_group_metrics={
            "atrial_fibrillation_vs_psvt": {
                "num_cases": 1,
                "top_1_accuracy": 0.2,
                "post_clarification_top_1_accuracy": 0.9,
                "clarification_improved_rate": 0.4,
            },
            "new_group": {
                "num_cases": 1,
                "top_1_accuracy": 1.0,
                "post_clarification_top_1_accuracy": 1.0,
                "clarification_improved_rate": 0.0,
            },
        }
    )

    report = build_ambiguity_delta_report(
        current_summary=current,
        baseline_summary=baseline,
        regression_threshold=-0.05,
        required_groups=["atrial_fibrillation_vs_psvt", "pneumonia_vs_bronchospasm"],
    )

    assert report["decision"] == "regression-detected"
    assert "atrial_fibrillation_vs_psvt" in report["regression_groups_by_metric"]["top_1_accuracy"]

    required = {item["group"]: item for item in report["required_group_checks"]}
    assert required["atrial_fibrillation_vs_psvt"]["passed"] is False
    assert required["pneumonia_vs_bronchospasm"]["in_current"] is False


def test_build_delta_report_stable_when_no_regressions_and_required_present():
    baseline = _summary()
    current = _summary(
        clinical_top_1_accuracy=0.6,
        ambiguity_group_metrics={
            "atrial_fibrillation_vs_psvt": {
                "num_cases": 1,
                "top_1_accuracy": 0.5,
                "post_clarification_top_1_accuracy": 1.0,
                "clarification_improved_rate": 0.5,
            },
            "pneumonia_vs_bronchospasm": {
                "num_cases": 1,
                "top_1_accuracy": 0.5,
                "post_clarification_top_1_accuracy": 1.0,
                "clarification_improved_rate": 0.5,
            },
        },
    )

    report = build_ambiguity_delta_report(
        current_summary=current,
        baseline_summary=baseline,
        required_groups=["atrial_fibrillation_vs_psvt", "pneumonia_vs_bronchospasm"],
    )

    assert report["decision"] == "stable-or-improved"
    assert report["group_summary"]["regressed_groups"] == []


def test_discover_latest_summaries_prefers_version_order(tmp_path: Path):
    for version in [2, 7, 11]:
        path = tmp_path / f"targeted_cases_v1_summary_track2_canonicalized_v{version}.json"
        path.write_text(json.dumps({"version": version}), encoding="utf-8")

    baseline, current = discover_latest_summaries(
        str(tmp_path / "targeted_cases_v1_summary_track2_canonicalized_v*.json")
    )

    assert baseline.name.endswith("_v7.json")
    assert current.name.endswith("_v11.json")
