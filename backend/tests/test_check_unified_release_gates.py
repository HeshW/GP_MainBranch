from __future__ import annotations

from scripts.check_release_gates import GateThresholds
from scripts.check_unified_release_gates import build_unified_report


def _summary(**overrides):
    payload = {
        "num_cases": 16,
        "clinical_top_1_accuracy": 0.5,
        "ai_scope_top_1_accuracy": 0.5,
        "post_clarification_clinical_top_1_accuracy": 1.0,
        "clarification_utility_rate": 0.6,
        "low_information_clarification_rate": 0.12,
        "ambiguity_group_metrics": {
            "atrial_fibrillation_vs_psvt": {
                "top_1_accuracy": 0.5,
                "post_clarification_top_1_accuracy": 1.0,
                "clarification_improved_rate": 0.5,
            },
            "pneumonia_vs_bronchospasm": {
                "top_1_accuracy": 0.5,
                "post_clarification_top_1_accuracy": 1.0,
                "clarification_improved_rate": 0.5,
            },
        },
    }
    payload.update(overrides)
    return payload


def test_build_unified_report_go_when_both_components_pass():
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

    report = build_unified_report(
        discussion_summary=_summary(),
        targeted_summary=_summary(),
        discussion_thresholds=thresholds,
        targeted_thresholds=thresholds,
        required_targeted_ambiguity_groups=["atrial_fibrillation_vs_psvt", "pneumonia_vs_bronchospasm"],
    )

    assert report["decision"] == "go"
    assert report["all_gates_passed"] is True


def test_build_unified_report_no_go_when_targeted_fails_gate():
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

    report = build_unified_report(
        discussion_summary=_summary(),
        targeted_summary=_summary(low_information_clarification_rate=0.9),
        discussion_thresholds=thresholds,
        targeted_thresholds=thresholds,
        required_targeted_ambiguity_groups=["atrial_fibrillation_vs_psvt", "pneumonia_vs_bronchospasm"],
    )

    assert report["decision"] == "no-go"
    assert report["all_gates_passed"] is False
    assert report["components"]["targeted"]["decision"] == "no-go"
