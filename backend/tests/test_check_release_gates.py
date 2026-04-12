from scripts.check_release_gates import GateThresholds, evaluate_summary


def _summary(**overrides):
    payload = {
        "num_cases": 120,
        "clinical_top_1_accuracy": 0.63,
        "ai_scope_top_1_accuracy": 0.61,
        "post_clarification_clinical_top_1_accuracy": 0.76,
        "clarification_utility_rate": 0.31,
        "low_information_clarification_rate": 0.24,
        "ambiguity_group_metrics": {
            "cardio": {"top_1_accuracy": 0.57},
            "resp": {"top_1_accuracy": 0.61},
        },
    }
    payload.update(overrides)
    return payload


def test_evaluate_summary_go_when_all_thresholds_pass():
    thresholds = GateThresholds(min_cases=100)
    report = evaluate_summary(
        _summary(),
        thresholds,
        required_ambiguity_groups=["cardio"],
    )

    assert report["decision"] == "go"
    assert report["all_gates_passed"] is True


def test_evaluate_summary_no_go_when_low_information_followups_too_high():
    thresholds = GateThresholds(max_low_information_clarification_rate=0.45)
    report = evaluate_summary(_summary(low_information_clarification_rate=0.8), thresholds)

    assert report["decision"] == "no-go"
    failed = [gate for gate in report["gates"] if not gate["passed"]]
    assert any(gate["name"] == "low_information_clarification_rate" for gate in failed)


def test_evaluate_summary_no_go_when_required_ambiguity_group_missing():
    report = evaluate_summary(_summary(), GateThresholds(), required_ambiguity_groups=["neuro"])

    assert report["decision"] == "no-go"
    missing_group_gate = next(gate for gate in report["gates"] if gate["name"] == "ambiguity_group:neuro")
    assert missing_group_gate["passed"] is False


def test_evaluate_summary_no_go_on_baseline_regression_gate():
    current = _summary(clinical_top_1_accuracy=0.61)
    baseline = _summary(clinical_top_1_accuracy=0.64)
    thresholds = GateThresholds(min_clinical_top_1_delta_vs_baseline=0.0)

    report = evaluate_summary(current, thresholds, baseline_summary=baseline)

    assert report["decision"] == "no-go"
    regression_gate = next(gate for gate in report["gates"] if gate["name"] == "clinical_top_1_delta_vs_baseline")
    assert regression_gate["passed"] is False


def test_evaluate_summary_uses_post_clarification_metric_for_ambiguity_group_gate():
    current = _summary(
        ambiguity_group_metrics={
            "cardio": {
                "top_1_accuracy": 0.0,
                "post_clarification_top_1_accuracy": 1.0,
            }
        }
    )
    thresholds = GateThresholds(
        min_ambiguity_group_top_1_accuracy=0.5,
        ambiguity_group_metric_key="post_clarification_top_1_accuracy",
    )

    report = evaluate_summary(current, thresholds, required_ambiguity_groups=["cardio"])

    group_gate = next(gate for gate in report["gates"] if gate["name"] == "ambiguity_group:cardio")
    assert group_gate["passed"] is True
