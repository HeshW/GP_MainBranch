import json
from argparse import Namespace
from pathlib import Path

from manager.runtime import run_async
from manager.symptom_parser import parse_symptoms
from manager.symptom_validator import validate_parsed
from scripts import evaluate_pipeline_quality as epq


class FakeManager:
    async def run_from_symptoms(self, text):
        if "urinating" in text or "glucose" in text:
            return {
                "status": "ok",
                "diagnosis": {
                    "final_diagnosis": {
                        "diagnosis": "Possible hyperglycemia / diabetes symptom pattern",
                        "confidence": 0.45,
                        "source": "symptom_rules",
                        "mode": "rules_fallback",
                    },
                    "findings": [
                        {
                            "condition": "Possible hyperglycemia / diabetes symptom pattern",
                            "source": "symptom_rules",
                            "confidence": "low",
                        }
                    ],
                    "decision_fusion": {"primary_source": "symptom_rules"},
                    "safety": {"clinician_review_required": True},
                    "rag_metadata": {"rag_scope_status": "out_of_scope_or_low_confidence"},
                },
                "elapsed_ms": 1.0,
            }
        return {
            "status": "ok",
            "diagnosis": {
                "final_diagnosis": {
                    "diagnosis": "Pneumonia",
                    "confidence": 0.86,
                    "source": "classifier_rag_consensus",
                    "mode": "ai_primary",
                },
                "diagnostic_candidates": [{"label": "Pneumonia", "confidence": 0.86, "sources": ["classifier"]}],
                "classifier_prediction": {
                    "predicted_label": "Pneumonia",
                    "confidence": 0.84,
                    "top_predictions": [{"label": "Pneumonia"}],
                },
                "retrieved_cases": [{"pathology": "Pneumonia"}],
                "findings": [{"condition": "Possible lower respiratory infection pattern", "source": "symptom_rules"}],
                "decision_fusion": {"primary_source": "classifier_rag_consensus"},
                "safety": {"clinician_review_required": True},
            },
            "elapsed_ms": 1.0,
        }


def test_evaluate_case_offline_no_llm_mode_in_scope():
    detail = run_async(
        epq.evaluate_case(
            FakeManager(),
            {
                "case_id": "x",
                "input_text": "fever, productive cough, and shortness of breath",
                "expected_label": "Pneumonia",
                "scope": "in_scope",
                "language": "en",
            },
            supported_labels={"Pneumonia"},
        )
    )

    assert detail["top_1_hit"] is True
    assert detail["classifier_agrees"] is True
    assert detail["rag_agrees"] is True


def test_out_of_scope_case_does_not_become_confident_in_scope():
    detail = run_async(
        epq.evaluate_case(
            FakeManager(),
            {
                "case_id": "oos",
                "input_text": "thirst and urinating frequently with glucose 240 mg/dL",
                "expected_family": "out of scope",
                "scope": "out_of_scope",
                "language": "en",
            },
            supported_labels={"Pneumonia", "Stable angina"},
        )
    )

    assert detail["safe_out_of_scope_handling"] is True
    assert detail["unsafe_confident_diagnosis"] is False


def test_threshold_flag_behavior():
    args = Namespace(
        in_scope_top1_threshold=0.70,
        in_scope_top3_threshold=0.85,
        out_of_scope_safe_threshold=0.90,
        parser_success_threshold=0.85,
        unsafe_confident_threshold=0.05,
    )
    payload = epq.threshold_results(
        {
            "in_scope": {"top_1_accuracy": 0.5, "top_3_accuracy": 0.9},
            "out_of_scope": {"safe_handling_rate": 1.0, "unsafe_confident_diagnosis_rate": 0.0},
            "overall": {"parser_success_rate": 1.0},
        },
        args,
    )

    assert payload["passed"] is False
    assert payload["checks"]["in_scope_top_1_accuracy"]["passed"] is False


def test_arabic_symptom_parser_support_is_measured():
    validated = validate_parsed(parse_symptoms("عندي دوخة وتعب"))

    assert "fatigue" in validated["symptoms"]
    assert "dizziness" in validated["symptoms"]


def test_noisy_input_category_metrics_are_reported():
    details = [
        {
            "scope": "in_scope",
            "case_set": "pipeline_noisy_typo_cases",
            "parser_success": False,
            "normalization_success": False,
            "top_3_hit": True,
            "abstention_or_safe_fallback": True,
        }
    ]

    metrics = epq.compute_metrics(details)

    assert metrics["input_robustness"]["case_count"] == 1
    assert metrics["input_robustness"]["parser_success_rate"] == 0.0
    assert metrics["input_robustness"]["final_diagnosis_hit_rate"] == 1.0


def test_output_report_files_are_generated(tmp_path: Path):
    metrics = {
        "case_counts": {"by_scope": {"in_scope": 1}, "by_case_set": {"sample": 1}},
        "in_scope": {
            "top_1_accuracy": 1.0,
            "top_3_accuracy": 1.0,
            "expected_label_or_family_hit_rate": 1.0,
            "classifier_agreement_rate": 1.0,
            "rag_agreement_rate": 1.0,
            "rules_agreement_rate": 1.0,
        },
        "out_of_scope": {
            "safe_handling_rate": 1.0,
            "unsafe_confident_diagnosis_rate": 0.0,
            "rag_classifier_dominated_count": 0,
            "professional_care_or_safe_fallback_rate": 1.0,
        },
        "overall": {"parser_success_rate": 1.0, "normalization_success_rate": 1.0},
        "fusion_source_contribution": {"classifier": 1},
    }
    rows = [
        {
            "failed": False,
            "case_id": "sample",
            "input_text": "fever cough",
            "expected_label": "Pneumonia",
            "expected_family": "",
            "final_diagnosis": "Pneumonia",
            "final_confidence": 0.9,
            "final_source": "classifier",
            "classifier_top_predictions": "Pneumonia",
            "rag_retrieved_labels": "Pneumonia",
            "rules_candidates": "",
            "clarification_needed": False,
            "likely_failure_cause": "",
        }
    ]
    threshold_payload = {"passed": True, "checks": {}}

    epq.write_markdown_reports(tmp_path, metrics=metrics, rows=rows, threshold_payload=threshold_payload)
    epq.write_csv(tmp_path / "pipeline_eval_cases.csv", rows)
    (tmp_path / "pipeline_eval_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    assert (tmp_path / "pipeline_eval_report.md").exists()
    assert (tmp_path / "pipeline_failure_analysis.md").exists()
    assert (tmp_path / "pipeline_safety_report.md").exists()
    assert (tmp_path / "pipeline_eval_cases.csv").exists()
    assert (tmp_path / "pipeline_eval_summary.json").exists()
