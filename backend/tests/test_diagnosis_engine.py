import pytest

from manager.runtime import run_async
from models.diagnosis.diagnosisengine import DiagnosisEngine, diagnose
from models.diagnosis.text import build_combined_text


def test_rule_based_detects_diabetes():
    report = {"labs": {"glucose": {"value": 130, "unit": "mg/dL"}}}
    out = run_async(diagnose(report))
    conditions = [finding["condition"] for finding in out["findings"]]
    assert any("Diabetes" in condition or "diabetes" in condition for condition in conditions)


def test_accepts_plain_float_value():
    report = {"labs": {"glucose": 140}}
    out = run_async(diagnose(report))
    assert out["findings"]


def test_no_labs_returns_empty_summary():
    out = run_async(diagnose({}))
    assert "No clinically significant" in out["summary"]
    assert out["decision_fusion"]["primary_source"] == "rules"
    assert out["safety"]["clinician_review_required"] is False


def test_symptoms_only_returns_symptom_rule_finding():
    out = run_async(diagnose({"symptoms": ["fatigue", "thirst"], "raw_text": "fatigue thirst"}))

    assert out["findings"]
    assert out["decision_fusion"]["primary_source"] == "rules_fallback"
    assert any(finding["source"] == "symptom_rules" for finding in out["findings"])
    assert "AI-assisted assessment suggests" in out["summary"]


def test_reflux_context_triggers_symptom_rule():
    out = run_async(
        diagnose(
            {
                "symptoms": ["reflux"],
                "raw_text": "Burning retrosternal discomfort after meals with sour taste and reflux.",
            }
        )
    )

    conditions = [finding["condition"] for finding in out["findings"]]
    assert any("reflux" in condition.lower() or "gastroesophageal" in condition.lower() for condition in conditions)


def test_diagnosis_returns_fusion_and_safety_metadata():
    report = {"labs": {"hemoglobin": {"value": 7.5, "unit": "g/dL"}}}
    out = run_async(diagnose(report))

    assert out["decision_fusion"]["primary_source"] == "rules_fallback"
    assert "lab_rules" in out["decision_fusion"]["supporting_sources"]
    assert out["safety"]["clinician_review_required"] is True
    assert out["safety"]["highest_rule_severity"] == "critical"
    assert out["safety"]["emergency_attention_recommended"] is True
    assert out["final_diagnosis"]["mode"] == "rules_fallback"


def test_classifier_becomes_primary_diagnosis_source(monkeypatch):
    class StubClassifier:
        def __init__(self, model_dir, max_length=256, device=None):
            pass

        def predict(self, text):
            return {
                "predicted_label": "Influenza",
                "confidence": 0.91,
                "top_predictions": [{"label": "Influenza", "confidence": 0.91}],
            }

    monkeypatch.setattr("models.diagnosis.diagnosisengine.FineTunedDiagnosisClassifier", StubClassifier)

    engine = DiagnosisEngine(
        use_finetuned_classifier=True,
        finetuned_model_dir="fake-model-dir",
    )
    out = run_async(engine.diagnose({"raw_text": "fever cough fatigue", "labs": {}}))

    assert out["final_diagnosis"]["diagnosis"] == "Influenza"
    assert out["decision_fusion"]["primary_source"] == "classifier"
    assert out["final_diagnosis"]["mode"] == "ai_primary"


def test_symptom_like_rag_condition_is_not_selected_as_final_diagnosis():
    final = DiagnosisEngine._build_final_diagnosis(
        findings=[
            {
                "condition": "Possible hyperglycemia / diabetes symptom pattern",
                "confidence": "low",
                "evidence": "Symptoms suggestive of hyperglycemia: fatigue, thirst.",
                "severity": "moderate",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["fatigue", "thirst"],
        rag_out={
            "structured_diagnosis": {
                "assessment_summary": "Non-specific symptoms only.",
                "findings": [
                    {
                        "condition": "Fatigue",
                        "confidence": "High",
                        "evidence": "Reported symptom.",
                        "severity": "Info",
                    }
                ],
            },
            "retrieved_cases": [
                {
                    "pathology": "Myocarditis",
                    "similarity": 0.62,
                }
            ],
        },
        classifier_prediction=None,
    )

    assert final is not None
    assert final["diagnosis"] != "Fatigue"
    assert final["diagnosis"] == "Possible hyperglycemia / diabetes symptom pattern"
    assert final["source"] == "rules_fallback"


def test_lab_rule_overrides_conflicting_ai_label():
    final = DiagnosisEngine._build_final_diagnosis(
        findings=[
            {
                "condition": "Diabetes Mellitus (suspected)",
                "confidence": "high",
                "evidence": "glucose=145 mg/dL",
                "severity": "high",
                "source": "lab_rules",
            }
        ],
        patient_symptoms=[],
        rag_out={
            "retrieved_cases": [
                {
                    "pathology": "Tuberculosis",
                    "similarity": 0.69,
                }
            ]
        },
        classifier_prediction={
            "predicted_label": "Viral pharyngitis",
            "confidence": 0.16,
            "top_predictions": [{"label": "Viral pharyngitis", "confidence": 0.16}],
        },
    )

    assert final is not None
    assert final["diagnosis"] == "Diabetes Mellitus (suspected)"
    assert final["source"] == "rules_fallback"


def test_diagnose_raises_on_invalid_report_type():
    engine = DiagnosisEngine()
    with pytest.raises(TypeError):
        run_async(engine.diagnose(None))


def test_build_combined_text_includes_raw_text_and_symptoms():
    report = {
        "raw_text": "Fatigue and increased thirst for two weeks.",
        "symptoms": ["fatigue", "thirst"],
        "labs": {},
    }

    combined = build_combined_text(report)

    assert "Clinical text: Fatigue and increased thirst for two weeks." in combined
    assert "Symptoms: fatigue, thirst." in combined
