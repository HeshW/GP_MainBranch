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
    assert "clarification" not in out


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


def test_classifier_is_preferred_over_retrieval_when_supportive():
    final = DiagnosisEngine._build_final_diagnosis(
        findings=[],
        patient_symptoms=["chest pain", "shortness of breath"],
        rag_out={
            "retrieved_cases": [
                {
                    "pathology": "Pericarditis",
                    "similarity": 0.43,
                }
            ]
        },
        classifier_prediction={
            "predicted_label": "Atrial fibrillation",
            "confidence": 0.41,
            "top_predictions": [
                {"label": "Atrial fibrillation", "confidence": 0.41},
                {"label": "Pericarditis", "confidence": 0.37},
            ],
        },
    )

    assert final is not None
    assert final["diagnosis"] == "Atrial fibrillation"
    assert final["source"] == "classifier"


def test_build_final_diagnosis_prefers_pulmonary_embolism_with_strong_pe_context():
    final = DiagnosisEngine._build_final_diagnosis(
        findings=[
            {
                "condition": "Possible cardiopulmonary red-flag symptom pattern",
                "confidence": "low",
                "evidence": "chest pain + dyspnea red-flag pattern",
                "severity": "moderate",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["chest pain", "shortness of breath"],
        report_text=(
            "Sudden pleuritic chest pain with shortness of breath after recent immobility "
            "and unilateral leg swelling."
        ),
        rag_out={
            "retrieved_cases": [
                {"pathology": "Myocarditis", "similarity": 0.77},
                {"pathology": "Pericarditis", "similarity": 0.71},
            ]
        },
        classifier_prediction={
            "predicted_label": "Myocarditis",
            "confidence": 0.46,
            "top_predictions": [
                {"label": "Myocarditis", "confidence": 0.46},
                {"label": "Pericarditis", "confidence": 0.38},
            ],
        },
    )

    assert final is not None
    assert final["diagnosis"] == "Pulmonary embolism"


def test_collect_diagnostic_candidates_prioritizes_respiratory_context_over_neuro_noise():
    candidates = DiagnosisEngine._collect_diagnostic_candidates(
        findings=[
            {
                "condition": "Possible lower respiratory infection pattern",
                "confidence": "low",
                "evidence": "respiratory symptom cluster",
                "severity": "moderate",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["wheezing", "cough", "chest tightness"],
        report_text="Wheezing and chest tightness without fever or productive cough.",
        rag_out={
            "retrieved_cases": [
                {"pathology": "Myasthenia gravis", "similarity": 0.76},
                {"pathology": "Bronchitis", "similarity": 0.72},
            ]
        },
        classifier_prediction={
            "predicted_label": "Bronchitis",
            "confidence": 0.39,
            "top_predictions": [
                {"label": "Bronchitis", "confidence": 0.39},
                {"label": "Bronchospasm / acute asthma exacerbation", "confidence": 0.35},
            ],
        },
    )

    assert candidates
    assert candidates[0]["label"] == "Bronchospasm / acute asthma exacerbation"


def test_build_final_diagnosis_prefers_af_over_psvt_when_irregular_context_present():
    final = DiagnosisEngine._build_final_diagnosis(
        findings=[],
        patient_symptoms=["palpitations"],
        report_text="Palpitations are irregular and uneven throughout the episode.",
        rag_out={
            "retrieved_cases": [
                {"pathology": "PSVT", "similarity": 0.43},
                {"pathology": "Atrial fibrillation", "similarity": 0.53},
            ]
        },
        classifier_prediction={
            "predicted_label": "PSVT",
            "confidence": 0.41,
            "top_predictions": [
                {"label": "PSVT", "confidence": 0.41},
                {"label": "Atrial fibrillation", "confidence": 0.34},
            ],
        },
    )

    assert final is not None
    assert final["diagnosis"] == "Atrial fibrillation"


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


def test_uncertain_ai_case_returns_disease_targeted_follow_up_questions(monkeypatch):
    class StubClassifier:
        def __init__(self, model_dir, max_length=256, device=None):
            pass

        def predict(self, text):
            return {
                "predicted_label": "Pericarditis",
                "confidence": 0.42,
                "top_predictions": [
                    {"label": "Pericarditis", "confidence": 0.42},
                    {"label": "Pulmonary embolism", "confidence": 0.37},
                    {"label": "Atrial fibrillation", "confidence": 0.31},
                ],
            }

    monkeypatch.setattr("models.diagnosis.diagnosisengine.FineTunedDiagnosisClassifier", StubClassifier)

    engine = DiagnosisEngine(
        use_finetuned_classifier=True,
        finetuned_model_dir="fake-model-dir",
    )
    out = run_async(
        engine.diagnose(
            {
                "raw_text": "chest pain shortness of breath",
                "symptoms": ["chest pain", "shortness of breath"],
                "labs": {},
            }
        )
    )

    clarification = out.get("clarification")
    assert clarification is not None
    assert clarification["needed"] is True
    assert clarification["questions"]
    assert clarification["candidate_diseases"]
    joined_questions = " ".join(item["question"] for item in clarification["questions"]).lower()
    assert any(term in joined_questions for term in ["breathing", "palpitations", "exertion", "heartbeat"])


def test_build_clarification_uses_suspected_diseases_for_questions():
    clarification = DiagnosisEngine._build_clarification(
        report={"raw_text": "reflux chest discomfort", "labs": {}, "symptoms": ["reflux"]},
        findings=[],
        patient_symptoms=["reflux"],
        candidates=[
            {
                "label": "GERD",
                "confidence": 0.61,
                "sources": ["classifier"],
                "reasoning": "Classifier",
                "evidence": ["Classifier candidate: GERD"],
                "rule_alignment": False,
            },
            {
                "label": "Larygospasm",
                "confidence": 0.56,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Larygospasm"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "GERD",
            "confidence": 0.61,
            "source": "classifier",
            "rule_alignment": False,
        },
    )

    assert clarification is not None
    questions = clarification["questions"]
    assert questions
    assert any("GERD" in item["target_conditions"] for item in questions)
    assert any("Larygospasm" in item["target_conditions"] for item in questions)


def test_apply_follow_up_scoring_can_promote_target_candidate():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Pericarditis",
                "confidence": 0.42,
                "source": "classifier",
            },
            "diagnostic_candidates": [
                {"label": "Pericarditis", "confidence": 0.42, "sources": ["classifier"]},
                {"label": "Atrial fibrillation", "confidence": 0.37, "sources": ["classifier"]},
            ],
            "clarification": {
                "needed": True,
            },
        },
        answers=[
            "There is irregular heartbeat with palpitations and this sounds like atrial fibrillation."
        ],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {"label": "Pericarditis", "confidence": 0.42, "sources": ["classifier"]},
                    {"label": "Atrial fibrillation", "confidence": 0.37, "sources": ["classifier"]},
                ],
                "questions": [
                    {
                        "question": "Is the chest discomfort related to exertion, deep breathing, or an irregular heartbeat/palpitations?",
                        "target_conditions": ["Pericarditis", "Atrial fibrillation"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Atrial fibrillation"
    assert updated["final_diagnosis"]["source"] == "clarification_rerank"


def test_apply_follow_up_scoring_marks_clarification_complete_after_answers():
    answer = "There is irregular heartbeat with palpitations."
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Pericarditis",
                "confidence": 0.42,
                "source": "classifier",
            },
            "diagnostic_candidates": [
                {"label": "Pericarditis", "confidence": 0.42, "sources": ["classifier"]},
                {"label": "Atrial fibrillation", "confidence": 0.37, "sources": ["classifier"]},
            ],
            "clarification": {
                "needed": True,
                "questions": [
                    {
                        "question": "Is the chest discomfort related to exertion, deep breathing, or an irregular heartbeat/palpitations?",
                        "target_conditions": ["Pericarditis", "Atrial fibrillation"],
                    }
                ],
            },
        },
        answers=[answer],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {"label": "Pericarditis", "confidence": 0.42, "sources": ["classifier"]},
                    {"label": "Atrial fibrillation", "confidence": 0.37, "sources": ["classifier"]},
                ],
                "questions": [
                    {
                        "question": "Is the chest discomfort related to exertion, deep breathing, or an irregular heartbeat/palpitations?",
                        "target_conditions": ["Pericarditis", "Atrial fibrillation"],
                    }
                ],
            }
        },
    )

    clarification = updated["clarification"]
    assert clarification["applied"] is True
    assert clarification["completed"] is True
    assert clarification["needed"] is False
    assert clarification["answers_used"] == [answer]


def test_apply_follow_up_scoring_penalizes_psvt_when_irregular_heartbeat_is_reported():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "PSVT",
                "confidence": 0.45,
                "source": "classifier",
            },
            "diagnostic_candidates": [
                {"label": "PSVT", "confidence": 0.45, "sources": ["classifier"]},
                {"label": "Atrial fibrillation", "confidence": 0.38, "sources": ["classifier"]},
            ],
        },
        answers=[
            "The heartbeat feels irregular rather than just fast and this sounds like atrial fibrillation."
        ],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {"label": "PSVT", "confidence": 0.45, "sources": ["classifier"]},
                    {"label": "Atrial fibrillation", "confidence": 0.38, "sources": ["classifier"]},
                ],
                "questions": [
                    {
                        "question": "Is the chest discomfort related to exertion, deep breathing, or an irregular heartbeat/palpitations?",
                        "target_conditions": ["PSVT", "Atrial fibrillation"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Atrial fibrillation"


def test_apply_follow_up_scoring_promotes_pneumonia_over_bronchospasm():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Bronchospasm / acute asthma exacerbation",
                "confidence": 0.44,
                "source": "classifier_rag_consensus",
            },
            "diagnostic_candidates": [
                {
                    "label": "Bronchospasm / acute asthma exacerbation",
                    "confidence": 0.44,
                    "sources": ["classifier_rag_consensus"],
                },
                {"label": "Pneumonia", "confidence": 0.34, "sources": ["rag_retrieval"]},
            ],
        },
        answers=[
            "There is fever with productive cough and pleuritic chest pain, so this feels like infection rather than isolated wheezing."
        ],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {
                        "label": "Bronchospasm / acute asthma exacerbation",
                        "confidence": 0.44,
                        "sources": ["classifier_rag_consensus"],
                    },
                    {"label": "Pneumonia", "confidence": 0.34, "sources": ["rag_retrieval"]},
                ],
                "questions": [
                    {
                        "question": "Do you also have fever, cough, sore throat, or nasal congestion?",
                        "target_conditions": ["Pneumonia", "Bronchospasm / acute asthma exacerbation"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Pneumonia"


def test_apply_follow_up_scoring_demotes_bronchitis_when_pleuritic_infective_features_present():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Bronchitis",
                "confidence": 0.71,
                "source": "rag_retrieval",
            },
            "diagnostic_candidates": [
                {
                    "label": "Bronchitis",
                    "confidence": 0.71,
                    "sources": ["rag_retrieval"],
                },
                {
                    "label": "Pneumonia",
                    "confidence": 0.42,
                    "sources": ["clarification_expansion"],
                },
            ],
        },
        answers=[
            "There is fever with productive cough and pleuritic chest pain, which sounds more like pneumonia than bronchitis."
        ],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {
                        "label": "Bronchitis",
                        "confidence": 0.71,
                        "sources": ["rag_retrieval"],
                    },
                    {
                        "label": "Pneumonia",
                        "confidence": 0.42,
                        "sources": ["clarification_expansion"],
                    },
                ],
                "questions": [
                    {
                        "question": "Do you also have fever, cough, sore throat, or nasal congestion?",
                        "target_conditions": ["Bronchitis", "Pneumonia"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Pneumonia"


def test_apply_follow_up_scoring_promotes_pulmonary_embolism_over_myocarditis_for_pe_signals():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Myocarditis",
                "confidence": 0.58,
                "source": "rules_fallback",
            },
            "diagnostic_candidates": [
                {
                    "label": "Myocarditis",
                    "confidence": 0.77,
                    "sources": ["rag_retrieval"],
                },
                {
                    "label": "Pulmonary embolism",
                    "confidence": 0.43,
                    "sources": ["cardiopulmonary_pattern_expansion"],
                },
                {
                    "label": "Pericarditis",
                    "confidence": 0.40,
                    "sources": ["classifier"],
                },
            ],
        },
        answers=[
            "The pain is pleuritic, started suddenly, and there was recent immobility with unilateral leg swelling."
        ],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {
                        "label": "Myocarditis",
                        "confidence": 0.77,
                        "sources": ["rag_retrieval"],
                    },
                    {
                        "label": "Pulmonary embolism",
                        "confidence": 0.43,
                        "sources": ["cardiopulmonary_pattern_expansion"],
                    },
                ],
                "questions": [
                    {
                        "question": "Did the shortness of breath start suddenly, or was there recent immobility, leg swelling, or chest pain that worsens with breathing?",
                        "target_conditions": ["Pulmonary embolism", "Myocarditis"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Pulmonary embolism"


def test_is_negated_signal_detects_longer_phrase_between_negation_and_signal():
    assert DiagnosisEngine._is_negated_signal(
        "This does not feel like a mild infection or gradual fatigue.",
        "fatigue",
    )


def test_apply_follow_up_scoring_does_not_promote_anemia_from_negated_fatigue_phrase():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Myocarditis",
                "confidence": 0.58,
                "source": "rules_fallback",
            },
            "diagnostic_candidates": [
                {
                    "label": "Myocarditis",
                    "confidence": 0.77,
                    "sources": ["rag_retrieval"],
                },
                {
                    "label": "Pulmonary embolism",
                    "confidence": 0.43,
                    "sources": ["cardiopulmonary_pattern_expansion"],
                },
                {
                    "label": "Anemia",
                    "confidence": 0.34,
                    "sources": ["symptom_rules"],
                },
            ],
        },
        answers=[
            "The chest pain is pleuritic and sudden, with recent immobility and leg swelling. This does not feel like a mild infection or gradual fatigue."
        ],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {
                        "label": "Myocarditis",
                        "confidence": 0.77,
                        "sources": ["rag_retrieval"],
                    },
                    {
                        "label": "Pulmonary embolism",
                        "confidence": 0.43,
                        "sources": ["cardiopulmonary_pattern_expansion"],
                    },
                    {
                        "label": "Anemia",
                        "confidence": 0.34,
                        "sources": ["symptom_rules"],
                    },
                ],
                "questions": [
                    {
                        "question": "Did the shortness of breath start suddenly, or was there recent immobility, leg swelling, or chest pain that worsens with breathing?",
                        "target_conditions": ["Pulmonary embolism", "Myocarditis", "Anemia"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] != "Anemia"


def test_apply_follow_up_scoring_promotes_stable_angina_over_arrhythmia_when_answer_is_explicit():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Atrial fibrillation",
                "confidence": 0.88,
                "source": "classifier_rag_consensus",
            },
            "diagnostic_candidates": [
                {
                    "label": "Atrial fibrillation",
                    "confidence": 0.98,
                    "sources": ["classifier", "rag_retrieval"],
                },
                {
                    "label": "Myocarditis",
                    "confidence": 0.87,
                    "sources": ["classifier", "rag_retrieval"],
                },
                {
                    "label": "Stable angina",
                    "confidence": 0.80,
                    "sources": ["cardiopulmonary_pattern_expansion"],
                },
                {
                    "label": "PSVT",
                    "confidence": 0.70,
                    "sources": ["classifier", "ambiguity_pair_expansion"],
                },
            ],
        },
        answers=[
            "The pain happens with exertion and improves with rest. This fits stable angina more than arrhythmia."
        ],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {
                        "label": "Atrial fibrillation",
                        "confidence": 0.80,
                        "sources": ["classifier", "rag_retrieval"],
                    },
                    {
                        "label": "Stable angina",
                        "confidence": 0.80,
                        "sources": ["cardiopulmonary_pattern_expansion"],
                    },
                ],
                "questions": [
                    {
                        "question": "Is the chest discomfort related to exertion, deep breathing, or an irregular heartbeat/palpitations?",
                        "target_conditions": ["Stable angina", "Atrial fibrillation"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Stable angina"


def test_build_clarification_adds_pair_question_for_af_vs_psvt():
    clarification = DiagnosisEngine._build_clarification(
        report={"raw_text": "episodes of palpitations and dizziness", "labs": {}, "symptoms": ["palpitations", "dizziness"]},
        findings=[],
        patient_symptoms=["palpitations", "dizziness"],
        candidates=[
            {
                "label": "Atrial fibrillation",
                "confidence": 0.49,
                "sources": ["classifier"],
                "reasoning": "Classifier",
                "evidence": ["Classifier candidate: Atrial fibrillation"],
                "rule_alignment": False,
            },
            {
                "label": "PSVT",
                "confidence": 0.46,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: PSVT"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "Atrial fibrillation",
            "confidence": 0.49,
            "source": "classifier",
            "rule_alignment": False,
        },
    )

    assert clarification is not None
    assert clarification["questions"]
    first_question = clarification["questions"][0]
    assert set(first_question["target_conditions"]) == {"Atrial fibrillation", "PSVT"}
    assert any(term in first_question["question"].lower() for term in ["irregular", "sudden", "abrupt"])


def test_build_clarification_promotes_af_counterpart_when_psvt_leads():
    clarification = DiagnosisEngine._build_clarification(
        report={"raw_text": "palpitations with irregular rhythm", "labs": {}, "symptoms": ["palpitations"]},
        findings=[],
        patient_symptoms=["palpitations"],
        candidates=[
            {
                "label": "PSVT",
                "confidence": 0.83,
                "sources": ["classifier"],
                "reasoning": "Classifier",
                "evidence": ["Classifier candidate: PSVT"],
                "rule_alignment": False,
            },
            {
                "label": "Guillain-Barré syndrome",
                "confidence": 0.71,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Guillain-Barré syndrome"],
                "rule_alignment": False,
            },
            {
                "label": "Atrial fibrillation",
                "confidence": 0.04,
                "sources": ["classifier"],
                "reasoning": "Classifier",
                "evidence": ["Classifier candidate: Atrial fibrillation"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "PSVT",
            "confidence": 0.83,
            "source": "classifier",
            "rule_alignment": False,
        },
    )

    assert clarification is not None
    labels = [item["label"] for item in clarification["candidate_diseases"]]
    assert "Atrial fibrillation" in labels


def test_build_clarification_expands_lower_respiratory_pattern_with_pneumonia():
    clarification = DiagnosisEngine._build_clarification(
        report={"raw_text": "fever productive cough pleuritic chest pain", "labs": {}, "symptoms": ["fever", "cough", "chest pain"]},
        findings=[],
        patient_symptoms=["fever", "cough", "chest pain"],
        candidates=[
            {
                "label": "Bronchitis",
                "confidence": 0.71,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Bronchitis"],
                "rule_alignment": False,
            },
            {
                "label": "Possible lower respiratory infection pattern",
                "confidence": 0.45,
                "sources": ["symptom_rules"],
                "reasoning": "Rules",
                "evidence": ["Respiratory symptom cluster detected"],
                "rule_alignment": True,
            },
            {
                "label": "Pericarditis",
                "confidence": 0.15,
                "sources": ["classifier"],
                "reasoning": "Classifier",
                "evidence": ["Classifier candidate: Pericarditis"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "Bronchitis",
            "confidence": 0.71,
            "source": "rag_retrieval",
            "rule_alignment": False,
        },
    )

    assert clarification is not None
    labels = [item["label"] for item in clarification["candidate_diseases"]]
    assert "Pneumonia" in labels


def test_clarification_reasons_skip_high_confidence_respiratory_bronchitis_case():
    reasons = DiagnosisEngine._clarification_reasons(
        findings=[
            {
                "condition": "Possible lower respiratory infection pattern",
                "confidence": "low",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["cough", "shortness of breath"],
        candidates=[
            {
                "label": "Bronchitis",
                "confidence": 0.82,
                "sources": ["respiratory_pattern_expansion"],
            },
            {
                "label": "Pneumonia",
                "confidence": 0.81,
                "sources": ["respiratory_pattern_expansion"],
            },
        ],
        final_diagnosis={
            "diagnosis": "Bronchitis",
            "confidence": 0.82,
            "source": "respiratory_pattern_expansion",
            "rule_alignment": True,
        },
    )

    assert reasons == []


def test_build_clarification_adds_sarcoidosis_for_chronic_dry_cough_context():
    clarification = DiagnosisEngine._build_clarification(
        report={
            "raw_text": "I have dry cough with gradual and persistent shortness of breath, and this does not feel like acute infection.",
            "labs": {},
            "symptoms": ["dry cough", "shortness of breath"],
        },
        findings=[
            {
                "condition": "Possible lower respiratory infection pattern",
                "confidence": "low",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["dry cough", "shortness of breath"],
        candidates=[
            {
                "label": "Bronchospasm / acute asthma exacerbation",
                "confidence": 0.79,
                "sources": ["respiratory_pattern_expansion", "respiratory_context_rebalance"],
                "reasoning": "Rules",
                "evidence": ["Lower respiratory pattern expansion"],
                "rule_alignment": False,
            },
            {
                "label": "Bronchitis",
                "confidence": 0.75,
                "sources": ["respiratory_pattern_expansion", "respiratory_context_rebalance"],
                "reasoning": "Rules",
                "evidence": ["Lower respiratory pattern expansion"],
                "rule_alignment": False,
            },
            {
                "label": "Pneumonia",
                "confidence": 0.75,
                "sources": ["respiratory_pattern_expansion", "respiratory_context_rebalance"],
                "reasoning": "Rules",
                "evidence": ["Lower respiratory pattern expansion"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "Myocarditis",
            "confidence": 0.81,
            "source": "classifier_rag_consensus",
            "rule_alignment": False,
        },
    )

    assert clarification is not None
    labels = [item["label"] for item in clarification["candidate_diseases"]]
    assert "Sarcoidosis" in labels


def test_build_clarification_adds_stable_angina_for_exertional_rest_pattern():
    clarification = DiagnosisEngine._build_clarification(
        report={
            "raw_text": "Chest pressure appears with exertion and improves with rest.",
            "labs": {},
            "symptoms": ["chest pain", "shortness of breath"],
        },
        findings=[
            {
                "condition": "Possible cardiopulmonary red-flag symptom pattern",
                "confidence": "low",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["chest pain", "shortness of breath"],
        candidates=[
            {
                "label": "Atrial fibrillation",
                "confidence": 0.84,
                "sources": ["classifier"],
                "reasoning": "Classifier",
                "evidence": ["Classifier candidate: Atrial fibrillation"],
                "rule_alignment": False,
            },
            {
                "label": "PSVT",
                "confidence": 0.79,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: PSVT"],
                "rule_alignment": False,
            },
            {
                "label": "Larygospasm",
                "confidence": 0.74,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Larygospasm"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "Atrial fibrillation",
            "confidence": 0.84,
            "source": "classifier",
            "rule_alignment": False,
        },
    )

    assert clarification is not None
    labels = [item["label"] for item in clarification["candidate_diseases"]]
    assert "Stable angina" in labels


def test_build_clarification_prioritizes_respiratory_shortlist_over_neuro_noise():
    clarification = DiagnosisEngine._build_clarification(
        report={
            "raw_text": "wheezing cough shortness of breath chest tightness",
            "labs": {},
            "symptoms": ["wheezing", "cough", "shortness of breath"],
        },
        findings=[
            {
                "condition": "Possible lower respiratory infection pattern",
                "confidence": "low",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["wheezing", "cough", "shortness of breath"],
        candidates=[
            {
                "label": "Myasthenia gravis",
                "confidence": 0.76,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Myasthenia gravis"],
                "rule_alignment": False,
            },
            {
                "label": "Guillain-Barré syndrome",
                "confidence": 0.76,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Guillain-Barré syndrome"],
                "rule_alignment": False,
            },
            {
                "label": "Possible lower respiratory infection pattern",
                "confidence": 0.45,
                "sources": ["symptom_rules"],
                "reasoning": "Rules",
                "evidence": ["Respiratory symptom cluster detected"],
                "rule_alignment": True,
            },
        ],
        final_diagnosis={
            "diagnosis": "Bronchitis",
            "confidence": 0.58,
            "source": "rules_fallback",
            "rule_alignment": True,
        },
    )

    assert clarification is not None
    labels = [item["label"] for item in clarification["candidate_diseases"]]
    assert any(label in labels for label in ["Pneumonia", "Bronchospasm / acute asthma exacerbation", "Bronchitis"])
    first_question_targets = set(clarification["questions"][0].get("target_conditions", []))
    assert first_question_targets != {"Myasthenia gravis", "Guillain-Barré syndrome"}


def test_build_clarification_prefers_pneumonia_vs_bronchospasm_pair_when_available():
    clarification = DiagnosisEngine._build_clarification(
        report={
            "raw_text": "wheezing with chest tightness and cough",
            "labs": {},
            "symptoms": ["wheezing", "cough", "chest tightness"],
        },
        findings=[
            {
                "condition": "Possible lower respiratory infection pattern",
                "confidence": "low",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["wheezing", "cough", "chest tightness"],
        candidates=[
            {
                "label": "Bronchitis",
                "confidence": 0.71,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Bronchitis"],
                "rule_alignment": False,
            },
            {
                "label": "Possible lower respiratory infection pattern",
                "confidence": 0.45,
                "sources": ["symptom_rules"],
                "reasoning": "Rules",
                "evidence": ["Respiratory symptom cluster detected"],
                "rule_alignment": True,
            },
            {
                "label": "Myasthenia gravis",
                "confidence": 0.76,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Myasthenia gravis"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "Bronchitis",
            "confidence": 0.58,
            "source": "rules_fallback",
            "rule_alignment": True,
        },
    )

    assert clarification is not None
    first_question_targets = set(clarification["questions"][0].get("target_conditions", []))
    assert first_question_targets == {"Pneumonia", "Bronchospasm / acute asthma exacerbation"}


def test_build_clarification_expands_cardiopulmonary_pattern_with_pulmonary_embolism():
    clarification = DiagnosisEngine._build_clarification(
        report={
            "raw_text": "sudden chest pain with shortness of breath",
            "labs": {},
            "symptoms": ["chest pain", "shortness of breath"],
        },
        findings=[
            {
                "condition": "Possible cardiopulmonary red-flag symptom pattern",
                "confidence": "low",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["chest pain", "shortness of breath"],
        candidates=[
            {
                "label": "Myocarditis",
                "confidence": 0.77,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Myocarditis"],
                "rule_alignment": False,
            },
            {
                "label": "Possible cardiopulmonary red-flag symptom pattern",
                "confidence": 0.45,
                "sources": ["symptom_rules"],
                "reasoning": "Rules",
                "evidence": ["Red-flag symptoms detected"],
                "rule_alignment": True,
            },
            {
                "label": "Pericarditis",
                "confidence": 0.40,
                "sources": ["classifier"],
                "reasoning": "Classifier",
                "evidence": ["Classifier candidate: Pericarditis"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "Myocarditis",
            "confidence": 0.58,
            "source": "rules_fallback",
            "rule_alignment": True,
        },
    )

    assert clarification is not None
    labels = [item["label"] for item in clarification["candidate_diseases"]]
    assert "Pulmonary embolism" in labels


def test_build_clarification_adds_spontaneous_pneumothorax_for_sudden_one_sided_pain():
    clarification = DiagnosisEngine._build_clarification(
        report={
            "raw_text": "Sudden one-sided sharp chest pain with acute shortness of breath.",
            "labs": {},
            "symptoms": ["chest pain", "shortness of breath"],
        },
        findings=[],
        patient_symptoms=["chest pain", "shortness of breath"],
        candidates=[
            {
                "label": "Myocarditis",
                "confidence": 0.77,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Myocarditis"],
                "rule_alignment": False,
            },
            {
                "label": "Pericarditis",
                "confidence": 0.71,
                "sources": ["classifier"],
                "reasoning": "Classifier",
                "evidence": ["Classifier candidate: Pericarditis"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "Myocarditis",
            "confidence": 0.58,
            "source": "rag_retrieval",
            "rule_alignment": False,
        },
    )

    assert clarification is not None
    labels = [item["label"] for item in clarification["candidate_diseases"]]
    assert "Spontaneous pneumothorax" in labels


def test_apply_follow_up_scoring_promotes_spontaneous_pneumothorax_for_unilateral_sudden_dyspnea():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Myocarditis",
                "confidence": 0.58,
                "source": "rules_fallback",
            },
            "diagnostic_candidates": [
                {
                    "label": "Myocarditis",
                    "confidence": 0.77,
                    "sources": ["rag_retrieval"],
                },
                {
                    "label": "Spontaneous pneumothorax",
                    "confidence": 0.69,
                    "sources": ["cardiopulmonary_pattern_expansion"],
                },
                {
                    "label": "Pericarditis",
                    "confidence": 0.66,
                    "sources": ["classifier"],
                },
            ],
        },
        answers=[
            "The onset was sudden with one-sided sharp chest pain and acute shortness of breath, with no productive cough or fever."
        ],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {
                        "label": "Myocarditis",
                        "confidence": 0.77,
                        "sources": ["rag_retrieval"],
                    },
                    {
                        "label": "Spontaneous pneumothorax",
                        "confidence": 0.69,
                        "sources": ["cardiopulmonary_pattern_expansion"],
                    },
                ],
                "questions": [
                    {
                        "question": "Was the chest pain sudden and one-sided with acute breathing difficulty?",
                        "target_conditions": ["Spontaneous pneumothorax", "Myocarditis"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Spontaneous pneumothorax"


def test_apply_follow_up_scoring_respects_negated_infection_signals():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Bronchospasm / acute asthma exacerbation",
                "confidence": 0.44,
                "source": "classifier",
            },
            "diagnostic_candidates": [
                {
                    "label": "Bronchospasm / acute asthma exacerbation",
                    "confidence": 0.44,
                    "sources": ["classifier"],
                },
                {"label": "Pneumonia", "confidence": 0.42, "sources": ["rag_retrieval"]},
            ],
        },
        answers=[
            "No fever and no productive cough. Main issue is wheezing with chest tightness."
        ],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {
                        "label": "Bronchospasm / acute asthma exacerbation",
                        "confidence": 0.44,
                        "sources": ["classifier"],
                    },
                    {"label": "Pneumonia", "confidence": 0.42, "sources": ["rag_retrieval"]},
                ],
                "questions": [
                    {
                        "question": "Do you have clear infection signs (fever with productive cough), or mostly wheeze/chest tightness without infection features?",
                        "target_conditions": ["Pneumonia", "Bronchospasm / acute asthma exacerbation"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Bronchospasm / acute asthma exacerbation"


def test_apply_follow_up_scoring_keeps_label_for_low_information_answer():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Pericarditis",
                "confidence": 0.44,
                "source": "classifier",
            },
            "diagnostic_candidates": [
                {"label": "Pericarditis", "confidence": 0.44, "sources": ["classifier"]},
                {"label": "Pulmonary embolism", "confidence": 0.42, "sources": ["rag_retrieval"]},
            ],
        },
        answers=["I am not sure, it just feels bad."],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {"label": "Pericarditis", "confidence": 0.44, "sources": ["classifier"]},
                    {"label": "Pulmonary embolism", "confidence": 0.42, "sources": ["rag_retrieval"]},
                ],
                "questions": [
                    {
                        "question": "Is the chest discomfort related to exertion, deep breathing, or an irregular heartbeat/palpitations?",
                        "target_conditions": ["Pericarditis", "Pulmonary embolism"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Pericarditis"


def test_apply_follow_up_scoring_uses_comparative_preference_phrasing():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "PSVT",
                "confidence": 0.65,
                "source": "classifier",
            },
            "diagnostic_candidates": [
                {"label": "PSVT", "confidence": 0.65, "sources": ["classifier"]},
                {"label": "Atrial fibrillation", "confidence": 0.24, "sources": ["classifier"]},
            ],
        },
        answers=["This feels more like atrial fibrillation than PSVT."],
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {"label": "PSVT", "confidence": 0.65, "sources": ["classifier"]},
                    {"label": "Atrial fibrillation", "confidence": 0.24, "sources": ["classifier"]},
                ],
                "questions": [
                    {
                        "question": "Are the palpitations irregular and uneven, or mostly sudden fast episodes that start and stop abruptly?",
                        "target_conditions": ["Atrial fibrillation", "PSVT"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Atrial fibrillation"


def test_apply_label_canonicalization_maps_reflux_pattern_to_gerd():
    payload = {
        "final_diagnosis": {
            "diagnosis": "Possible gastroesophageal reflux pattern",
            "confidence": 0.58,
            "source": "rules_fallback",
            "reasoning": "Rules fallback selected.",
        },
        "diagnostic_candidates": [
            {
                "label": "Possible gastroesophageal reflux pattern",
                "confidence": 0.58,
                "sources": ["symptom_rules"],
            }
        ],
    }

    updated = DiagnosisEngine._apply_label_canonicalization(payload)

    assert updated["final_diagnosis"]["diagnosis"] == "GERD"
    assert updated["final_diagnosis"]["canonicalized_from"] == "Possible gastroesophageal reflux pattern"


def test_apply_label_canonicalization_maps_lower_respiratory_pattern_using_candidates():
    payload = {
        "final_diagnosis": {
            "diagnosis": "Possible lower respiratory infection pattern",
            "confidence": 0.58,
            "source": "rules_fallback",
            "reasoning": "Rules fallback selected.",
        },
        "diagnostic_candidates": [
            {
                "label": "Possible lower respiratory infection pattern",
                "confidence": 0.58,
                "sources": ["symptom_rules"],
            },
            {
                "label": "Pneumonia",
                "confidence": 0.52,
                "sources": ["classifier"],
            },
            {
                "label": "Bronchitis",
                "confidence": 0.44,
                "sources": ["rag_retrieval"],
            },
        ],
    }

    updated = DiagnosisEngine._apply_label_canonicalization(payload)

    assert updated["final_diagnosis"]["diagnosis"] == "Pneumonia"
    assert updated["final_diagnosis"]["canonicalized_from"] == "Possible lower respiratory infection pattern"


def test_apply_label_canonicalization_keeps_hyperglycemia_pattern_without_supported_mapping():
    payload = {
        "final_diagnosis": {
            "diagnosis": "Possible hyperglycemia / diabetes symptom pattern",
            "confidence": 0.58,
            "source": "rules_fallback",
            "reasoning": "Rules fallback selected.",
        },
        "diagnostic_candidates": [
            {
                "label": "Possible hyperglycemia / diabetes symptom pattern",
                "confidence": 0.58,
                "sources": ["symptom_rules"],
            },
            {
                "label": "Myocarditis",
                "confidence": 0.54,
                "sources": ["rag_retrieval"],
            },
        ],
    }

    updated = DiagnosisEngine._apply_label_canonicalization(payload)

    assert updated["final_diagnosis"]["diagnosis"] == "Possible hyperglycemia / diabetes symptom pattern"
    assert "canonicalized_from" not in updated["final_diagnosis"]


def test_build_clarification_returns_arabic_questions_for_arabic_input():
    clarification = DiagnosisEngine._build_clarification(
        report={"raw_text": "عندي ألم صدر وخفقان", "labs": {}, "symptoms": ["chest pain", "palpitations"]},
        findings=[],
        patient_symptoms=["chest pain", "palpitations"],
        candidates=[
            {
                "label": "Atrial fibrillation",
                "confidence": 0.51,
                "sources": ["classifier"],
                "reasoning": "Classifier",
                "evidence": ["Classifier candidate: Atrial fibrillation"],
                "rule_alignment": False,
            },
            {
                "label": "Pericarditis",
                "confidence": 0.47,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Pericarditis"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "Atrial fibrillation",
            "confidence": 0.51,
            "source": "classifier",
            "rule_alignment": False,
        },
    )

    assert clarification is not None
    assert clarification["questions"]
    assert any("هل" in item["question"] for item in clarification["questions"])


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


def test_apply_follow_up_scoring_uses_normalized_follow_up_text_for_mixed_language_answers():
    updated = DiagnosisEngine.apply_follow_up_scoring(
        {
            "final_diagnosis": {
                "diagnosis": "Bronchospasm / acute asthma exacerbation",
                "confidence": 0.57,
                "source": "classifier",
            },
            "diagnostic_candidates": [
                {
                    "label": "Bronchospasm / acute asthma exacerbation",
                    "confidence": 0.57,
                    "sources": ["classifier"],
                },
                {
                    "label": "Pneumonia",
                    "confidence": 0.56,
                    "sources": ["rag_retrieval"],
                },
            ],
        },
        answers=["نعم"],
        normalized_follow_up_text="Patient reports: fever, productive cough, pleuritic chest pain.",
        prior_diagnosis={
            "clarification": {
                "candidate_diseases": [
                    {
                        "label": "Bronchospasm / acute asthma exacerbation",
                        "confidence": 0.57,
                        "sources": ["classifier"],
                    },
                    {
                        "label": "Pneumonia",
                        "confidence": 0.56,
                        "sources": ["rag_retrieval"],
                    },
                ],
                "questions": [
                    {
                        "question": "Do you have clear infection signs (fever with productive cough), or mostly wheeze/chest tightness without infection features?",
                        "target_conditions": ["Pneumonia", "Bronchospasm / acute asthma exacerbation"],
                    }
                ],
            }
        },
    )

    assert updated["final_diagnosis"]["diagnosis"] == "Pneumonia"


def test_build_clarification_uses_single_question_when_candidate_margin_is_high():
    clarification = DiagnosisEngine._build_clarification(
        report={
            "raw_text": "Chest discomfort with exertion and shortness of breath.",
            "labs": {},
            "symptoms": ["chest pain", "shortness of breath"],
        },
        findings=[
            {
                "condition": "Possible cardiopulmonary red-flag symptom pattern",
                "confidence": "low",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["chest pain", "shortness of breath"],
        candidates=[
            {
                "label": "Stable angina",
                "confidence": 0.89,
                "sources": ["classifier"],
                "reasoning": "Classifier",
                "evidence": ["Classifier candidate: Stable angina"],
                "rule_alignment": False,
            },
            {
                "label": "Atrial fibrillation",
                "confidence": 0.60,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Atrial fibrillation"],
                "rule_alignment": False,
            },
            {
                "label": "Pericarditis",
                "confidence": 0.41,
                "sources": ["rag_retrieval"],
                "reasoning": "RAG",
                "evidence": ["Top retrieved case pathology: Pericarditis"],
                "rule_alignment": False,
            },
        ],
        final_diagnosis={
            "diagnosis": "Stable angina",
            "confidence": 0.69,
            "source": "classifier",
            "rule_alignment": False,
        },
    )

    assert clarification is not None
    assert len(clarification["questions"]) == 1
