import pytest

from manager.runtime import run_async
from manager.symptom_normalizer import build_normalized_symptom_text
from manager.symptom_parser import parse_symptoms
from manager.symptom_validator import validate_parsed
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
    assert "final_diagnosis" not in out
    assert out["differential_diagnosis"]
    assert any(finding["source"] == "symptom_rules" for finding in out["findings"])
    assert "uncertain" in out["summary"].lower()


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


def test_stroke_like_out_of_scope_does_not_confidently_become_gbs():
    class StubClassifier:
        def predict(self, text):
            return {
                "predicted_label": "Guillain-Barré syndrome",
                "confidence": 0.82,
                "top_predictions": [
                    {"label": "Guillain-Barré syndrome", "confidence": 0.82},
                    {"label": "Myasthenia gravis", "confidence": 0.58},
                ],
            }

    class StubRag:
        async def query(self, patient_text, top_k=5, query_symptoms=None):
            return {
                "response": "retrieval",
                "retrieved_cases": [
                    {"pathology": "Guillain-Barré syndrome", "rerank_score": 0.77},
                ],
                "rag_confidence": {
                    "level": "high",
                    "usable_for_fusion": True,
                    "scope_status": "supported_scope",
                },
                "rag_scope_status": "supported_scope",
                "detected_out_of_scope_signals": [],
            }

    engine = DiagnosisEngine()
    engine._classifier = StubClassifier()
    engine._rag_assistant = StubRag()
    out = run_async(
        engine.diagnose(
            {
                "raw_text": "Sudden facial droop, arm weakness, and trouble speaking.",
                "symptoms": ["weakness", "difficulty speaking"],
                "labs": {},
            }
        )
    )

    assert out["final_diagnosis"]["source"] == "safety_scope_gate"
    assert out["final_diagnosis"]["mode"] == "safe_fallback"
    assert out["final_diagnosis"]["confidence"] < 0.55
    assert "stroke_like_neurologic_emergency" in out["safety"]["unsupported_scope_signals"]
    assert out["safety"]["emergency_attention_recommended"] is True


def test_arabic_stable_angina_maps_to_cardiac_signals():
    parsed = parse_symptoms("عندي ألم وضغط في الصدر مع المجهود ويتحسن مع الراحة")
    validated = validate_parsed(parsed)
    normalized_text = build_normalized_symptom_text(parsed, validated)

    assert "chest pain" in normalized_text.lower()
    assert "chest pressure" in normalized_text.lower()
    assert "with exertion" in normalized_text.lower()
    assert "improves with rest" in normalized_text.lower()


def test_typo_noisy_pneumonia_maps_to_respiratory_signals():
    parsed = parse_symptoms("feevr and prodctive coff with shortnes of breth and pleuritic chest pain")
    validated = validate_parsed(parsed)

    assert "fever" in validated["symptoms"]
    assert "productive cough" in validated["symptoms"]
    assert "shortness of breath" in validated["symptoms"]
    assert "chest pain" in validated["symptoms"]


def test_pe_context_is_not_dominated_by_localized_edema():
    final = DiagnosisEngine._build_final_diagnosis(
        findings=[
            {
                "condition": "Localized edema",
                "confidence": "moderate",
                "source": "symptom_rules",
                "severity": "moderate",
            }
        ],
        patient_symptoms=["shortness of breath", "leg swelling"],
        report_text="Patient suddenly became short of breath after a long flight and one calf is swollen.",
        classifier_prediction={
            "predicted_label": "Pancreatic neoplasm",
            "confidence": 0.35,
            "top_predictions": [{"label": "Pancreatic neoplasm", "confidence": 0.35}],
        },
    )

    assert final is not None
    assert final["diagnosis"] == "Pulmonary embolism"


def test_dystonic_medication_spasm_context_is_recognized():
    final = DiagnosisEngine._build_final_diagnosis(
        findings=[],
        patient_symptoms=[],
        report_text="Involuntary neck twisting and muscle spasms after starting a new nausea medication.",
        classifier_prediction={
            "predicted_label": "Chagas",
            "confidence": 0.58,
            "top_predictions": [{"label": "Chagas", "confidence": 0.58}],
        },
    )

    assert final is not None
    assert final["diagnosis"] == "Acute dystonic reactions"


def test_confidence_is_reduced_when_classifier_and_rag_disagree():
    calibrated = DiagnosisEngine._calibrate_final_diagnosis(
        {
            "diagnosis": "Pneumonia",
            "confidence": 0.82,
            "source": "classifier",
            "mode": "ai_primary",
            "rule_alignment": False,
        },
        candidates=[
            {"label": "Pneumonia"},
            {"label": "Unstable angina"},
            {"label": "GERD"},
        ],
        findings=[],
        rag_out={
            "retrieved_cases": [{"pathology": "GERD", "rerank_score": 0.7}],
            "rag_confidence": {"usable_for_fusion": True, "level": "moderate"},
        },
        classifier_prediction={"predicted_label": "Pneumonia", "confidence": 0.82},
        report_text="uncertain symptoms",
        patient_symptoms=[],
        unsupported_scope_signals=[],
    )

    assert calibrated["confidence"] < 0.82
    assert calibrated["confidence_calibration"]["reductions"]


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


def test_out_of_scope_rag_status_is_not_used_for_fusion():
    candidates = DiagnosisEngine._collect_diagnostic_candidates(
        findings=[],
        patient_symptoms=["fatigue", "thirst", "polyuria"],
        report_text="fatigue thirst polyuria glucose",
        rag_out={
            "rag_scope_status": "out_of_scope_or_low_confidence",
            "rag_confidence": {
                "level": "low",
                "usable_for_fusion": False,
                "scope_status": "out_of_scope_or_low_confidence",
                "detected_out_of_scope_signals": ["diabetes_hyperglycemia"],
            },
            "retrieved_cases": [
                {
                    "pathology": "Pulmonary neoplasm",
                    "rerank_score": 0.62,
                }
            ],
        },
        classifier_prediction=None,
    )
    fusion = DiagnosisEngine._build_decision_fusion(
        [],
        rag_out={
            "rag_scope_status": "out_of_scope_or_low_confidence",
            "rag_confidence": {
                "level": "low",
                "usable_for_fusion": False,
                "scope_status": "out_of_scope_or_low_confidence",
            },
        },
        classifier_prediction=None,
        final_diagnosis=None,
    )

    assert not any(candidate["label"] == "Pulmonary neoplasm" for candidate in candidates)
    assert fusion["rag_scope_status"] == "out_of_scope_or_low_confidence"
    assert fusion["rag_usable_for_fusion"] is False


def test_rag_debug_metadata_only_when_rag_is_enabled():
    out = run_async(DiagnosisEngine().diagnose({"raw_text": "fatigue", "symptoms": ["fatigue"], "labs": {}}))

    assert "rag_metadata" not in out
    assert "retrieved_cases" not in out


def test_enabled_rag_includes_scope_debug_metadata():
    class StubRAG:
        async def query(self, patient_text, top_k=5, query_symptoms=None):
            return {
                "response": "retrieval only",
                "retrieved_cases": [
                    {
                        "pathology": "Pulmonary neoplasm",
                        "rerank_score": 0.1,
                        "symptom_overlap": 0.1,
                    }
                ],
                "rag_query_text": patient_text,
                "rag_mode": "retrieval_only",
                "rag_scope_status": "out_of_scope_or_low_confidence",
                "detected_out_of_scope_signals": ["diabetes_hyperglycemia"],
                "rag_confidence": {
                    "level": "low",
                    "usable_for_fusion": False,
                    "scope_status": "out_of_scope_or_low_confidence",
                    "detected_out_of_scope_signals": ["diabetes_hyperglycemia"],
                },
            }

    engine = DiagnosisEngine()
    engine._rag_assistant = StubRAG()
    out = run_async(
        engine.diagnose(
            {
                "raw_text": "fatigue thirst polyuria",
                "symptoms": ["fatigue", "thirst", "polyuria"],
                "labs": {},
            }
        )
    )

    assert out["rag_metadata"]["rag_scope_status"] == "out_of_scope_or_low_confidence"
    assert out["rag_metadata"]["usable_for_fusion"] is False
    assert out["rag_metadata"]["detected_out_of_scope_signals"] == ["diabetes_hyperglycemia"]
    assert out["decision_fusion"]["rag_usable_for_fusion"] is False
    assert "final_diagnosis" not in out
    assert any(
        "hyperglycemia" in item["label"].lower()
        for item in out["differential_diagnosis"]
    )


def test_diabetes_scope_signal_does_not_override_symptom_rule_after_clarification():
    class StubRAG:
        async def query(self, patient_text, top_k=5, query_symptoms=None):
            return {
                "response": "retrieval only",
                "retrieved_cases": [
                    {
                        "pathology": "Myocarditis",
                        "rerank_score": 0.1,
                        "symptom_overlap": 0.1,
                    }
                ],
                "rag_query_text": patient_text,
                "rag_mode": "retrieval_only",
                "rag_scope_status": "out_of_scope_or_low_confidence",
                "detected_out_of_scope_signals": ["diabetes_hyperglycemia"],
                "rag_confidence": {
                    "level": "low",
                    "usable_for_fusion": False,
                    "scope_status": "out_of_scope_or_low_confidence",
                    "detected_out_of_scope_signals": ["diabetes_hyperglycemia"],
                },
            }

    engine = DiagnosisEngine()
    engine._rag_assistant = StubRAG()
    out = run_async(
        engine.diagnose(
            {
                "raw_text": (
                    "Patient-reported complaint: I have increased thirst and feel fatigue for over 2 weeks.\n"
                    "Patient reports: fatigue, increased thirst\n"
                    "Follow-up clarification: no chest discomfort, just increased thirst and urination"
                ),
                "symptoms": ["fatigue", "thirst", "polyuria"],
                "labs": {},
                "follow_up_answers": ["no chest discomfort, just increased thirst and urination"],
            }
        )
    )

    assert "final_diagnosis" not in out
    assert any(
        "hyperglycemia" in item["label"].lower()
        for item in out["differential_diagnosis"]
    )
    assert out["safety"]["unsupported_scope_signals"] == ["diabetes_hyperglycemia"]


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


def test_serious_respiratory_output_requires_follow_up_reasons():
    reasons = DiagnosisEngine._clarification_reasons(
        findings=[
            {
                "condition": "Possible lower respiratory infection pattern",
                "confidence": "low",
                "evidence": "Respiratory symptom cluster detected.",
                "severity": "high",
                "source": "symptom_rules",
            }
        ],
        patient_symptoms=["cough", "shortness of breath"],
        candidates=[
            {"label": "Pneumonia", "confidence": 0.86, "sources": ["respiratory_pattern_expansion"]},
            {"label": "Bronchitis", "confidence": 0.64, "sources": ["respiratory_pattern_expansion"]},
        ],
        final_diagnosis={
            "diagnosis": "Pneumonia",
            "confidence": 0.86,
            "source": "respiratory_pattern_expansion",
            "rule_alignment": True,
        },
    )

    assert any("Serious or high-risk diagnosis" in reason for reason in reasons)


def test_upper_respiratory_context_adds_common_uri_candidates():
    expanded = DiagnosisEngine._expand_base_diagnostic_candidates(
        [
            {
                "label": "Possible lower respiratory infection pattern",
                "confidence": 0.45,
                "sources": ["symptom_rules"],
                "reasoning": "rule",
                "evidence": ["cluster"],
            }
        ],
        report_text="sore throat with nasal congestion and cough after a recent cold",
    )

    labels = {item["label"] for item in expanded}
    assert "URTI" in labels or "Viral pharyngitis" in labels


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


def test_build_combined_text_accepts_sections_list_from_ocr_output():
    report = {
        "raw_text": "Chest discomfort and dyspnea.",
        "labs": {},
        "sections": [
            {"label": "Clinical", "text": "Dyspnea and chest pain"},
            {"label": "Diagnosis", "text": "Consider cardiopulmonary differential"},
            {"label": "Clinical", "text": "Progressive over two days"},
        ],
    }

    combined = build_combined_text(report)

    assert "Clinical: Dyspnea and chest pain" in combined
    assert "Progressive over two days" in combined
    assert "Diagnosis: Consider cardiopulmonary differential" in combined


def test_diagnose_arabic_input_localizes_summary_and_safety_text():
    out = run_async(
        diagnose(
            {
                "raw_text": "ما عندي كحة ولا حرارة لكن عندي دوخة وتعب من يومين",
                "symptoms": ["dizziness", "fatigue"],
                "labs": {},
            }
        )
    )

    assert out["response_language"] == "ar"
    assert "التقييم" in out["summary"]
    assert any(("يوصى" in reason or "تم تصنيف" in reason) for reason in out["safety"]["reasons"])


def test_diagnose_keeps_arabic_response_language_with_mixed_wrapper_text():
    out = run_async(
        diagnose(
            {
                "raw_text": "Patient-reported complaint: ما عندي كحة ولا حرارة لكن عندي دوخة",
                "symptoms": ["dizziness", "fatigue"],
                "labs": {},
            }
        )
    )

    assert out["response_language"] == "ar"
