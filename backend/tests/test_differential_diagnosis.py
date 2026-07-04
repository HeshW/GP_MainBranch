from manager.runtime import run_async
from models.diagnosis.diagnosisengine import DiagnosisEngine, diagnose


def _labels(result):
    return [item["label"].lower() for item in result.get("differential_diagnosis", [])]


def test_vague_symptoms_do_not_promote_dangerous_final_diagnosis():
    result = run_async(
        diagnose(
            {
                "symptoms": ["fatigue", "thirst"],
                "raw_text": "I feel tired and thirsty.",
                "labs": {},
            }
        )
    )

    assert "final_diagnosis" not in result
    labels = _labels(result)
    assert labels
    assert not any(
        danger in labels[0]
        for danger in ("myocarditis", "pulmonary embolism", "neoplasm", "angina")
    )
    assert any("dehydration" in label or "hyperglycemia" in label for label in labels[:3])


def test_myocarditis_is_differential_not_final_with_partial_support():
    result = run_async(
        diagnose(
            {
                "symptoms": ["chest pain", "fatigue", "viral prodrome"],
                "raw_text": "Patient reports chest pain and fatigue after a recent viral illness.",
                "labs": {},
            }
        )
    )

    assert "final_diagnosis" not in result
    myocarditis = [
        item for item in result.get("differential_diagnosis", [])
        if "myocarditis" in item["label"].lower()
    ]
    assert myocarditis
    assert myocarditis[0]["urgency"] == "urgent"
    assert myocarditis[0]["missing_evidence"]


def test_sudden_dyspnea_leg_swelling_escalates_pe_differential():
    result = run_async(
        diagnose(
            {
                "symptoms": ["shortness of breath", "chest pain", "leg swelling"],
                "raw_text": "Sudden shortness of breath with pleuritic chest pain after a long flight and one calf is swollen.",
                "labs": {},
            }
        )
    )

    pe = [
        item for item in result.get("differential_diagnosis", [])
        if "pulmonary embolism" in item["label"].lower()
    ]
    assert pe
    assert pe[0]["urgency"] == "emergency"
    assert result["safety"]["clinician_review_required"] is True


def test_negated_symptoms_count_as_evidence_against_ddx_candidate():
    result = run_async(
        diagnose(
            {
                "symptoms": ["chest pain"],
                "raw_text": "Chest pain, but no shortness of breath and no leg swelling.",
                "labs": {},
            }
        )
    )

    pe = [
        item for item in result.get("differential_diagnosis", [])
        if "pulmonary embolism" in item["label"].lower()
    ]
    assert pe
    assert any("shortness of breath" in item for item in pe[0]["evidence_against"])
    assert any("leg swelling" in item for item in pe[0]["evidence_against"])


def test_stroke_like_symptoms_trigger_safety_without_gbs_final():
    class StubClassifier:
        def predict(self, text):
            return {
                "predicted_label": "Guillain-Barré syndrome",
                "confidence": 0.82,
                "top_predictions": [{"label": "Guillain-Barré syndrome", "confidence": 0.82}],
            }

    engine = DiagnosisEngine()
    engine._classifier = StubClassifier()
    result = run_async(
        engine.diagnose(
            {
                "raw_text": "Sudden facial droop, arm weakness, and trouble speaking.",
                "symptoms": ["weakness", "difficulty speaking"],
                "labs": {},
            }
        )
    )

    assert result["final_diagnosis"]["source"] == "safety_scope_gate"
    assert "stroke_like_neurologic_emergency" in result["safety"]["unsupported_scope_signals"]
    assert result["safety"]["emergency_attention_recommended"] is True
