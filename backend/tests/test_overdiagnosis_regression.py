"""
Tests for the diagnosis engine to prevent over-diagnosis of rare conditions.
"""
import pytest
from backend.models.diagnosis.diagnosisengine import DiagnosisEngine
from backend.manager.runtime import run_async

# Basic engine configuration for tests
engine = DiagnosisEngine(
    use_rag=False,
    use_finetuned_classifier=False,
)

def test_vague_symptoms_do_not_overdiagnose_rare_conditions():
    """
    Vague Symptom Benchmark:
    Given a patient with vague symptoms like "I feel tired and thirsty",
    the system should NOT rank a rare disease like myocarditis as the top diagnosis.
    It should instead favor common conditions like dehydration.
    """
    report = {
        "symptoms": ["fatigue", "thirst"],
        "raw_text": "I feel tired and thirsty",
    }
    result = run_async(engine.diagnose(report))
    
    assert "diagnostic_candidates" in result
    candidates = result["diagnostic_candidates"]
    
    assert len(candidates) > 0, "Should have at least one candidate"

    # Metric 1 (Alarmism Rate): Assert that for this input, myocarditis is NOT the top-ranked diagnosis.
    top_diagnosis_label = candidates[0]["label"]
    assert "myocarditis" not in top_diagnosis_label.lower(), f"Myocarditis should not be the top diagnosis for vague symptoms. Got: {top_diagnosis_label}"

    # Metric 2 (Top-3 DDx Accuracy): Assert that Dehydration or an acute viral illness IS present in the top 3.
    top_3_labels = [c["label"].lower() for c in candidates[:3]]
    
    dehydration_present = any("dehydration" in label for label in top_3_labels)
    urti_present = any("urti" in label for label in top_3_labels)
    
    assert dehydration_present or urti_present, f"Dehydration or URTI should be in the top 3 diagnoses for 'tired and thirsty'. Got: {top_3_labels}"
    
    # Also check that dehydration is ranked higher than myocarditis if both are present
    dehydration_rank = -1
    myocarditis_rank = -1
    for i, candidate in enumerate(candidates):
        label = candidate["label"].lower()
        if "dehydration" in label:
            dehydration_rank = i
        if "myocarditis" in label:
            myocarditis_rank = i
            
    if myocarditis_rank != -1 and dehydration_rank != -1:
        assert dehydration_rank < myocarditis_rank, "Dehydration should be ranked higher than myocarditis for these symptoms."

def test_myocarditis_can_be_diagnosed_with_right_symptoms():
    """
    Required Feature Gating Test:
    Given a patient with classic myocarditis symptoms like "chest pain and fatigue after a viral illness",
    the system should be able to diagnose myocarditis.
    This ensures we haven't lost sensitivity for legitimate presentations.
    """
    report = {
        "symptoms": ["chest pain", "fatigue", "viral prodrome"],
        "raw_text": "Patient reports chest pain and fatigue after a recent viral illness.",
    }
    result = run_async(engine.diagnose(report))
    
    assert "diagnostic_candidates" in result
    candidates = result["diagnostic_candidates"]
    
    assert len(candidates) > 0, "Should have at least one candidate"
    
    # Assert that myocarditis CAN appear high in the DDx in this case
    myocarditis_present_in_top_3 = any("myocarditis" in c["label"].lower() for c in candidates[:3])
    
    assert myocarditis_present_in_top_3, f"Myocarditis should be a top candidate for 'chest pain after viral illness'. Got: {[c['label'] for c in candidates[:3]]}"

