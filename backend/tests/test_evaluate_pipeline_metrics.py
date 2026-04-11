from manager.runtime import run_async
from scripts.evaluate_pipeline_end_to_end import clinically_matches, evaluate_case, is_in_supported_scope


def test_clinically_matches_diabetes_family_rule_output():
    assert clinically_matches(
        "Diabetes Mellitus (suspected)",
        "Possible hyperglycemia / diabetes symptom pattern",
    )


def test_clinically_matches_anemia_family_variants():
    assert clinically_matches("Anemia", "Moderate Anemia")


def test_is_in_supported_scope_uses_normalized_labels():
    supported = {"Viral pharyngitis", "GERD"}
    assert is_in_supported_scope("viral pharyngitis", supported)
    assert not is_in_supported_scope("Diabetes Mellitus (suspected)", supported)


def test_evaluate_case_tracks_post_clarification_predictions():
    class FakeManager:
        async def run_from_symptoms(self, text):
            return {
                "report": {
                    "raw_text": text,
                    "symptoms": ["chest pain"],
                    "labs": {},
                },
                "diagnosis": {
                    "final_diagnosis": {
                        "diagnosis": "Pericarditis",
                    },
                    "clarification": {
                        "needed": True,
                        "candidate_diseases": [
                            {"label": "Pericarditis"},
                            {"label": "Atrial fibrillation"},
                        ],
                    },
                },
                "therapy": {"therapy_plan": "placeholder"},
            }

        async def run_clarification(self, report, answers, prior_diagnosis=None):
            return {
                "report": report,
                "diagnosis": {
                    "final_diagnosis": {
                        "diagnosis": "Atrial fibrillation",
                    }
                },
                "therapy": {"therapy_plan": "placeholder"},
            }

    detail = run_async(
        evaluate_case(
            FakeManager(),
            {
                "raw_text": "chest pain",
                "expected_conditions": ["Atrial fibrillation"],
                "follow_up_answers": ["There are palpitations and irregular heartbeat."],
            },
            1,
            supported_labels={"Atrial fibrillation", "Pericarditis"},
        )
    )

    assert detail["clarification_needed"] is True
    assert detail["clarification_applied"] is True
    assert detail["clarification_top_1_prediction"] == "Atrial fibrillation"
    assert detail["clarification_top_1_correct"] is True
