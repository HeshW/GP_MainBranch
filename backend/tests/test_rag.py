from models.diagnosis.rag import MedicalCaseSearcher


def test_rerank_results_prefers_feature_aligned_case():
    results = [
        {
            "similarity": 0.74,
            "pathology": "Myocarditis",
            "case_text": (
                "Patient: 34 year old M. Presenting symptoms: chest pain, shortness of breath, "
                "viral infection, fatigue."
            ),
        },
        {
            "similarity": 0.61,
            "pathology": "Diabetes",
            "case_text": (
                "Patient: 50 year old F. Presenting symptoms: fatigue, thirst, polyuria, "
                "weight loss."
            ),
        },
    ]

    reranked = MedicalCaseSearcher._rerank_results(
        results,
        query_text="Fatigue and increased thirst for two weeks.",
        query_symptoms=["fatigue", "thirst"],
    )

    assert reranked[0]["pathology"] == "Diabetes"


def test_feature_mismatch_penalty_detects_unrequested_cardiac_features():
    query_features = MedicalCaseSearcher._extract_feature_flags(
        "Fatigue and increased thirst for two weeks.",
        ["fatigue", "thirst"],
    )

    penalty = MedicalCaseSearcher._feature_mismatch_penalty(
        query_features,
        "Patient reports chest pain, shortness of breath, and recent viral infection with fatigue.",
    )

    assert penalty > 0
