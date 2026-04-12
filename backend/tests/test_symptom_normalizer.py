from manager.symptom_normalizer import build_normalized_symptom_text


def test_build_normalized_symptom_text_includes_raw_symptoms_and_duration():
    parsed = {
        "raw_text": "Fatigue and increased thirst for two weeks.",
        "symptoms": [
            {"symptom": "fatigue", "source": "fatigue", "confidence": 0.85},
            {"symptom": "thirst", "source": "increased thirst", "confidence": 0.85},
        ],
    }
    validated = {
        "symptoms": ["fatigue", "thirst"],
    }

    normalized = build_normalized_symptom_text(parsed, validated)

    assert "Patient-reported complaint: Fatigue and increased thirst for two weeks." in normalized
    assert "Patient reports: fatigue, increased thirst" in normalized
    assert "Normalized symptoms: fatigue, thirst" in normalized
    assert "Do you feel more thirsty than usual?" in normalized
    assert "Duration: for two weeks" in normalized


def test_build_normalized_symptom_text_includes_exertion_rest_context():
    parsed = {
        "raw_text": "Chest pain with exertion that improves with rest.",
        "symptoms": [
            {"symptom": "chest pain", "source": "chest pain", "confidence": 0.85},
        ],
    }
    validated = {
        "symptoms": ["chest pain"],
    }

    normalized = build_normalized_symptom_text(parsed, validated)

    assert "with exertion" in normalized
    assert "improves with rest" in normalized
