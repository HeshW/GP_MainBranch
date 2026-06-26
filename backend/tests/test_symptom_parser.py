from manager.symptom_parser import parse_symptoms
from manager.symptom_normalizer import build_normalized_symptom_text
from manager.symptom_validator import validate_parsed


def test_parse_symptoms_basic_lab_and_symptom():
    text = "Patient with fatigue, petechial rash, glucose 145 mg/dL and hemoglobin: 10.2 g/dL"
    parsed = parse_symptoms(text)

    assert parsed["raw_text"] == text
    assert "glucose" in parsed["labs"]
    assert parsed["labs"]["glucose"]["value"] == 145.0
    assert parsed["labs"]["glucose"]["unit"] == "mg/dl" or parsed["labs"]["glucose"]["unit"] is None
    assert "hemoglobin" in parsed["labs"]
    assert any(s["symptom"] == "fatigue" for s in parsed["symptoms"])


def test_parse_symptoms_negation():
    text = "No fever, denies cough, but has chest pain and dizziness."
    parsed = parse_symptoms(text)
    assert any(sym.get("negated") for sym in parsed["symptoms"])


def test_parse_symptoms_detects_broader_human_medical_phrases():
    text = (
        "Patient reports palpitations, wheezing, chest tightness, sore throat, "
        "hoarseness, reflux, and shortness of breath."
    )
    parsed = parse_symptoms(text)
    found = {sym["symptom"] for sym in parsed["symptoms"]}

    assert "palpitations" in found
    assert "wheezing" in found
    assert "chest tightness" in found
    assert "sore throat" in found
    assert "hoarseness" in found
    assert "reflux" in found
    assert "shortness of breath" in found


def test_parse_symptoms_supports_arabic_and_mixed_language():
    text = "عندي ألم صدر وخفقان وضيق تنفس منذ يومين بعد الأكل"
    parsed = parse_symptoms(text)
    found = {sym["symptom"] for sym in parsed["symptoms"] if not sym.get("negated")}

    assert "chest pain" in found
    assert "palpitations" in found
    assert "shortness of breath" in found
    assert "duration" in parsed["context"]
    assert "triggers" in parsed["context"]


def test_parse_symptoms_negation_handles_longer_scope_window():
    text = "I have fatigue and dizziness, but I do not currently have any fever at this time."
    parsed = parse_symptoms(text)

    negated = {sym["symptom"] for sym in parsed["symptoms"] if sym.get("negated")}
    assert "fever" in negated


def test_parse_symptoms_negation_handles_arabic_variants():
    text = "ما عندي كحة ولا حرارة لكن عندي دوخة"
    parsed = parse_symptoms(text)

    negated = {sym["symptom"] for sym in parsed["symptoms"] if sym.get("negated")}
    positive = {sym["symptom"] for sym in parsed["symptoms"] if not sym.get("negated")}

    assert "cough" in negated
    assert "fever" in negated
    assert "dizziness" in positive


def test_parse_symptoms_qualified_cough_negation_does_not_cancel_plain_cough():
    text = "No productive cough, but I still have cough and fever."
    parsed = parse_symptoms(text)

    positive = {sym["symptom"] for sym in parsed["symptoms"] if not sym.get("negated")}
    assert "cough" in positive


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


def test_validate_parsed_can_support_lab_conversion():
    parsed = {
        "labs": {
            "glucose": {"value": 180, "unit": "mg/dl", "source": "glucose 180 mg/dL", "confidence": 0.8},
            "unknown": {"value": 5.1, "unit": "", "source": "unknown 5.1", "confidence": 0.3},
        },
        "symptoms": [{"symptom": "fatigue", "source": "fatigue", "confidence": 0.8}],
    }
    validated = validate_parsed(parsed, low_confidence_threshold=0.5)

    assert validated["labs"]["glucose"]["value"] == 180.0
    assert validated["labs"]["glucose"]["unit"] in ("mg/dL", "mg/dl") or validated["labs"]["glucose"]["unit"] == "mg/dL"
    assert validated["symptoms"] == ["fatigue"]
    assert validated["confidence"]["glucose"] == 0.8
    assert validated["review_required"] is True


def test_validate_parsed_no_data_triggers_review():
    parsed = {"labs": {}, "symptoms": []}
    validated = validate_parsed(parsed)
    assert validated["review_required"] is True
    assert "No symptoms or lab data" in validated["warnings"][0]
