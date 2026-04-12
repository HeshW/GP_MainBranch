from manager.symptom_parser import parse_symptoms


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


def test_parse_symptoms_strips_encoded_presenting_symptoms_tail():
    text = (
        "Patient reports cough and fever for two days. "
        "Presenting symptoms: 53, 54 @ V 161, 54 @ V 180, 55 @ V 101, "
        "55 @ V 103, 56 @ 4, 57 @ V 29, 58 @ 8, 59 @ 3, 210, 215, 217."
    )
    parsed = parse_symptoms(text)

    assert "Presenting symptoms" not in parsed["raw_text"]
    assert "raw_text_original" in parsed
    assert any(item["symptom"] == "cough" for item in parsed["symptoms"])
    assert any(item["symptom"] == "fever" for item in parsed["symptoms"])


def test_parse_symptoms_detects_exertion_and_rest_relief_context():
    parsed = parse_symptoms("Chest pain with exertion that improves with rest.")

    triggers = parsed.get("context", {}).get("triggers", [])
    assert "with exertion" in triggers
    assert "improves with rest" in triggers


def test_parse_symptoms_handles_extended_arabic_negation_cues():
    parsed = parse_symptoms("ما في حرارة ولا يوجد كحة لكن عندي ضيق تنفس.")

    negated = {item["symptom"] for item in parsed["symptoms"] if item.get("negated")}
    positives = {item["symptom"] for item in parsed["symptoms"] if not item.get("negated")}

    assert "fever" in negated
    assert "cough" in negated
    assert "shortness of breath" in positives
