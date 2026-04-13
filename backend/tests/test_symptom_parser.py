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
