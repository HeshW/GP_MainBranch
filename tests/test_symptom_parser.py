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
