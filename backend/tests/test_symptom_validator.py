from manager.symptom_validator import validate_parsed


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
