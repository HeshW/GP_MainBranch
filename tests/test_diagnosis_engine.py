import pytest

from models.diagnosis.diagnosisengine import diagnose, DiagnosisEngine


def test_rule_based_detects_diabetes():
    report = {"labs": {"glucose": {"value": 130, "unit": "mg/dL"}}}
    out = diagnose(report)
    conditions = [f["condition"] for f in out["findings"]]
    assert any("Diabetes" in c or "diabetes" in c for c in conditions)


def test_accepts_plain_float_value():
    report = {"labs": {"glucose": 140}}
    out = diagnose(report)
    assert out["findings"]


def test_no_labs_returns_empty_summary():
    report = {}
    out = diagnose(report)
    assert "No clinically significant" in out["summary"]


def test_diagnose_raises_on_invalid_report_type():
    engine = DiagnosisEngine()
    with pytest.raises(TypeError):
        engine.diagnose(None)
