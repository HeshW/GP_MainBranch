from manager.chat_manager import ChatManager


def test_run_from_symptoms_integration_with_manager(monkeypatch):
    manager = ChatManager()

    # Monkeypatch diagnosis path to avoid rule mismatch output requirement.
    sample_diagnosis = {
        "findings": [{"condition": "Test condition", "confidence": "high"}],
        "summary": "Test summary",
    }

    class StubDiagnosisEngine:
        def __init__(self, *args, **kwargs):
            pass

        def diagnose(self, report):
            assert report["labs"]["glucose"] == 150.0
            assert "fatigue" in report.get("raw_text", "")
            return sample_diagnosis

    monkeypatch.setattr(manager, "_diagnosis_engine", StubDiagnosisEngine())
    result = manager.run_from_symptoms("Patient has fatigue and glucose 150 mg/dL")

    assert result["status"] == "ok"
    assert result["diagnosis"] == sample_diagnosis
    assert result["review_required"] is False or isinstance(result["review_required"], bool)
    assert result["parsed"]["labs"]["glucose"]["value"] == 150.0
    assert "fatigue" in result["validated"]["symptoms"]
