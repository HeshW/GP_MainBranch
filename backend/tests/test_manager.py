import pytest

from manager.chat_manager import ChatManager


def test_run_pipeline_from_labs():
    manager = ChatManager()
    payload = {"glucose": 130.0, "hemoglobin": 10.5}

    result = manager.run_pipeline(labs=payload)

    assert result["status"] == "ok"
    assert result["ocr"] is None
    assert "diagnosis" in result
    findings = result["diagnosis"].get("findings", [])
    assert isinstance(findings, list)
    assert any(f.get("condition") in ["Diabetes Mellitus (suspected)", "Mild Anemia"] for f in findings)


def test_run_pipeline_from_image(monkeypatch):
    manager = ChatManager()

    fake_ocr = {
        "labs": {"glucose": {"value": 145.0, "unit": "mg/dL"}},
        "raw_text": "Glucose 145 mg/dL",
    }

    monkeypatch.setattr(manager, "run_ocr", lambda image: fake_ocr)

    result = manager.run_pipeline(image="fake-path.png")

    assert result["status"] == "ok"
    assert result["ocr"] is not None
    assert result["ocr"]["labs"]["glucose"]["value"] == 145.0
    findings = result["diagnosis"]["findings"]
    assert any(f.get("condition") == "Diabetes Mellitus (suspected)" for f in findings)


def test_run_pipeline_manual_symptoms(monkeypatch):
    manager = ChatManager()

    # Add a small symptom parser by injecting manual labs path
    manual_input = {
        "symptoms": "I feel tired and weak",
        "labs": {"hemoglobin": 9.2},
    }

    result = manager.run_pipeline(manual_input=manual_input)

    assert result["status"] == "ok"
    assert result["diagnosis"] is not None
    findings = result["diagnosis"].get("findings", [])
    assert any(f.get("condition") == "Moderate Anemia" for f in findings)


def test_run_diagnosis_adapter_delegation(monkeypatch):
    manager = ChatManager()
    ocr_result = {
        "labs": {"glucose": {"value": 70.0, "unit": "mg/dL"}},
        "raw_text": "Glucose 70 mg/dL",
    }

    monkeypatch.setattr(manager, "run_ocr", lambda image: ocr_result)
    result = manager.run_pipeline(image="anything.png")

    assert result["status"] == "ok"
    assert result["ocr"] == ocr_result
    assert result["diagnosis"]["summary"] is not None


def test_run_from_labs_via_adapter():
    from manager.diagnosis_adapter import run_from_labs

    result = run_from_labs({"glucose": 132.0})

    assert result["status"] == "ok"
    assert result["diagnosis"]["findings"]


def test_run_from_image_via_adapter(monkeypatch):
    from manager.diagnosis_adapter import run_from_image

    class FakeManager(ChatManager):
        def run_pipeline(self, **kwargs):
            return {
                "status": "ok",
                "ocr": {"labs": {"glucose": {"value": 145.0}}},
                "diagnosis": {"findings": [{"condition": "Diabetes Mellitus (suspected)"}]},
            }

    monkeypatch.setattr("manager.diagnosis_adapter.ChatManager", FakeManager)

    result = run_from_image("dummy.png")
    assert result["status"] == "ok"
    assert result["diagnosis"]["findings"][0]["condition"] == "Diabetes Mellitus (suspected)"


def test_rag_disabled_by_default():
    from manager.chat_manager import ChatManager

    manager = ChatManager(use_rag=False)
    result = manager.run_pipeline(labs={"glucose": 102.0})
    assert result["status"] == "ok"
    assert "rag_response" not in result["diagnosis"]
    assert "retrieved_cases" not in result["diagnosis"]


def test_manual_input_only_path():
    from manager.chat_manager import ChatManager

    manager = ChatManager()
    result = manager.run_pipeline(manual_input={"symptoms": "I feel weak", "labs": {"hemoglobin": 9.8}})
    assert result["status"] == "ok"
    assert result["diagnosis"]["findings"]
