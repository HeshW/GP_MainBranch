import pytest

from manager.chat_support import get_stream_error_message, get_unavailable_message
from manager.chat_manager import ChatManager
from manager.runtime import run_async


async def _collect_stream_chunks(stream):
    return [chunk async for chunk in stream]


def test_run_pipeline_from_labs():
    manager = ChatManager()
    payload = {"glucose": 130.0, "hemoglobin": 10.5}

    result = run_async(manager.run_pipeline(labs=payload))

    assert result["status"] == "ok"
    assert result["ocr"] is None
    assert "diagnosis" in result
    findings = result["diagnosis"].get("findings", [])
    assert isinstance(findings, list)
    assert any(
        finding.get("condition") in ["Diabetes Mellitus (suspected)", "Mild Anemia"]
        for finding in findings
    )


def test_run_pipeline_from_image(monkeypatch):
    manager = ChatManager()

    fake_ocr = {
        "labs": {"glucose": {"value": 145.0, "unit": "mg/dL"}},
        "raw_text": "Glucose 145 mg/dL",
    }

    monkeypatch.setattr(manager, "run_ocr", lambda image: fake_ocr)
    result = run_async(manager.run_pipeline(image="fake-path.png"))

    assert result["status"] == "ok"
    assert result["ocr"] is not None
    assert result["ocr"]["labs"]["glucose"]["value"] == 145.0
    findings = result["diagnosis"]["findings"]
    assert any(finding.get("condition") == "Diabetes Mellitus (suspected)" for finding in findings)


def test_run_pipeline_manual_symptoms():
    manager = ChatManager()
    manual_input = {
        "symptoms": "I feel tired and weak",
        "labs": {"hemoglobin": 9.2},
    }

    result = run_async(manager.run_pipeline(manual_input=manual_input))

    assert result["status"] == "ok"
    assert result["diagnosis"] is not None
    findings = result["diagnosis"].get("findings", [])
    assert any(finding.get("condition") == "Moderate Anemia" for finding in findings)


def test_run_diagnosis_adapter_delegation(monkeypatch):
    manager = ChatManager()
    ocr_result = {
        "labs": {"glucose": {"value": 70.0, "unit": "mg/dL"}},
        "raw_text": "Glucose 70 mg/dL",
    }

    monkeypatch.setattr(manager, "run_ocr", lambda image: ocr_result)
    result = run_async(manager.run_pipeline(image="anything.png"))

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
    manager = ChatManager(use_rag=False)
    result = run_async(manager.run_pipeline(labs={"glucose": 102.0}))
    assert result["status"] == "ok"
    assert "rag_response" not in result["diagnosis"]
    assert "retrieved_cases" not in result["diagnosis"]


def test_manual_input_only_path():
    manager = ChatManager()
    result = run_async(
        manager.run_pipeline(manual_input={"symptoms": "I feel weak", "labs": {"hemoglobin": 9.8}})
    )
    assert result["status"] == "ok"
    assert result["diagnosis"]["findings"]


def test_run_diagnosis_only_returns_report_and_diagnosis():
    manager = ChatManager()
    report = {"raw_text": "fatigue thirst", "symptoms": ["fatigue", "thirst"], "labs": {}}

    result = run_async(manager.run_diagnosis_only(report))

    assert result["status"] == "ok"
    assert result["report"] == report
    assert "diagnosis" in result
    assert "therapy" in result
    assert result["therapy"]["metadata"]["mode"] == "disabled"
    assert result["therapy"]["metadata"]["enabled"] is False


def test_run_diagnosis_only_can_enable_therapy_generation(monkeypatch):
    manager = ChatManager(enable_therapy=True)

    async def stub_run_diagnosis(report):
        return {"findings": [{"condition": "Hypertension", "severity": "moderate"}]}

    class StubTherapyEngine:
        async def generate_therapy(self, diagnosis, patient_info):
            return {
                "therapy_plan": "stub therapy",
                "structured_therapy": None,
                "metadata": {"mode": "llm", "enabled": True},
            }

    monkeypatch.setattr(manager, "run_diagnosis", stub_run_diagnosis)
    monkeypatch.setattr(manager, "_therapy_engine", StubTherapyEngine())

    result = run_async(manager.run_diagnosis_only({"raw_text": "headache", "symptoms": ["headache"], "labs": {}}))

    assert result["therapy"]["metadata"]["mode"] == "llm"


def test_run_from_symptoms_preserves_original_raw_text():
    manager = ChatManager()

    result = run_async(manager.run_from_symptoms("Fatigue and increased thirst for two weeks."))

    assert result["parsed"]["raw_text"] == "Fatigue and increased thirst for two weeks."
    assert "Patient-reported complaint" in result["validated"]["raw_text"]
    assert "Duration: for two weeks" in result["validated"]["raw_text"]
    assert result["normalized_text"] == result["validated"]["raw_text"]


def test_run_from_symptoms_excludes_negated_symptoms():
    manager = ChatManager()

    result = run_async(manager.run_from_symptoms("I have fatigue but no fever and no cough."))

    assert "fatigue" in result["validated"]["symptoms"]
    assert "fever" not in result["validated"]["symptoms"]
    assert "cough" not in result["validated"]["symptoms"]
    assert "fever" in result["validated"]["negated_symptoms"]


def test_run_from_symptoms_keeps_plain_cough_when_only_qualified_cough_is_negated():
    manager = ChatManager()

    result = run_async(manager.run_from_symptoms("No productive cough, but I have cough and fever."))

    assert "cough" in result["validated"]["symptoms"]


def test_run_from_symptoms_arabic_localizes_diagnosis_summary():
    manager = ChatManager()

    result = run_async(manager.run_from_symptoms("ما عندي كحة ولا حرارة لكن عندي دوخة وتعب من يومين"))

    assert result["diagnosis"]["response_language"] == "ar"
    assert "التقييم" in result["diagnosis"]["summary"]


def test_run_clarification_merges_follow_up_answers_and_re_diagnoses(monkeypatch):
    manager = ChatManager()

    class StubDiagnosisEngine:
        def apply_follow_up_scoring(self, diagnosis, *, answers, prior_diagnosis=None):
            diagnosis["final_diagnosis"]["diagnosis"] = "Atrial fibrillation"
            diagnosis["final_diagnosis"]["source"] = "clarification_rerank"
            return diagnosis

        async def diagnose(self, report):
            raw_text = str(report.get("raw_text", "")).lower()
            symptoms = [str(item).lower() for item in report.get("symptoms", [])]
            if "palpitations" in raw_text or "palpitations" in symptoms:
                return {
                    "findings": [],
                    "summary": "AI-assisted assessment suggests Atrial fibrillation.",
                    "final_diagnosis": {
                        "diagnosis": "Atrial fibrillation",
                        "confidence": 0.88,
                        "source": "classifier",
                        "mode": "ai_primary",
                    },
                    "clarification": None,
                }
            return {
                "findings": [],
                "summary": "The first-pass assessment is still uncertain.",
                "final_diagnosis": {
                    "diagnosis": "Pericarditis",
                    "confidence": 0.42,
                    "source": "classifier",
                    "mode": "ai_primary",
                },
                "clarification": {
                    "needed": True,
                    "questions": [
                        {
                            "question": "Is the chest discomfort related to exertion, deep breathing, or an irregular heartbeat/palpitations?",
                            "target_conditions": ["Pericarditis", "Atrial fibrillation"],
                        }
                    ],
                },
            }

    monkeypatch.setattr(manager, "_diagnosis_engine", StubDiagnosisEngine())

    initial_report = {
        "raw_text": "Chest pain and shortness of breath",
        "symptoms": ["chest pain", "shortness of breath"],
        "labs": {},
    }

    result = run_async(
        manager.run_clarification(
            initial_report,
            ["The pain comes with palpitations and an irregular heartbeat."],
        )
    )

    assert result["diagnosis"]["final_diagnosis"]["diagnosis"] == "Atrial fibrillation"
    assert "Follow-up clarification" in result["report"]["raw_text"]
    assert "palpitations" in result["report"]["symptoms"]
    assert result["follow_up"]["answers"]


def test_run_ocr_reuses_single_engine_instance(monkeypatch):
    manager = ChatManager()
    init_count = 0

    class FakeOCREngine:
        def __init__(self):
            nonlocal init_count
            init_count += 1

        def extract(self, image):
            return {"raw_text": str(image), "labs": {}}

    monkeypatch.setattr("models.ocr.engine.OCREngine", FakeOCREngine)

    first = run_async(manager.run_ocr("first.png"))
    second = run_async(manager.run_ocr("second.png"))

    assert first["raw_text"] == "first.png"
    assert second["raw_text"] == "second.png"
    assert init_count == 1


def test_stream_chat_unavailable_response_is_persisted_in_session_history():
    manager = ChatManager(gemini_api_key="")

    chunks = run_async(_collect_stream_chunks(manager.stream_chat("s-unavailable", "hello")))
    expected = get_unavailable_message("hello")
    history = manager._chat_sessions.get("s-unavailable")

    assert chunks == [expected]
    assert history[-2] == {"role": "user", "content": "hello"}
    assert history[-1] == {"role": "model", "content": expected}


def test_stream_chat_error_chunk_is_persisted_in_session_history(monkeypatch):
    manager = ChatManager(gemini_api_key="local-key")

    class StubProvider:
        async def generate_stream(self, prompt, system_instruction=None):
            yield "partial "
            raise RuntimeError("stream failed")

    monkeypatch.setattr(manager, "_chat_provider", StubProvider())

    chunks = run_async(_collect_stream_chunks(manager.stream_chat("s-error", "hello")))
    expected_error = get_stream_error_message("hello")
    history = manager._chat_sessions.get("s-error")

    assert chunks == ["partial ", expected_error]
    assert history[-2] == {"role": "user", "content": "hello"}
    assert history[-1] == {"role": "model", "content": f"partial {expected_error}"}


def test_chat_remains_available_when_therapy_is_disabled(monkeypatch):
    manager = ChatManager(gemini_api_key="local-key", enable_therapy=False)

    class StubProvider:
        async def generate_content(self, prompt, system_instruction=None):
            return "chat ok"

    class StubTherapyEngine:
        api_key_valid = False
        _provider = None

    monkeypatch.setattr(manager, "_chat_provider", StubProvider())
    monkeypatch.setattr(manager, "_therapy_engine", StubTherapyEngine())

    out = run_async(manager.run_chat("s-chat", "hello"))

    assert out["response"] == "chat ok"
