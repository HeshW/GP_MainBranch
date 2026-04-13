from __future__ import annotations

from manager.runtime import run_async
from models.diagnosis.synthesis import DiagnosisResponseSynthesizer


def test_synthesizer_fallback_marks_missing_api_key():
    synthesizer = DiagnosisResponseSynthesizer("")
    result = run_async(
        synthesizer.synthesize(
            {"raw_text": "fatigue"},
            {
                "summary": "Preliminary assessment suggests influenza.",
                "final_diagnosis": {
                    "diagnosis": "Influenza",
                    "confidence": 0.77,
                    "source": "classifier",
                },
            },
        )
    )

    assert result["metadata"]["mode"] == "fallback"
    assert result["metadata"]["provider_status"] == "missing_api_key"
    assert result["metadata"]["response_language"] == "en"


def test_synthesizer_fallback_localizes_arabic_response():
    synthesizer = DiagnosisResponseSynthesizer("")
    result = run_async(
        synthesizer.synthesize(
            {"raw_text": "عندي دوخة وتعب"},
            {
                "final_diagnosis": {
                    "diagnosis": "Anemia",
                    "confidence": 0.58,
                    "source": "rules_fallback",
                },
            },
        )
    )

    assert result["metadata"]["mode"] == "fallback"
    assert result["metadata"]["response_language"] == "ar"
    assert "يشير التقييم الأولي" in result["response_text"]
    assert "التشخيص الأرجح" in result["response_text"]


def test_synthesizer_accepts_non_ai_prefix_key(monkeypatch):
    captured = {}

    class StubProvider:
        def __init__(self, api_key, model_name):
            self.api_key = api_key

        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            captured["prompt"] = prompt
            captured["system_instruction"] = system_instruction
            return (
                '{"diagnosis_summary":"Likely influenza","patient_friendly_explanation":"Viral syndrome",'
                '"recommended_next_steps":["Hydration"],"red_flags":["Breathing difficulty"],'
                '"disclaimer":"Consult clinician."}'
            )

    monkeypatch.setattr("models.diagnosis.synthesis.GeminiProvider", StubProvider)

    synthesizer = DiagnosisResponseSynthesizer("seminar-local-key")
    result = run_async(
        synthesizer.synthesize(
            {"raw_text": "fever and cough", "symptoms": ["fever", "cough"], "labs": {}},
            {
                "summary": "Likely influenza.",
                "final_diagnosis": {"diagnosis": "Influenza", "confidence": 0.86, "source": "classifier"},
            },
        )
    )

    assert synthesizer.api_key_valid is True
    assert result["metadata"]["mode"] == "llm"
    assert result["metadata"]["response_language"] == "en"
    assert "Likely influenza" in result["response_text"]
    assert "Target response language: English" in captured["prompt"]
    assert "Respond strictly in English." in captured["system_instruction"]


def test_synthesizer_requests_arabic_generation_for_arabic_input(monkeypatch):
    captured = {}

    class StubProvider:
        def __init__(self, api_key, model_name):
            self.api_key = api_key

        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            captured["prompt"] = prompt
            captured["system_instruction"] = system_instruction
            return (
                '{"diagnosis_summary":"ملخص تشخيصي","patient_friendly_explanation":"شرح مبسط",'
                '"recommended_next_steps":["المتابعة مع الطبيب"],"red_flags":["ضيق تنفس شديد"],'
                '"disclaimer":"هذه المعلومات تعليمية."}'
            )

    monkeypatch.setattr("models.diagnosis.synthesis.GeminiProvider", StubProvider)

    synthesizer = DiagnosisResponseSynthesizer("seminar-local-key")
    result = run_async(
        synthesizer.synthesize(
            {"raw_text": "عندي ألم صدر"},
            {
                "summary": "تقييم أولي.",
                "final_diagnosis": {"diagnosis": "Pericarditis", "confidence": 0.72, "source": "classifier"},
            },
        )
    )

    assert result["metadata"]["mode"] == "llm"
    assert result["metadata"]["response_language"] == "ar"
    assert "Target response language: Arabic" in captured["prompt"]
    assert "Respond strictly in Arabic." in captured["system_instruction"]
    assert "الخطوات التالية الموصى بها" in result["response_text"]


def test_synthesizer_exposes_provider_error_status(monkeypatch):
    class StubProvider:
        def __init__(self, api_key, model_name):
            pass

        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            raise RuntimeError("429 rate limit")

    monkeypatch.setattr("models.diagnosis.synthesis.GeminiProvider", StubProvider)

    synthesizer = DiagnosisResponseSynthesizer("seminar-local-key")
    result = run_async(
        synthesizer.synthesize(
            {"raw_text": "fever"},
            {
                "summary": "Preliminary assessment.",
                "final_diagnosis": {"diagnosis": "Influenza", "confidence": 0.66, "source": "rag"},
            },
        )
    )

    assert result["metadata"]["mode"] == "fallback"
    assert result["metadata"]["provider_status"] == "provider_rate_limited"
    assert result["metadata"]["response_language"] == "en"
