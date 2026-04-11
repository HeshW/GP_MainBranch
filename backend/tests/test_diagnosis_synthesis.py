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


def test_synthesizer_accepts_non_ai_prefix_key(monkeypatch):
    class StubProvider:
        def __init__(self, api_key, model_name):
            self.api_key = api_key

        async def generate_content(self, prompt, system_instruction=None, response_model=None):
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
    assert "Likely influenza" in result["response_text"]


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
