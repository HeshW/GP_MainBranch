from manager.runtime import run_async
from models.therapy.engine import TherapyEngine


def test_therapy_engine_fallback_initializes_without_gemini_key():
    engine = TherapyEngine("")
    result = run_async(
        engine.generate_therapy(
            {
                "findings": [
                    {
                        "condition": "Diabetes Mellitus (suspected)",
                        "severity": "high",
                        "confidence": "high",
                        "evidence": "glucose=130 mg/dL",
                    }
                ],
                "safety": {"clinician_review_required": True},
            },
            "Age: Unknown, Sex: Unknown",
        )
    )

    assert engine.api_key_valid is False
    assert result["metadata"]["mode"] == "fallback"
    assert result["metadata"]["provider_status"] == "missing_api_key"
    assert "clinician" in result["therapy_plan"].lower()


def test_therapy_engine_returns_no_findings_payload():
    engine = TherapyEngine("")
    result = run_async(engine.generate_therapy({"findings": []}))

    assert result["metadata"]["mode"] == "no_findings"
    assert result["structured_therapy"] is None


def test_therapy_engine_accepts_non_ai_prefix_key(monkeypatch):
    class StubProvider:
        def __init__(self, api_key, model_name):
            self.api_key = api_key

        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            return (
                '{"clinical_analysis":"stable","recommendations":[{"category":"Lifestyle",'
                '"description":"Hydration","urgency":"Routine"}],"lifestyle_advice":"Sleep well",'
                '"emergency_signs":["Chest pain"],"disclaimer":"Consult clinician."}'
            )

    monkeypatch.setattr("models.therapy.engine.GeminiProvider", StubProvider)

    engine = TherapyEngine("seminar-local-key")
    result = run_async(
        engine.generate_therapy(
            {
                "findings": [
                    {
                        "condition": "Diabetes Mellitus (suspected)",
                        "severity": "high",
                        "confidence": "high",
                        "evidence": "glucose=130 mg/dL",
                    }
                ],
                "safety": {"clinician_review_required": True},
            },
            "Age: Unknown, Sex: Unknown",
        )
    )

    assert engine.api_key_valid is True
    assert result["metadata"]["mode"] == "llm"
    assert result["structured_therapy"]["clinical_analysis"] == "stable"


def test_therapy_engine_exposes_provider_error_status(monkeypatch):
    class StubProvider:
        def __init__(self, api_key, model_name):
            pass

        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            raise RuntimeError("401 unauthorized")

    monkeypatch.setattr("models.therapy.engine.GeminiProvider", StubProvider)

    engine = TherapyEngine("seminar-local-key")
    result = run_async(
        engine.generate_therapy(
            {
                "findings": [
                    {
                        "condition": "Hypertension",
                        "severity": "moderate",
                        "confidence": "moderate",
                        "evidence": "bp=160/95",
                    }
                ]
            }
        )
    )

    assert result["metadata"]["mode"] == "fallback"
    assert result["metadata"]["provider_status"] == "provider_unauthorized"
