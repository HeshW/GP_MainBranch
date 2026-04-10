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
    assert "clinician" in result["therapy_plan"].lower()


def test_therapy_engine_returns_no_findings_payload():
    engine = TherapyEngine("")
    result = run_async(engine.generate_therapy({"findings": []}))

    assert result["metadata"]["mode"] == "no_findings"
    assert result["structured_therapy"] is None
