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
    assert result["metadata"]["provider_name"] == "gemini"


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
        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            captured["prompt"] = prompt
            captured["system_instruction"] = system_instruction
            return (
                '{"diagnosis_summary":"Likely influenza","patient_friendly_explanation":"Viral syndrome",'
                '"recommended_next_steps":["Hydration"],"red_flags":["Breathing difficulty"],'
                '"disclaimer":"Consult clinician."}'
            )

    monkeypatch.setattr(
        "models.diagnosis.synthesis.create_model_provider",
        lambda **kwargs: ("gemini", StubProvider(), "gemini-2.5-flash-lite"),
    )

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
        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            captured["prompt"] = prompt
            captured["system_instruction"] = system_instruction
            return (
                '{"diagnosis_summary":"ملخص تشخيصي","patient_friendly_explanation":"شرح مبسط",'
                '"recommended_next_steps":["المتابعة مع الطبيب"],"red_flags":["ضيق تنفس شديد"],'
                '"disclaimer":"هذه المعلومات تعليمية."}'
            )

    monkeypatch.setattr(
        "models.diagnosis.synthesis.create_model_provider",
        lambda **kwargs: ("gemini", StubProvider(), "gemini-2.5-flash-lite"),
    )

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
        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            raise RuntimeError("429 rate limit")

    monkeypatch.setattr(
        "models.diagnosis.synthesis.create_model_provider",
        lambda **kwargs: ("gemini", StubProvider(), "gemini-2.5-flash-lite"),
    )

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


def test_synthesizer_normalizes_openrouter_style_payload(monkeypatch):
    class StubProvider:
        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            return (
                '{"patient_explanation": {'
                '"summary":"Possible NSTEMI based on current symptoms.",'
                '"next_steps":"Seek urgent clinician review and repeat troponin.",'
                '"uncertainty":"This remains preliminary and needs clinical confirmation."'
                '}, "clinician_review_status": {'
                '"review_required": true,'
                '"emergency_recommendation":"Worsening chest pain requires immediate emergency care."'
                '}}'
            )

    monkeypatch.setattr(
        "models.diagnosis.synthesis.create_model_provider",
        lambda **kwargs: ("openrouter", StubProvider(), "openrouter/auto"),
    )

    synthesizer = DiagnosisResponseSynthesizer(
        "unused",
        llm_provider="openrouter",
        llm_api_key="openrouter-key",
        llm_model_name="openrouter/auto",
    )
    result = run_async(
        synthesizer.synthesize(
            {"raw_text": "chest pain and shortness of breath"},
            {
                "summary": "Preliminary assessment.",
                "final_diagnosis": {"diagnosis": "Possible NSTEMI", "confidence": 0.71, "source": "fusion"},
            },
        )
    )

    assert result["metadata"]["mode"] == "llm"
    assert result["metadata"]["provider_name"] == "openrouter"
    assert result["metadata"]["model_name"] == "openrouter/auto"
    assert "Possible NSTEMI" in result["response_text"]
    assert "Recommended next steps" in result["response_text"]


def test_synthesizer_avoids_duplicate_summary_and_enriches_sparse_defaults(monkeypatch):
    duplicate_line = "AI-assisted assessment suggests URTI (confidence 0.79, source: classifier)."

    class StubProvider:
        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            return f'{{"summary":"{duplicate_line}"}}'

    monkeypatch.setattr(
        "models.diagnosis.synthesis.create_model_provider",
        lambda **kwargs: ("openrouter", StubProvider(), "openrouter/auto"),
    )

    synthesizer = DiagnosisResponseSynthesizer(
        "unused",
        llm_provider="openrouter",
        llm_api_key="openrouter-key",
        llm_model_name="openrouter/auto",
    )
    result = run_async(
        synthesizer.synthesize(
            {"raw_text": "cough and sore throat"},
            {
                "summary": duplicate_line,
                "final_diagnosis": {"diagnosis": "URTI", "confidence": 0.79, "source": "classifier"},
                "safety": {"reasons": ["Clinical review is recommended for any abnormal finding."]},
            },
        )
    )

    assert result["metadata"]["mode"] == "llm"
    assert result["response_text"].count(duplicate_line) == 1
    structured = result["structured_response"]
    assert structured is not None
    assert len(structured["recommended_next_steps"]) >= 2
    assert len(structured["red_flags"]) >= 2


def test_synthesizer_collapses_duplicate_paragraphs_inside_provider_summary(monkeypatch):
    duplicate_line = "AI-assisted assessment suggests URTI (confidence 0.79, source: classifier) with rule-based safety checks attached."

    class StubProvider:
        async def generate_content(self, prompt, system_instruction=None, response_model=None):
            return (
                '{"diagnosis_summary":"'
                + duplicate_line
                + "\\n\\n"
                + duplicate_line
                + '","patient_friendly_explanation":"Please continue with clinician follow-up.",'
                '"recommended_next_steps":["Hydration"],"red_flags":["Worsening breathing difficulty"],'
                '"disclaimer":"Consult clinician."}'
            )

    monkeypatch.setattr(
        "models.diagnosis.synthesis.create_model_provider",
        lambda **kwargs: ("openrouter", StubProvider(), "openrouter/auto"),
    )

    synthesizer = DiagnosisResponseSynthesizer(
        "unused",
        llm_provider="openrouter",
        llm_api_key="openrouter-key",
        llm_model_name="openrouter/auto",
    )
    result = run_async(
        synthesizer.synthesize(
            {"raw_text": "cough and sore throat"},
            {
                "summary": duplicate_line,
                "final_diagnosis": {"diagnosis": "URTI", "confidence": 0.79, "source": "classifier"},
            },
        )
    )

    assert result["metadata"]["mode"] == "llm"
    assert result["response_text"].count(duplicate_line) == 1
