from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi.testclient import TestClient

from app import main as app_main
from models.mental_health import mental_llm


class _Settings:
    api_title = "GP Medical Analysis API"
    api_version = "1.0.0"
    cors_origin_list = ["http://127.0.0.1:3000"]

    require_service_api_key = False
    service_api_key = None
    max_upload_bytes = 10 * 1024 * 1024

    use_rag = False
    faiss_index_dir = None
    clinicalbert_model_dir = None
    allow_unsafe_pickle_metadata = False
    llm_provider = "gemini"
    llm_api_key = None
    llm_model_name = None
    openrouter_base_url = "https://openrouter.ai/api/v1"
    openrouter_site_url = None
    openrouter_app_name = "GP Medical Analysis"
    gemini_api_key = ""
    gemini_model_name = "gemini-2.5-flash-lite"
    enable_therapy = False
    rag_top_k = 5
    rag_translate_arabic = True

    use_finetuned_classifier = False
    finetuned_model_dir = None
    classifier_max_length = 256
    classifier_translate_arabic = True

    mental_health_model_dir = "missing"
    mental_health_enabled = True
    mental_health_load_in_4bit = True
    mental_health_max_new_tokens = 400
    mental_health_device = "auto"

    @property
    def resolved_llm_provider(self):
        return self.llm_provider

    @property
    def resolved_llm_api_key(self):
        return None

    @property
    def resolved_llm_model_name(self):
        return "gemini-2.5-flash-lite"


class _StubManager:
    async def run_chat(self, *, session_id: str, message: str):
        return {"session_id": session_id, "message": message, "response": "stub"}

    async def stream_chat(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        yield "stub"


def _client(monkeypatch, settings=None):
    settings = settings or _Settings()
    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    monkeypatch.setattr(mental_llm, "get_settings", lambda: settings)
    monkeypatch.setattr("manager.chat_manager.ChatManager", lambda **kwargs: _StubManager())
    return TestClient(app_main.create_app())


def _write_adapter_fixture(path):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "README.md",
    ):
        (path / name).write_text("fixture", encoding="utf-8")


def test_artifact_detection_identifies_lora_adapter(tmp_path):
    adapter_dir = tmp_path / "mental-health"
    _write_adapter_fixture(adapter_dir)

    report = mental_llm.inspect_mental_model_assets(adapter_dir)

    assert report["status"] == "ok"
    assert report["is_lora_adapter"] is True
    assert report["is_merged_model"] is False
    assert report["base_model_name_or_path"] == "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
    assert report["missing_files"] == []


def test_guardrail_detects_self_harm_without_loading_model(monkeypatch):
    monkeypatch.setitem(mental_llm._STATE, "loaded", False)

    result = mental_llm.generate_mental_support_reply("I want to kill myself", language="en")

    assert result["safety_status"] == "crisis"
    assert result["model_loaded"] is False
    assert "emergency" in result["reply"].lower()


def test_medication_request_refusal():
    result = mental_llm.generate_mental_support_reply("Please prescribe Xanax dosage for panic", language="en")

    assert result["safety_status"] == "medication_refusal"
    assert "can't prescribe" in result["reply"].lower()


def test_diagnosis_request_refusal():
    result = mental_llm.generate_mental_support_reply("Can you diagnose me with depression?", language="en")

    assert result["safety_status"] == "diagnosis_refusal"
    assert "formal psychiatric diagnosis" in result["reply"].lower()


def test_endpoint_returns_disclaimer(monkeypatch):
    monkeypatch.setattr(
        mental_llm,
        "generate_mental_support_reply",
        lambda message, language=None, max_new_tokens=None: {
            "reply": "supportive reply",
            "detected_language": "en",
            "safety_status": "safe",
            "model_loaded": False,
            "latency_ms": 1,
        },
    )

    with _client(monkeypatch) as client:
        response = client.post("/api/v1/mental-health/chat", json={"message": "I feel anxious", "language": "en"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["disclaimer"] == mental_llm.DISCLAIMER
    assert payload["model"] == mental_llm.MODEL_ID


def test_endpoint_handles_disabled_model_gracefully(monkeypatch):
    settings = _Settings()
    settings.mental_health_enabled = False

    with _client(monkeypatch, settings) as client:
        response = client.post("/api/v1/mental-health/chat", json={"message": "I feel anxious", "language": "en"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["safety_status"] == "unavailable"
    assert payload["model_loaded"] is False
    assert payload["disclaimer"] == mental_llm.DISCLAIMER


def test_arabic_language_detection_uses_ar_not_fa():
    assert mental_llm.detect_language("أنا حزين ومتوتر") == "ar"
    assert mental_llm.detect_language("I feel sad") == "en"

