from __future__ import annotations

from fastapi.testclient import TestClient

from app import main as app_main


class _Settings:
    api_title = "GP Medical Analysis API"
    api_version = "1.0.0"
    cors_origin_list = ["http://127.0.0.1:5173"]

    use_rag = True
    faiss_index_dir = "missing/faiss"
    clinicalbert_model_dir = "missing/clinicalbert"
    allow_unsafe_pickle_metadata = False
    gemini_api_key = ""
    enable_therapy = False
    rag_top_k = 5
    rag_translate_arabic = True

    use_finetuned_classifier = True
    finetuned_model_dir = "missing/classifier"
    classifier_max_length = 256
    classifier_translate_arabic = True


def test_startup_disables_optional_ai_modules_when_assets_missing(monkeypatch):
    calls = []

    class StubChatManager:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(app_main, "get_settings", lambda: _Settings())
    monkeypatch.setattr("manager.chat_manager.ChatManager", StubChatManager)

    app = app_main.create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/meta")

    assert response.status_code == 200
    payload = response.json()
    assert payload["rag_enabled"] is False
    assert payload["finetuned_classifier_enabled"] is False
    assert payload["therapy_enabled"] is False
    assert len(calls) == 1
    assert calls[0]["use_rag"] is False
    assert calls[0]["use_finetuned_classifier"] is False


def test_startup_falls_back_to_degraded_mode_if_advanced_init_fails(monkeypatch):
    calls = []

    class FlakyChatManager:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("simulated init failure")

    settings = _Settings()
    settings.use_rag = False
    settings.use_finetuned_classifier = False

    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    monkeypatch.setattr("manager.chat_manager.ChatManager", FlakyChatManager)

    app = app_main.create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert len(calls) == 2
    assert app.state.runtime_features["rag_enabled"] is False
    assert app.state.runtime_features["finetuned_classifier_enabled"] is False
    assert app.state.runtime_features["therapy_enabled"] is False
