from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi.testclient import TestClient

from app import main as app_main


class _Settings:
    api_title = "GP Medical Analysis API"
    api_version = "1.0.0"
    cors_origin_list = ["http://127.0.0.1:5173"]

    require_service_api_key = False
    service_api_key = None
    max_upload_bytes = 10 * 1024 * 1024

    use_rag = False
    faiss_index_dir = None
    clinicalbert_model_dir = None
    allow_unsafe_pickle_metadata = False
    gemini_api_key = ""
    enable_therapy = False
    rag_top_k = 5
    rag_translate_arabic = True

    use_finetuned_classifier = False
    finetuned_model_dir = None
    classifier_max_length = 256
    classifier_translate_arabic = True


class _StubManager:
    async def run_pipeline(self, *, image=None, labs=None, manual_input=None):
        return {
            "status": "ok",
            "report": {
                "raw_text": "stub report",
                "labs": labs or {},
                "symptoms": [manual_input.get("symptoms")] if isinstance(manual_input, dict) and manual_input.get("symptoms") else [],
            },
            "diagnosis": {"summary": "stub diagnosis"},
            "therapy": {"therapy_plan": "stub therapy"},
            "warnings": [],
            "elapsed_ms": 1.0,
        }

    async def run_ocr_only(self, image):
        return {"status": "ok", "ocr": {"raw_text": "stub", "image": str(image)}}

    async def run_clarification(self, report, answers, *, prior_diagnosis=None, low_confidence_threshold=0.7):
        return {
            "status": "ok",
            "report": report,
            "diagnosis": {
                "summary": "clarified",
                "clarification": {"applied": True, "answers_used": answers},
            },
            "therapy": {"therapy_plan": "stub therapy"},
        }

    async def run_chat(self, *, session_id: str, message: str):
        return {
            "session_id": session_id,
            "message": message,
            "response": "stub response",
        }

    async def stream_chat(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        yield f"{session_id}:{message}"


def _build_client(monkeypatch):
    monkeypatch.setattr(app_main, "get_settings", lambda: _Settings())
    monkeypatch.setattr("manager.chat_manager.ChatManager", lambda **kwargs: _StubManager())
    app = app_main.create_app()
    return TestClient(app)


def test_health_and_meta_contract(monkeypatch):
    with _build_client(monkeypatch) as client:
        health = client.get("/api/v1/health")
        meta = client.get("/api/v1/meta")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    assert meta.status_code == 200
    payload = meta.json()
    assert "api_version" in payload
    assert "rag_enabled" in payload
    assert "finetuned_classifier_enabled" in payload
    assert "therapy_enabled" in payload


def test_pipeline_labs_contract(monkeypatch):
    with _build_client(monkeypatch) as client:
        response = client.post(
            "/api/v1/pipeline/labs",
            json={"labs": {"glucose": 135}, "symptoms": "fatigue"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "diagnosis" in payload
    assert "therapy" in payload


def test_chat_and_stream_contract(monkeypatch):
    with _build_client(monkeypatch) as client:
        chat = client.post(
            "/api/v1/chat",
            json={"session_id": "s1", "message": "hello"},
        )
        stream = client.post(
            "/api/v1/chat/stream",
            json={"session_id": "s1", "message": "hello"},
        )

    assert chat.status_code == 200
    assert chat.json()["response"] == "stub response"

    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert "data: s1:hello" in stream.text


def test_clarification_contract(monkeypatch):
    with _build_client(monkeypatch) as client:
        response = client.post(
            "/api/v1/pipeline/diagnosis/clarify",
            json={
                "report": {"raw_text": "fatigue", "labs": {}, "symptoms": ["fatigue"]},
                "answers": ["symptoms worse at night"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["diagnosis"]["clarification"]["applied"] is True
