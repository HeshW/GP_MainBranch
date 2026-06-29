from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

from fastapi.testclient import TestClient

from app import main as app_main
from app.database import SessionLocal
from app.db_models import User


class _Settings:
    api_title = "GP Medical Analysis API"
    api_version = "1.0.0"
    cors_origin_list = ["http://127.0.0.1:3000"]

    database_url = "sqlite:///backend/chatbot.db"
    jwt_secret_key = "test-secret"
    jwt_algorithm = "HS256"
    jwt_access_token_expire_minutes = 60

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

    mental_health_enabled = False


class _StubManager:
    async def run_chat(self, *, session_id: str, message: str):
        return {"session_id": session_id, "message": message, "response": "stub response"}

    async def stream_chat(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        yield "stub response"


def _build_client(monkeypatch):
    app_main.get_settings.cache_clear()
    monkeypatch.setattr(app_main, "get_settings", lambda: _Settings())
    monkeypatch.setattr("manager.chat_manager.ChatManager", lambda **kwargs: _StubManager())
    app = app_main.create_app()
    return TestClient(app)


def _register(client: TestClient, name: str) -> tuple[dict, str]:
    email = f"{name}-{uuid.uuid4().hex}@example.test"
    response = client.post(
        "/api/v1/auth/register",
        json={"name": name, "email": email, "password": "password123"},
    )
    assert response.status_code == 201
    payload = response.json()
    return payload, email


def test_auth_and_chat_history_are_user_scoped(monkeypatch):
    with _build_client(monkeypatch) as client:
        first, first_email = _register(client, "First User")
        second, _ = _register(client, "Second User")

        first_headers = {"Authorization": f"Bearer {first['access_token']}"}
        second_headers = {"Authorization": f"Bearer {second['access_token']}"}

        me = client.get("/api/v1/auth/me", headers=first_headers)
        assert me.status_code == 200
        assert me.json()["email"] == first_email.lower()

        login = client.post(
            "/api/v1/auth/login",
            json={"email": first_email, "password": "password123"},
        )
        assert login.status_code == 200
        assert login.json()["access_token"]

        chat = client.post("/api/v1/chats", json={}, headers=first_headers)
        assert chat.status_code == 201
        chat_id = chat.json()["id"]

        user_message = client.post(
            f"/api/v1/chats/{chat_id}/messages",
            json={"role": "user", "content": "Chest pain when exercising"},
            headers=first_headers,
        )
        assistant_message = client.post(
            f"/api/v1/chats/{chat_id}/messages",
            json={"role": "assistant", "content": "Please seek urgent medical care."},
            headers=first_headers,
        )
        assert user_message.status_code == 201
        assert assistant_message.status_code == 201

        chats = client.get("/api/v1/chats", headers=first_headers)
        assert chats.status_code == 200
        assert chats.json()[0]["title"] == "Chest pain when exercising"

        messages = client.get(f"/api/v1/chats/{chat_id}/messages", headers=first_headers)
        assert messages.status_code == 200
        assert [item["role"] for item in messages.json()] == ["user", "assistant"]

        blocked_read = client.get(f"/api/v1/chats/{chat_id}/messages", headers=second_headers)
        blocked_delete = client.delete(f"/api/v1/chats/{chat_id}", headers=second_headers)
        assert blocked_read.status_code == 404
        assert blocked_delete.status_code == 404

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == first_email.lower()).one()
            assert user.password_hash != "password123"
        finally:
            db.close()
