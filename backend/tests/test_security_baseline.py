from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import deps as app_deps
from app.routers import chat as chat_router
from app.routers import health as health_router
from app.routers import pipeline as pipeline_router


class _DummyManager:
    async def run_chat(self, *, session_id: str, message: str):
        return {
            "session_id": session_id,
            "message": message,
            "response": "ok",
        }

    async def stream_chat(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        yield f"{session_id}:{message}"

    async def run_pipeline(self, **kwargs):
        return {"status": "ok", "payload": kwargs}


class _Settings:
    def __init__(self, *, require_service_api_key: bool, service_api_key: str | None) -> None:
        self.require_service_api_key = require_service_api_key
        self.service_api_key = service_api_key


def _build_client(monkeypatch, *, require_service_api_key: bool, service_api_key: str | None):
    app = FastAPI()
    app.include_router(health_router.router, prefix="/api/v1")
    app.include_router(pipeline_router.router, prefix="/api/v1")
    app.include_router(chat_router.router, prefix="/api/v1")
    app.state.chat_manager = _DummyManager()

    monkeypatch.setattr(
        app_deps,
        "get_settings",
        lambda: _Settings(
            require_service_api_key=require_service_api_key,
            service_api_key=service_api_key,
        ),
    )
    return TestClient(app)


def test_protected_routes_allow_requests_when_key_not_required(monkeypatch):
    client = _build_client(
        monkeypatch,
        require_service_api_key=False,
        service_api_key=None,
    )

    response = client.post("/api/v1/pipeline/labs", json={"labs": {"glucose": 101}})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_protected_routes_require_api_key_when_enabled(monkeypatch):
    client = _build_client(
        monkeypatch,
        require_service_api_key=True,
        service_api_key="local-secret",
    )

    unauthorized = client.post("/api/v1/pipeline/labs", json={"labs": {"glucose": 101}})
    authorized = client.post(
        "/api/v1/pipeline/labs",
        json={"labs": {"glucose": 101}},
        headers={"X-API-Key": "local-secret"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_protected_routes_accept_bearer_token_when_enabled(monkeypatch):
    client = _build_client(
        monkeypatch,
        require_service_api_key=True,
        service_api_key="local-secret",
    )

    response = client.post(
        "/api/v1/pipeline/labs",
        json={"labs": {"glucose": 101}},
        headers={"Authorization": "Bearer local-secret"},
    )

    assert response.status_code == 200


def test_auth_enabled_without_key_returns_service_unavailable(monkeypatch):
    client = _build_client(
        monkeypatch,
        require_service_api_key=True,
        service_api_key="",
    )

    response = client.post("/api/v1/pipeline/labs", json={"labs": {"glucose": 101}})

    assert response.status_code == 503
    assert "SERVICE_API_KEY" in response.json()["detail"]


def test_health_route_stays_open(monkeypatch):
    client = _build_client(
        monkeypatch,
        require_service_api_key=True,
        service_api_key="local-secret",
    )

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_stream_uses_post_body_not_query_params(monkeypatch):
    client = _build_client(
        monkeypatch,
        require_service_api_key=True,
        service_api_key="local-secret",
    )

    post_response = client.post(
        "/api/v1/chat/stream",
        json={"session_id": "s1", "message": "hello"},
        headers={"X-API-Key": "local-secret"},
    )
    get_response = client.get(
        "/api/v1/chat/stream",
        params={"session_id": "s1", "message": "hello"},
        headers={"X-API-Key": "local-secret"},
    )

    assert post_response.status_code == 200
    assert post_response.headers["content-type"].startswith("text/event-stream")
    assert "data: s1:hello" in post_response.text
    assert get_response.status_code == 405
