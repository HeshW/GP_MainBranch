from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import pipeline as pipeline_router


class _DummyManager:
    def __init__(self) -> None:
        self.pipeline_calls = 0
        self.ocr_calls = 0

    async def run_pipeline(self, *, image=None, **kwargs):
        self.pipeline_calls += 1
        return {"status": "ok", "image": image}

    async def run_ocr_only(self, image):
        self.ocr_calls += 1
        return {"status": "ok", "ocr": {"image": image}}


class _Settings:
    def __init__(self, max_upload_bytes: int) -> None:
        self.max_upload_bytes = max_upload_bytes


def _build_client(monkeypatch, max_upload_bytes: int):
    app = FastAPI()
    app.include_router(pipeline_router.router, prefix="/api/v1")
    app.state.chat_manager = _DummyManager()
    monkeypatch.setattr(pipeline_router, "get_settings", lambda: _Settings(max_upload_bytes))
    return TestClient(app), app.state.chat_manager


def test_pipeline_image_rejects_oversized_upload(monkeypatch):
    client, manager = _build_client(monkeypatch, max_upload_bytes=8)

    response = client.post(
        "/api/v1/pipeline/image",
        files={"file": ("report.png", b"123456789", "image/png")},
    )

    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]
    assert manager.pipeline_calls == 0


def test_pipeline_ocr_rejects_oversized_upload(monkeypatch):
    client, manager = _build_client(monkeypatch, max_upload_bytes=8)

    response = client.post(
        "/api/v1/pipeline/ocr",
        files={"file": ("report.png", b"123456789", "image/png")},
    )

    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]
    assert manager.ocr_calls == 0


def test_pipeline_image_allows_small_upload(monkeypatch):
    client, manager = _build_client(monkeypatch, max_upload_bytes=8)

    response = client.post(
        "/api/v1/pipeline/image",
        files={"file": ("report.png", b"1234", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert manager.pipeline_calls == 1
