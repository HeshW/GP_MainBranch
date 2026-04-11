"""Shared FastAPI dependencies."""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from fastapi import Header, HTTPException, Request, status

from app.config import get_settings

if TYPE_CHECKING:
    from manager.chat_manager import ChatManager


def get_chat_manager(request: Request) -> "ChatManager":
    """Return the process-wide :class:`~manager.chat_manager.ChatManager` instance."""
    manager = getattr(request.app.state, "chat_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat manager is not initialized.",
        )
    return manager


def require_service_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """Enforce optional service-level API key checks without user accounts."""
    settings = get_settings()
    if not settings.require_service_api_key:
        return

    expected = str(settings.service_api_key or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service API key auth is enabled but SERVICE_API_KEY is missing.",
        )

    candidate = ""
    if x_api_key:
        candidate = x_api_key.strip()
    elif authorization:
        value = authorization.strip()
        if value.lower().startswith("bearer "):
            candidate = value[7:].strip()

    if not candidate or not hmac.compare_digest(candidate, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: valid service API key is required.",
        )
