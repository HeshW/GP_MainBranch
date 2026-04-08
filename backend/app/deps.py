"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from manager.chat_manager import ChatManager


def get_chat_manager(request: Request) -> "ChatManager":
    """Return the process-wide :class:`~manager.chat_manager.ChatManager` instance."""
    return request.app.state.chat_manager
