from __future__ import annotations

from typing import Dict, List

ChatMessage = Dict[str, str]


class ChatSessionStore:
    """Keep lightweight in-memory histories for chat sessions."""

    def __init__(self, *, max_messages: int = 12, keep_recent: int = 10) -> None:
        self._sessions: Dict[str, List[ChatMessage]] = {}
        self._max_messages = max_messages
        self._keep_recent = keep_recent

    def get(self, session_id: str) -> List[ChatMessage]:
        return self._sessions.setdefault(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> List[ChatMessage]:
        history = self.get(session_id)
        history.append({"role": role, "content": content})
        self._trim(session_id)
        return self._sessions[session_id]

    def _trim(self, session_id: str) -> None:
        history = self._sessions[session_id]
        if len(history) > self._max_messages:
            self._sessions[session_id] = history[-self._keep_recent :]
