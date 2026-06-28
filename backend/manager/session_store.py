from __future__ import annotations

import time
from typing import Callable, Dict, List

ChatMessage = Dict[str, str]


class ChatSessionStore:
    """Keep lightweight in-memory histories for chat sessions."""

    def __init__(
        self,
        *,
        max_messages: int = 12,
        keep_recent: int = 10,
        max_sessions: int = 500,
        session_ttl_seconds: int = 60 * 60,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._sessions: Dict[str, List[ChatMessage]] = {}
        self._last_seen: Dict[str, float] = {}
        self._max_messages = max_messages
        self._keep_recent = keep_recent
        self._max_sessions = max(1, max_sessions)
        self._session_ttl_seconds = max(1, session_ttl_seconds)
        self._clock = clock or time.monotonic

    def get(self, session_id: str) -> List[ChatMessage]:
        self._prune_expired()
        history = self._sessions.setdefault(session_id, [])
        self._touch(session_id)
        self._evict_if_needed()
        return history

    def append(self, session_id: str, role: str, content: str) -> List[ChatMessage]:
        history = self.get(session_id)
        history.append({"role": role, "content": content})
        self._trim(session_id)
        self._touch(session_id)
        return self._sessions[session_id]

    def _touch(self, session_id: str) -> None:
        self._last_seen[session_id] = self._clock()

    def _delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._last_seen.pop(session_id, None)

    def _prune_expired(self) -> None:
        now = self._clock()
        cutoff = now - self._session_ttl_seconds
        expired = [session_id for session_id, seen_at in self._last_seen.items() if seen_at < cutoff]
        for session_id in expired:
            self._delete(session_id)

    def _evict_if_needed(self) -> None:
        while len(self._sessions) > self._max_sessions:
            oldest_session_id = min(self._last_seen, key=self._last_seen.get)
            self._delete(oldest_session_id)

    def _trim(self, session_id: str) -> None:
        history = self._sessions[session_id]
        if len(history) > self._max_messages:
            self._sessions[session_id] = history[-self._keep_recent :]
