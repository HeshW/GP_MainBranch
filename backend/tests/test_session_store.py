from __future__ import annotations

from manager.session_store import ChatSessionStore


def _clock_at(start: float = 0.0):
    current = [start]

    def now() -> float:
        return current[0]

    return current, now


def test_session_store_evicts_oldest_when_max_sessions_exceeded():
    current, now = _clock_at()
    store = ChatSessionStore(max_sessions=2, session_ttl_seconds=1000, clock=now)

    store.append("s1", "user", "first")
    current[0] += 1
    store.append("s2", "user", "second")
    current[0] += 1
    store.append("s3", "user", "third")

    assert "s1" not in store._sessions
    assert set(store._sessions.keys()) == {"s2", "s3"}


def test_session_store_prunes_expired_sessions_by_ttl():
    current, now = _clock_at()
    store = ChatSessionStore(max_sessions=10, session_ttl_seconds=5, clock=now)

    store.append("expired", "user", "hello")
    current[0] = 6
    store.append("active", "user", "world")

    assert "expired" not in store._sessions
    assert "active" in store._sessions


def test_session_store_touch_on_get_extends_ttl_window():
    current, now = _clock_at()
    store = ChatSessionStore(max_sessions=10, session_ttl_seconds=5, clock=now)

    store.append("session", "user", "hi")
    current[0] = 4
    store.get("session")
    current[0] = 8
    store.append("other", "user", "yo")

    assert "session" in store._sessions


def test_session_store_trims_history_to_keep_recent_messages():
    store = ChatSessionStore(max_messages=3, keep_recent=2)

    store.append("s1", "user", "m1")
    store.append("s1", "model", "m2")
    store.append("s1", "user", "m3")
    history = store.append("s1", "model", "m4")

    assert len(history) == 2
    assert [item["content"] for item in history] == ["m3", "m4"]
