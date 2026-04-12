from __future__ import annotations

from manager.runtime import run_async
from models.common import ai_provider


class _Chunk:
    def __init__(self, text: str | None) -> None:
        self.text = text


async def _collect_stream(stream):
    return [item async for item in stream]


def test_generate_stream_supports_coroutine_returning_stream(monkeypatch):
    async def stream_iter():
        yield _Chunk("hello ")
        yield _Chunk(None)
        yield _Chunk("world")

    class StubModels:
        async def generate_content_stream(self, **kwargs):
            return stream_iter()

    class StubAio:
        models = StubModels()

    class StubClient:
        def __init__(self, api_key):
            self.aio = StubAio()

    monkeypatch.setattr(ai_provider.genai, "Client", StubClient)

    provider = ai_provider.GeminiProvider(api_key="local-key")
    chunks = run_async(_collect_stream(provider.generate_stream("prompt")))

    assert chunks == ["hello ", "world"]


def test_generate_stream_supports_async_iterator_returning_stream(monkeypatch):
    async def stream_iter():
        yield _Chunk("chunk-1")
        yield _Chunk("")
        yield _Chunk("chunk-2")

    class StubModels:
        def generate_content_stream(self, **kwargs):
            return stream_iter()

    class StubAio:
        models = StubModels()

    class StubClient:
        def __init__(self, api_key):
            self.aio = StubAio()

    monkeypatch.setattr(ai_provider.genai, "Client", StubClient)

    provider = ai_provider.GeminiProvider(api_key="local-key")
    chunks = run_async(_collect_stream(provider.generate_stream("prompt")))

    assert chunks == ["chunk-1", "chunk-2"]
