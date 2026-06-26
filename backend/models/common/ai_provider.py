"""Async abstraction for LLM providers used by the project."""

from __future__ import annotations

import abc
import asyncio
import inspect
import json
import logging
import threading
from typing import Any, AsyncGenerator, Optional, Type

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from pydantic import BaseModel
import requests

logger = logging.getLogger(__name__)


class BaseModelProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    async def generate_content(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text response asynchronously."""

    @abc.abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream a text response asynchronously."""
        yield ""


class GeminiProvider(BaseModelProvider):
    """Google GenAI implementation backed by ``google.genai``."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite") -> None:
        if genai is None:
            raise ImportError(
                "The 'google-genai' package is required for GeminiProvider. "
                "Install it with: pip install google-genai"
            )
        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _build_config(
        *,
        system_instruction: Optional[str],
        json_mode: bool,
        response_model: Optional[Type[BaseModel]],
        kwargs: dict[str, Any],
    ):
        if types is None:
            raise ImportError(
                "The 'google-genai' package is required for GeminiProvider. "
                "Install it with: pip install google-genai"
            )
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=kwargs.get("temperature", 0.3),
            top_p=kwargs.get("top_p", 0.8),
            top_k=kwargs.get("top_k", 40),
            max_output_tokens=kwargs.get("max_output_tokens", 4096),
        )

        if response_model:
            config.response_mime_type = "application/json"
            config.response_schema = response_model
        elif json_mode:
            config.response_mime_type = "application/json"

        return config

    async def generate_content(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        config = self._build_config(
            system_instruction=system_instruction,
            json_mode=json_mode,
            response_model=response_model,
            kwargs=kwargs,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc)
            raise

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        config = self._build_config(
            system_instruction=system_instruction,
            json_mode=False,
            response_model=None,
            kwargs={
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.8),
                "top_k": kwargs.get("top_k", 40),
                "max_output_tokens": kwargs.get("max_output_tokens", 4096),
            },
        )

        try:
            stream_handle = self._client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            # google.genai may return either an async iterator directly or
            # a coroutine that resolves to one, depending on SDK version.
            if inspect.isawaitable(stream_handle):
                stream_handle = await stream_handle

            async for chunk in stream_handle:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as exc:
            logger.error("Gemini streaming failed: %s", exc)
            raise


class OpenAICompatibleProvider(BaseModelProvider):
    """OpenAI-compatible HTTP provider for hosted APIs (e.g., OpenRouter)."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        *,
        base_url: str,
        default_headers: Optional[dict[str, str]] = None,
        provider_name: str = "OpenAI-compatible provider",
        timeout_seconds: float = 90.0,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._chat_completions_url = f"{self.base_url}/chat/completions"
        self._default_headers = default_headers or {}
        self._provider_name = provider_name
        self._timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._default_headers)
        return headers

    @staticmethod
    def _extract_content_text(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            parts: list[str] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            return "".join(parts)
        return ""

    def _build_payload(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str],
        json_mode: bool,
        response_model: Optional[Type[BaseModel]],
        kwargs: dict[str, Any],
        stream: bool,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.8),
            "max_tokens": kwargs.get("max_output_tokens", 4096),
        }

        if json_mode or response_model:
            payload["response_format"] = {"type": "json_object"}

        return payload

    def _generate_content_sync(self, payload: dict[str, Any]) -> str:
        response = requests.post(
            self._chat_completions_url,
            headers=self._headers(),
            json=payload,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return self._extract_content_text(content)

    def _iter_stream_sync(self, payload: dict[str, Any]):
        with requests.post(
            self._chat_completions_url,
            headers=self._headers(),
            json=payload,
            timeout=self._timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                line = (raw_line or "").strip()
                if not line or not line.startswith("data:"):
                    continue

                data_part = line[len("data:") :].strip()
                if not data_part:
                    continue
                if data_part == "[DONE]":
                    break

                try:
                    chunk_payload = json.loads(data_part)
                except json.JSONDecodeError:
                    continue

                choices = chunk_payload.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                chunk_text = self._extract_content_text(delta.get("content"))
                if chunk_text:
                    yield chunk_text

    async def generate_content(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        payload = self._build_payload(
            prompt,
            system_instruction=system_instruction,
            json_mode=json_mode,
            response_model=response_model,
            kwargs=kwargs,
            stream=False,
        )
        try:
            return await asyncio.to_thread(self._generate_content_sync, payload)
        except Exception as exc:
            logger.error("%s generation failed: %s", self._provider_name, exc)
            raise

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        payload = self._build_payload(
            prompt,
            system_instruction=system_instruction,
            json_mode=False,
            response_model=None,
            kwargs={
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.8),
                "max_output_tokens": kwargs.get("max_output_tokens", 4096),
            },
            stream=True,
        )

        queue: asyncio.Queue[Any] = asyncio.Queue()
        done_token = object()
        loop = asyncio.get_running_loop()

        def _reader() -> None:
            try:
                for chunk in self._iter_stream_sync(payload):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done_token)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is done_token:
                break
            if isinstance(item, Exception):
                logger.error("%s streaming failed: %s", self._provider_name, item)
                raise item
            if item:
                yield str(item)


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter implementation using OpenAI-compatible chat completions."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "openrouter/auto",
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        app_name: str = "GP Medical Analysis",
    ) -> None:
        headers: dict[str, str] = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            default_headers=headers,
            provider_name="OpenRouter",
        )
