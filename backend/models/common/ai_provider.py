"""Async abstraction for LLM providers used by the project."""

from __future__ import annotations

import abc
import logging
from typing import Any, AsyncGenerator, Optional, Type

from google import genai
from google.genai import types
from pydantic import BaseModel

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

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _build_config(
        *,
        system_instruction: Optional[str],
        json_mode: bool,
        response_model: Optional[Type[BaseModel]],
        kwargs: dict[str, Any],
    ) -> types.GenerateContentConfig:
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
            async for chunk in self._client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config,
            ):
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as exc:
            logger.error("Gemini streaming failed: %s", exc)
            raise
