"""backend/models/common/ai_provider.py

Asynchronous abstraction for LLM providers (Gemini, etc.) to ensure
professional-grade AI orchestration, structured outputs, and streaming support.
"""

from __future__ import annotations

import json
import abc
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union

import google.generativeai as genai
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BaseModelProvider(abc.ABC):
    """Abstract base class for all LLM providers."""

    @abc.abstractmethod
    async def generate_content(
        self, 
        prompt: str, 
        *, 
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> str:
        """Asynchronously generate a text response."""
        pass

    @abc.abstractmethod
    async def generate_stream(
        self, 
        prompt: str, 
        *, 
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Asynchronously stream a text response."""
        yield ""


class GeminiProvider(BaseModelProvider):
    """Google Gemini specific implementation."""

    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gemini-2.5-flash"
    ) -> None:
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self._model = genai.GenerativeModel(model_name)

    async def generate_content(
        self, 
        prompt: str, 
        *, 
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> str:
        generation_config = kwargs.get("generation_config", {
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 4096,
        })
        
        if response_model:
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = response_model
        elif json_mode:
            generation_config["response_mime_type"] = "application/json"

        model = self._model
        if system_instruction:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )

        try:
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc)
            raise

    async def generate_stream(
        self, 
        prompt: str, 
        *, 
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 4096,
        }

        model = self._model
        if system_instruction:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )

        try:
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.error("Gemini streaming failed: %s", exc)
            raise
