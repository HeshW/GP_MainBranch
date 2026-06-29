"""Factory helpers for selecting an LLM provider from environment-style settings."""

from __future__ import annotations

from typing import Optional

from models.common.ai_provider import BaseModelProvider, GeminiProvider, OpenRouterProvider

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OPENROUTER_MODEL = "openrouter/auto"


def _clean(value: Optional[str]) -> str:
    return str(value or "").strip()


def normalize_provider_name(provider_name: Optional[str]) -> str:
    normalized = _clean(provider_name).lower() or "gemini"
    aliases = {
        "google": "gemini",
        "google-genai": "gemini",
        "openai-compatible": "openrouter",
        "openai_compatible": "openrouter",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in {"gemini", "openrouter"}:
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider_name}'. Supported values: gemini, openrouter")
    return normalized


def resolve_provider_config(
    *,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model_name: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
) -> tuple[str, Optional[str], str]:
    provider_name = normalize_provider_name(llm_provider)
    llm_key = _clean(llm_api_key)
    gemini_key = _clean(gemini_api_key)
    openrouter_key = _clean(openrouter_api_key)

    if provider_name == "gemini":
        resolved_api_key = llm_key or gemini_key
    else:
        resolved_api_key = llm_key or openrouter_key

    llm_model = _clean(llm_model_name)
    gemini_model = _clean(gemini_model_name) or DEFAULT_GEMINI_MODEL

    if llm_model:
        resolved_model = llm_model
    elif provider_name == "gemini":
        resolved_model = gemini_model
    else:
        resolved_model = DEFAULT_OPENROUTER_MODEL

    return provider_name, (resolved_api_key or None), resolved_model


def create_model_provider(
    *,
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model_name: Optional[str] = None,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_site_url: Optional[str] = None,
    openrouter_app_name: str = "GP Medical Analysis",
    openrouter_api_key: Optional[str] = None,
) -> tuple[str, Optional[BaseModelProvider], str]:
    provider_name, api_key, model_name = resolve_provider_config(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model_name=llm_model_name,
        gemini_api_key=gemini_api_key,
        gemini_model_name=gemini_model_name,
        openrouter_api_key=openrouter_api_key,
    )

    if not api_key:
        return provider_name, None, model_name

    if provider_name == "gemini":
        return provider_name, GeminiProvider(api_key=api_key, model_name=model_name), model_name

    return (
        provider_name,
        OpenRouterProvider(
            api_key=api_key,
            model_name=model_name,
            base_url=openrouter_base_url,
            site_url=openrouter_site_url,
            app_name=openrouter_app_name,
        ),
        model_name,
    )
