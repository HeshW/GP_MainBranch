from models.common.ai_provider import GeminiProvider, OpenRouterProvider
from models.common.provider_factory import (
    create_model_provider,
    normalize_provider_name,
    resolve_provider_config,
)


def test_normalize_provider_aliases_openai_compatible_to_openrouter():
    assert normalize_provider_name("openai-compatible") == "openrouter"


def test_resolve_provider_config_uses_legacy_gemini_values():
    provider, api_key, model_name = resolve_provider_config(
        llm_provider="gemini",
        llm_api_key=None,
        llm_model_name=None,
        gemini_api_key="legacy-gemini-key",
        gemini_model_name="gemini-2.5-flash-lite",
    )

    assert provider == "gemini"
    assert api_key == "legacy-gemini-key"
    assert model_name == "gemini-2.5-flash-lite"


def test_create_model_provider_requires_direct_key_for_openrouter():
    provider, model_provider, model_name = create_model_provider(
        llm_provider="openrouter",
        llm_api_key=None,
        llm_model_name=None,
        gemini_api_key="legacy-gemini-key",
        gemini_model_name="gemini-2.5-flash-lite",
    )

    assert provider == "openrouter"
    assert model_provider is None
    assert model_name == "openrouter/auto"


def test_create_model_provider_builds_openrouter_provider():
    provider, model_provider, model_name = create_model_provider(
        llm_provider="openrouter",
        llm_api_key="openrouter-key",
        llm_model_name="openrouter/auto",
    )

    assert provider == "openrouter"
    assert model_name == "openrouter/auto"
    assert isinstance(model_provider, OpenRouterProvider)


def test_create_model_provider_builds_gemini_provider_from_llm_key():
    provider, model_provider, model_name = create_model_provider(
        llm_provider="gemini",
        llm_api_key="gemini-key",
        llm_model_name="gemini-2.5-flash-lite",
    )

    assert provider == "gemini"
    assert model_name == "gemini-2.5-flash-lite"
    assert isinstance(model_provider, GeminiProvider)
