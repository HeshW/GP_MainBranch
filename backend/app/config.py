"""Environment-driven settings (see ``.env.example`` in ``backend/``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.is_file() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_title: str = "GP Medical Analysis API"
    api_version: str = "1.0.0"
    max_upload_bytes: int = 10 * 1024 * 1024
    require_service_api_key: bool = False
    service_api_key: Optional[str] = None
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000"
    )

    # RAG / LLM configuration (optional)
    use_rag: bool = False
    faiss_index_dir: Optional[str] = None
    clinicalbert_model_dir: Optional[str] = None
    allow_unsafe_pickle_metadata: bool = False

    llm_provider: str = "gemini"
    llm_api_key: Optional[str] = None
    llm_model_name: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: Optional[str] = None
    openrouter_app_name: str = "GP Medical Analysis"

    # Legacy compatibility aliases.
    gemini_api_key: Optional[str] = None
    gemini_model_name: str = "gemini-2.5-flash-lite"
    openrouter_api_key: Optional[str] = None

    enable_therapy: bool = False
    rag_top_k: int = 5
    rag_translate_arabic: bool = True

    # Fine-tuned ClinicalBERT classifier (optional)
    use_finetuned_classifier: bool = False
    finetuned_model_dir: Optional[str] = None
    classifier_max_length: int = 256
    classifier_translate_arabic: bool = True

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @field_validator("llm_provider")
    @classmethod
    def _normalize_llm_provider(cls, value: str) -> str:
        normalized = str(value or "gemini").strip().lower()
        aliases = {
            "google": "gemini",
            "google-genai": "gemini",
            "openai-compatible": "openrouter",
            "openai_compatible": "openrouter",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"gemini", "openrouter"}:
            raise ValueError("LLM_PROVIDER must be one of: gemini, openrouter")
        return normalized

    @property
    def resolved_llm_provider(self) -> str:
        return self.llm_provider

    @property
    def resolved_llm_api_key(self) -> Optional[str]:
        direct_key = str(self.llm_api_key or "").strip()
        if direct_key:
            return direct_key
        if self.resolved_llm_provider == "gemini":
            legacy_key = str(self.gemini_api_key or "").strip()
            return legacy_key or None
        if self.resolved_llm_provider == "openrouter":
            legacy_key = str(self.openrouter_api_key or "").strip()
            return legacy_key or None
        return None

    @property
    def resolved_llm_model_name(self) -> str:
        direct_model = str(self.llm_model_name or "").strip()
        if direct_model:
            return direct_model
        if self.resolved_llm_provider == "gemini":
            return str(self.gemini_model_name or "gemini-2.5-flash-lite")
        return "openrouter/auto"


@lru_cache
def get_settings() -> Settings:
    return Settings()
