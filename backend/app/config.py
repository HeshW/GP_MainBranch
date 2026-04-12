"""Environment-driven settings (see ``.env.example`` in ``backend/``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

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
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000"
    )

    # RAG (optional — requires FAISS index + Gemini key on disk / env)
    use_rag: bool = False
    faiss_index_dir: Optional[str] = None
    clinicalbert_model_dir: Optional[str] = None
    gemini_api_key: Optional[str] = None
    rag_top_k: int = 7
    rag_translate_arabic: bool = True

    # Fine-tuned ClinicalBERT classifier (optional)
    use_finetuned_classifier: bool = False
    finetuned_model_dir: Optional[str] = None
    classifier_max_length: int = 256
    classifier_translate_arabic: bool = True

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
