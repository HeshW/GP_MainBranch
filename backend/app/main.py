"""FastAPI entrypoint. Run from repository root::

    uvicorn app.main:app --reload --app-dir backend
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Repository root (parent of ``backend/``) — required for ``manager`` / ``models``.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import health, pipeline, chat

logger = logging.getLogger(__name__)


def _is_existing_dir(path_value: str | None) -> bool:
    return bool(path_value and Path(path_value).is_dir())


def _resolve_optional_ai_flags(settings) -> tuple[bool, bool, list[str]]:
    warnings: list[str] = []
    use_rag = bool(settings.use_rag)
    use_classifier = bool(settings.use_finetuned_classifier)

    if use_rag and not _is_existing_dir(settings.faiss_index_dir):
        warnings.append(
            "USE_RAG is enabled but FAISS_INDEX_DIR is missing/unavailable; disabling RAG for this startup."
        )
        use_rag = False
    if use_rag and not _is_existing_dir(settings.clinicalbert_model_dir):
        warnings.append(
            "USE_RAG is enabled but CLINICALBERT_MODEL_DIR is missing/unavailable; disabling RAG for this startup."
        )
        use_rag = False

    if use_classifier and not _is_existing_dir(settings.finetuned_model_dir):
        warnings.append(
            "USE_FINETUNED_CLASSIFIER is enabled but FINETUNED_MODEL_DIR is missing/unavailable; "
            "disabling classifier for this startup."
        )
        use_classifier = False

    return use_rag, use_classifier, warnings


@asynccontextmanager
async def lifespan(app: FastAPI):
    from manager.chat_manager import ChatManager

    s = get_settings()
    use_rag, use_classifier, startup_warnings = _resolve_optional_ai_flags(s)
    for item in startup_warnings:
        logger.warning(item)

    try:
        app.state.chat_manager = ChatManager(
            use_rag=use_rag,
            faiss_index_dir=s.faiss_index_dir,
            clinicalbert_model_dir=s.clinicalbert_model_dir,
            allow_unsafe_pickle_metadata=s.allow_unsafe_pickle_metadata,
            gemini_api_key=s.gemini_api_key,
            gemini_model_name=getattr(s, "gemini_model_name", "gemini-2.5-flash-lite"),
            rag_top_k=s.rag_top_k,
            rag_translate_arabic=s.rag_translate_arabic,
            use_finetuned_classifier=use_classifier,
            finetuned_model_dir=s.finetuned_model_dir,
            classifier_max_length=s.classifier_max_length,
            classifier_translate_arabic=s.classifier_translate_arabic,
        )
    except Exception as exc:
        logger.exception(
            "ChatManager advanced initialization failed; continuing in degraded mode. error=%s",
            exc,
        )
        app.state.chat_manager = ChatManager(
            use_rag=False,
            faiss_index_dir=s.faiss_index_dir,
            clinicalbert_model_dir=s.clinicalbert_model_dir,
            allow_unsafe_pickle_metadata=s.allow_unsafe_pickle_metadata,
            gemini_api_key=s.gemini_api_key,
            gemini_model_name=getattr(s, "gemini_model_name", "gemini-2.5-flash-lite"),
            rag_top_k=s.rag_top_k,
            rag_translate_arabic=s.rag_translate_arabic,
            use_finetuned_classifier=False,
            finetuned_model_dir=s.finetuned_model_dir,
            classifier_max_length=s.classifier_max_length,
            classifier_translate_arabic=s.classifier_translate_arabic,
        )
        use_rag = False
        use_classifier = False

    app.state.runtime_features = {
        "rag_enabled": use_rag,
        "finetuned_classifier_enabled": use_classifier,
    }
    yield


def create_app() -> FastAPI:
    s = get_settings()
    application = FastAPI(
        title=s.api_title,
        version=s.api_version,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_audit_middleware(request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request_error method=%s path=%s request_id=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                request_id,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "request method=%s path=%s status=%s request_id=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
            elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    application.include_router(health.router, prefix="/api/v1")
    application.include_router(pipeline.router, prefix="/api/v1")
    application.include_router(chat.router, prefix="/api/v1")
    return application


app = create_app()


@app.get("/")
def root() -> dict:
    return {
        "service": "gp-medical-api",
        "docs": "/api/docs",
        "health": "/api/v1/health",
    }
