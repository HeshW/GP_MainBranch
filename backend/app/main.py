"""FastAPI entrypoint. Run from repository root::

    uvicorn app.main:app --reload --app-dir backend
"""

from __future__ import annotations

import sys
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    from manager.chat_manager import ChatManager

    s = get_settings()
    app.state.chat_manager = ChatManager(
        use_rag=s.use_rag,
        faiss_index_dir=s.faiss_index_dir,
        clinicalbert_model_dir=s.clinicalbert_model_dir,
        gemini_api_key=s.gemini_api_key,
        rag_top_k=s.rag_top_k,
        rag_translate_arabic=s.rag_translate_arabic,
        use_finetuned_classifier=s.use_finetuned_classifier,
        finetuned_model_dir=s.finetuned_model_dir,
        classifier_max_length=s.classifier_max_length,
        classifier_translate_arabic=s.classifier_translate_arabic,
    )
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
