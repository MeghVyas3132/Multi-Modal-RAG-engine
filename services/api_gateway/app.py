"""
FastAPI Gateway — thin router mounting V2 endpoint modules.

Architecture decisions:
  1. ONE process, ONE worker. Models live in-process — scale horizontally.
  2. Startup hooks load all embedders (unified + legacy), VLM, graph, cache.
  3. Endpoint logic is in services/api_gateway/endpoints/ modules.
  4. Module-level accessors (get_active_embedder, get_text_embedder) let
     endpoint modules access shared state without circular imports.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles

from configs.settings import get_settings
from utils.logger import setup_logging, get_logger
from utils.timing import timed

from services.api_gateway.cache import init_cache

_log = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qdrant")

# Module-level references set at startup
_active_embedder = None
_text_embedder = None


def get_active_embedder():
    """Get the active CLIP/ONNX embedder. Used by endpoint modules."""
    return _active_embedder


def get_text_embedder():
    """Get the text embedder. Used by endpoint modules."""
    return _text_embedder


# ── Lifespan: startup + shutdown ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load all models + connect services.
    Shutdown: persist graph + clean up.
    """
    global _active_embedder, _text_embedder

    setup_logging(level="INFO")
    _log.info("startup_begin")

    cfg = get_settings()

    # 1. Load CLIP embedding model (legacy, always needed for backward compat)
    with timed("startup_clip_load"):
        from services.embedding_service import create_embedder_auto
        _active_embedder = create_embedder_auto()
    _log.info(
        "embedder_ready",
        backend="onnx" if cfg.use_onnx else "pytorch",
        device=str(_active_embedder.device),
        dim=_active_embedder.vector_dim,
    )

    # 2. Load text embedder (sentence-transformers for PDF RAG)
    with timed("startup_text_embedder_load"):
        from services.embedding_service.text_embedder import create_text_embedder
        _text_embedder = create_text_embedder()
    _log.info("text_embedder_ready", dim=_text_embedder.vector_dim)

    # 3. Connect to Qdrant and ensure legacy collections exist
    with timed("startup_qdrant_connect"):
        from services.retrieval_service.retriever import create_retriever
        retriever = create_retriever()
        retriever.ensure_collection()
        retriever.ensure_text_collection()
    _log.info("qdrant_ready")

    # 4. Initialize Redis cache
    init_cache()

    # 5. V2: Load unified embedder (Jina-CLIP v2)
    if cfg.unified_enabled:
        try:
            with timed("startup_unified_embedder"):
                from services.embedding_service.unified_embedder import create_unified_embedder
                create_unified_embedder()
            _log.info("unified_embedder_ready", model=cfg.unified_model_name)
        except Exception as e:
            _log.warning("unified_embedder_failed", error=str(e))

    # 6. V2: Initialize hybrid retriever (creates unified collection)
    if cfg.unified_enabled:
        try:
            from services.retrieval_service.hybrid_retriever import get_hybrid_retriever
            get_hybrid_retriever()
            _log.info("hybrid_retriever_ready")
        except Exception as e:
            _log.warning("hybrid_retriever_failed", error=str(e))

    # 7. V2: Initialize knowledge graph
    if cfg.graph_enabled:
        try:
            from services.graph_service.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            _log.info("knowledge_graph_ready", **kg.stats())
        except Exception as e:
            _log.warning("knowledge_graph_failed", error=str(e))

    # 8. V2: Initialize semantic cache
    if cfg.semantic_cache_enabled:
        try:
            from services.cache_service.semantic_cache import get_semantic_cache
            get_semantic_cache()
            _log.info("semantic_cache_ready")
        except Exception as e:
            _log.warning("semantic_cache_failed", error=str(e))

    # 9. OpenTelemetry (optional)
    if cfg.otel_enabled:
        try:
            from services.api_gateway.telemetry import setup_telemetry
            setup_telemetry(app)
            _log.info("otel_ready", endpoint=cfg.otel_endpoint)
        except Exception as e:
            _log.warning("otel_setup_failed", error=str(e))

    # 10. Auth + rate limiting middleware
    try:
        from services.api_gateway.middleware import setup_security
        setup_security(app)
    except Exception as e:
        _log.warning("security_setup_failed", error=str(e))

    _log.info("startup_complete")

    yield  # ← Application runs here

    # ── Shutdown ────────────────────────────────────────────
    _log.info("shutdown_begin")

    # Persist knowledge graph
    if cfg.graph_enabled:
        try:
            from services.graph_service.knowledge_graph import get_knowledge_graph
            get_knowledge_graph().save()
        except Exception:
            pass

    _executor.shutdown(wait=False)
    _log.info("shutdown_complete")


# ── FastAPI App ─────────────────────────────────────────────

app = FastAPI(
    title="Multi-Modal RAG Engine",
    description="Production V2: Unified embeddings, VLM, Knowledge Graph, Semantic Cache, Web Grounding",
    version="2.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
    docs_url="/docs",
    redoc_url=None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for images
_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
if _data_dir.exists():
    app.mount("/images", StaticFiles(directory=str(_data_dir)), name="images")

# ── Mount endpoint routers ──────────────────────────────────

from services.api_gateway.endpoints.search import router as search_router
from services.api_gateway.endpoints.chat import router as chat_router
from services.api_gateway.endpoints.upload import router as upload_router
from services.api_gateway.endpoints.web import router as web_router
from services.api_gateway.endpoints.graph import router as graph_router
from services.api_gateway.endpoints.health import router as health_router

app.include_router(search_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(web_router)
app.include_router(graph_router)
app.include_router(health_router)


# ── Entry point ─────────────────────────────────────────────

def start_server() -> None:
    """Start the server programmatically."""
    import uvicorn
    cfg = get_settings()
    uvicorn.run(
        "services.api_gateway.app:app",
        host=cfg.api_host,
        port=cfg.api_port,
        workers=cfg.api_workers,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    start_server()
