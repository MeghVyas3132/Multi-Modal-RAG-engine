"""
FastAPI Gateway — single entry point for the RAG image search system.

Architecture decisions:
  1. ONE process, ONE worker. The CLIP model lives in this process's memory.
     Scaling is done horizontally (multiple containers), not vertically
     (multiple workers sharing GPU — that's a recipe for OOM).
  2. Startup hooks load CLIP and warm the model before accepting traffic.
     No request ever hits a cold model.
  3. The embedding and retrieval services are called in-process (function calls),
     not over HTTP/gRPC. This eliminates ~2-5ms of serialization + network hop
     per request. Service boundaries exist in code, not in network topology.
  4. Redis caching is optional and fail-open. If Redis is down, we serve
     live results with no degradation except latency for repeated queries.
  5. We use run_in_executor for the Qdrant search call because the
     qdrant-client is synchronous. This keeps the event loop free.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from configs.settings import get_settings
from utils.logger import setup_logging, get_logger
from utils.metrics import metrics
from utils.timing import timed

from services.embedding_service import create_embedder_auto
from services.retrieval_service.retriever import create_retriever, get_retriever
from services.api_gateway.models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    HealthResponse,
    StatsResponse,
)
from services.api_gateway.cache import init_cache, get_cached, set_cached, is_available as redis_available

_log = get_logger(__name__)

# Thread pool for offloading synchronous Qdrant calls.
# Size 4 is enough — each search is ~5ms, so 4 threads handle ~800 QPS.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qdrant")

# Module-level reference to the active embedder (PyTorch or ONNX).
# Set at startup by the lifespan hook. Used by /search and /health.
_active_embedder = None


# ── Lifespan: startup + shutdown ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load CLIP (PyTorch or ONNX), connect to Qdrant, warm everything.
    Shutdown: clean up thread pool.
    """
    global _active_embedder

    setup_logging(level="INFO")
    _log.info("startup_begin")

    cfg = get_settings()

    # 1. Load embedding model via factory (selects PyTorch or ONNX)
    with timed("startup_clip_load"):
        _active_embedder = create_embedder_auto()
    _log.info(
        "embedder_ready",
        backend="onnx" if cfg.use_onnx else "pytorch",
        device=str(_active_embedder.device),
        dim=_active_embedder.vector_dim,
    )

    # 2. Connect to Qdrant and ensure collection exists
    with timed("startup_qdrant_connect"):
        retriever = create_retriever()
        retriever.ensure_collection()
    _log.info("qdrant_ready")

    # 3. Initialize Redis cache (optional, fail-open)
    init_cache()

    # 4. Setup OpenTelemetry instrumentation if enabled
    if cfg.otel_enabled:
        try:
            from services.api_gateway.telemetry import setup_telemetry
            setup_telemetry(app)
            _log.info("otel_ready", endpoint=cfg.otel_endpoint)
        except Exception as e:
            _log.warning("otel_setup_failed", error=str(e))

    # 5. Setup auth and rate limiting middleware
    try:
        from services.api_gateway.middleware import setup_security
        setup_security(app)
    except Exception as e:
        _log.warning("security_setup_failed", error=str(e))

    _log.info("startup_complete")

    yield  # ← Application runs here

    # Shutdown
    _log.info("shutdown_begin")
    _executor.shutdown(wait=False)
    _log.info("shutdown_complete")


# ── FastAPI App ─────────────────────────────────────────────

app = FastAPI(
    title="Multi-Modal RAG Image Search",
    description="Ultra-low latency text-to-image retrieval with CLIP + HNSW",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
    docs_url="/docs",
    redoc_url=None,
)

# CORS — allow all origins for local dev. Lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Search Endpoint (HOT PATH) ─────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Text-to-image search. The critical path is:
      1. Check Redis cache (~0.5ms)
      2. CLIP text encode (~3-8ms GPU, ~20-50ms CPU)
      3. Qdrant HNSW search (~2-10ms)
      4. Assemble response (~0.1ms)
    Target: < 50ms total (excluding optional LLM).
    """
    total_start = time.perf_counter_ns()

    # ── Cache check ─────────────────────────────────────────
    cached = get_cached(request.query, request.top_k, request.filters)
    if cached:
        metrics.record("total_latency_ms", 0.1)  # Cache hits are ~0.1ms
        return SearchResponse(
            query=request.query,
            results=[SearchResult(**r) for r in cached["results"]],
            total=cached["total"],
            latency_ms=0.1,
            cached=True,
        )

    # ── Embed query ─────────────────────────────────────────
    embedder = _active_embedder
    with timed("search_embedding") as embed_t:
        query_vector = embedder.encode_text(request.query)

    # ── Vector search (offload to thread pool) ──────────────
    retriever = get_retriever()
    loop = asyncio.get_event_loop()

    with timed("search_retrieval") as search_t:
        results = await loop.run_in_executor(
            _executor,
            lambda: retriever.search(
                query_vector=query_vector,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                filter_conditions=request.filters,
            ),
        )

    # ── Assemble response ───────────────────────────────────
    total_ns = time.perf_counter_ns() - total_start
    total_ms = total_ns / 1_000_000

    metrics.record("total_latency_ms", total_ms)

    response = SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results],
        total=len(results),
        latency_ms=round(total_ms, 2),
    )

    # ── Cache the response (fire-and-forget) ────────────────
    set_cached(
        request.query,
        request.top_k,
        {
            "results": results,
            "total": len(results),
        },
        request.filters,
    )

    # ── Optional LLM explanation (async, non-blocking) ──────
    if request.include_explanation:
        cfg = get_settings()
        if cfg.llm_enabled:
            from services.llm_service.llm import generate_explanation
            try:
                explanation = await generate_explanation(request.query, results)
                response.explanation = explanation
            except Exception as e:
                _log.warning("llm_failed", error=str(e))

    _log.info(
        "search_complete",
        query=request.query,
        results=len(results),
        embedding_ms=round(embed_t["ms"], 2),
        search_ms=round(search_t["ms"], 2),
        total_ms=round(total_ms, 2),
    )

    return response


# ── Health Check ────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness + readiness probe. Returns component status.
    Used by Docker health checks and load balancers.
    """
    try:
        embedder = _active_embedder
        clip_ok = embedder is not None and embedder.is_ready
        device = str(embedder.device) if embedder else "unknown"
    except Exception:
        clip_ok = False
        device = "unknown"

    try:
        retriever = get_retriever()
        # Quick ping — collection info is cached by Qdrant client
        _ = retriever.collection_info()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    redis_ok = redis_available()

    status = "healthy" if (clip_ok and qdrant_ok) else "degraded"

    return HealthResponse(
        status=status,
        clip_loaded=clip_ok,
        qdrant_connected=qdrant_ok,
        redis_connected=redis_ok,
        device=device,
    )


# ── Stats Endpoint ──────────────────────────────────────────

@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """
    Runtime performance statistics: latency percentiles, counts, memory.
    Used for monitoring and debugging.
    """
    try:
        retriever = get_retriever()
        collection = retriever.collection_info()
    except Exception:
        collection = {}

    return StatsResponse(
        metrics=metrics.snapshot(),
        collection=collection,
    )


# ── Entry point for `uvicorn` ──────────────────────────────

def start_server() -> None:
    """Start the server programmatically (for scripts/CLI)."""
    import uvicorn
    cfg = get_settings()
    uvicorn.run(
        "services.api_gateway.app:app",
        host=cfg.api_host,
        port=cfg.api_port,
        workers=cfg.api_workers,  # Keep at 1 — see architecture notes
        log_level="info",
        access_log=False,  # We do our own structured logging
    )


if __name__ == "__main__":
    start_server()
