"""
Search endpoints — text-to-image, image-to-image, and unified hybrid search.
"""

from __future__ import annotations

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from configs.settings import get_settings
from utils.logger import get_logger
from utils.metrics import metrics
from utils.timing import timed

from services.api_gateway.models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)

_log = get_logger(__name__)
router = APIRouter(tags=["search"])
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="search")


def _get_embedder():
    """Get the active embedder (unified or legacy)."""
    from services.api_gateway.app import get_active_embedder
    return get_active_embedder()


def _get_text_embedder():
    """Get the text embedder."""
    from services.api_gateway.app import get_text_embedder
    return get_text_embedder()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Unified search — routes through modality router, uses hybrid retriever
    with graph expansion and cross-encoder reranking.

    Pipeline:
      1. Semantic cache check (~0.01ms L1 / ~0.5ms L2 / ~5ms L3)
      2. Modality routing (~0.01ms heuristic / ~100ms LLM)
      3. Unified embedding (~10-50ms)
      4. Hybrid search with graph expansion (~5-15ms)
      5. Cross-encoder reranking (~100ms for 50 candidates)
      6. Cache store (fire-and-forget)
    """
    total_start = time.perf_counter_ns()
    cfg = get_settings()

    # ── 1. Semantic cache check ─────────────────────────────
    if cfg.semantic_cache_enabled:
        try:
            from services.cache_service.semantic_cache import get_semantic_cache
            cache = get_semantic_cache()
            cached = cache.get(request.query, search_type="search")
            if cached:
                metrics.record("total_latency_ms", 0.1)
                return SearchResponse(
                    query=request.query,
                    results=[SearchResult(**r) for r in cached.get("results", [])],
                    total=cached.get("total", 0),
                    latency_ms=0.1,
                    cached=True,
                )
        except Exception as e:
            _log.debug("cache_check_error", error=str(e))

    # ── Legacy cache fallback ───────────────────────────────
    from services.api_gateway.cache import get_cached, set_cached
    cached = get_cached(request.query, request.top_k, request.filters)
    if cached:
        metrics.record("total_latency_ms", 0.1)
        return SearchResponse(
            query=request.query,
            results=[SearchResult(**r) for r in cached["results"]],
            total=cached["total"],
            latency_ms=0.1,
            cached=True,
        )

    # ── 2. Modality routing ─────────────────────────────────
    modality_weights = None
    if cfg.router_enabled:
        try:
            from services.routing_service.modality_router import route_query
            modality_weights = route_query(request.query)
        except Exception as e:
            _log.debug("router_error", error=str(e))

    # ── 3. Embed query ──────────────────────────────────────
    loop = asyncio.get_event_loop()

    if cfg.unified_enabled:
        # Unified embedding — single vector for both text and image search
        try:
            from services.embedding_service.unified_embedder import get_unified_embedder
            embedder = get_unified_embedder()

            with timed("search_embedding") as embed_t:
                query_vector = await loop.run_in_executor(
                    _executor, lambda: embedder.encode_text(request.query)
                )

            # ── 4. Hybrid search ────────────────────────────
            from services.retrieval_service.hybrid_retriever import get_hybrid_retriever
            hybrid = get_hybrid_retriever()

            with timed("search_retrieval") as search_t:
                results = await loop.run_in_executor(
                    _executor,
                    lambda: hybrid.search(
                        query_vector=query_vector,
                        top_k=cfg.retrieval_top_k_initial,
                        score_threshold=request.score_threshold,
                        modality_weights=modality_weights,
                        filter_conditions=request.filters,
                        expand_with_graph=cfg.graph_enabled,
                        query_text=request.query,
                    ),
                )

            # ── 5. Reranking ────────────────────────────────
            if cfg.reranker_enabled and results:
                try:
                    from services.retrieval_service.reranker import get_reranker
                    reranker = get_reranker()
                    with timed("search_reranking") as rerank_t:
                        results = await loop.run_in_executor(
                            _executor,
                            lambda: reranker.rerank(
                                request.query, results, top_n=request.top_k
                            ),
                        )
                except Exception as e:
                    _log.warning("reranker_error", error=str(e))
                    results = results[:request.top_k]
            else:
                results = results[:request.top_k]

        except Exception as e:
            _log.warning("unified_search_fallback", error=str(e))
            # Fall through to legacy search
            return await _legacy_search(request, total_start)
    else:
        return await _legacy_search(request, total_start)

    # ── Assemble response ───────────────────────────────────
    total_ms = (time.perf_counter_ns() - total_start) / 1_000_000
    metrics.record("total_latency_ms", total_ms)

    response = SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results],
        total=len(results),
        latency_ms=round(total_ms, 2),
    )

    # ── 6. Cache store ──────────────────────────────────────
    cache_data = {"results": results, "total": len(results)}
    set_cached(request.query, request.top_k, cache_data, request.filters)

    if cfg.semantic_cache_enabled:
        try:
            from services.cache_service.semantic_cache import get_semantic_cache
            cache = get_semantic_cache()
            cache.put(request.query, cache_data, search_type="search")
        except Exception:
            pass

    # ── Optional LLM explanation ────────────────────────────
    if request.include_explanation and cfg.llm_enabled:
        try:
            from services.llm_service.llm import generate_explanation
            explanation = await generate_explanation(request.query, results)
            response.explanation = explanation
        except Exception as e:
            _log.warning("llm_explanation_failed", error=str(e))

    _log.info(
        "search_complete",
        query=request.query,
        results=len(results),
        total_ms=round(total_ms, 2),
        unified=True,
    )

    return response


async def _legacy_search(request: SearchRequest, total_start: int) -> SearchResponse:
    """Fallback to legacy CLIP + Qdrant search."""
    loop = asyncio.get_event_loop()
    embedder = _get_embedder()

    with timed("search_embedding") as embed_t:
        query_vector = await loop.run_in_executor(
            _executor, lambda: embedder.encode_text(request.query)
        )

    from services.retrieval_service.retriever import get_retriever
    retriever = get_retriever()

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

    total_ms = (time.perf_counter_ns() - total_start) / 1_000_000
    metrics.record("total_latency_ms", total_ms)

    from services.api_gateway.cache import set_cached
    set_cached(request.query, request.top_k, {"results": results, "total": len(results)}, request.filters)

    response = SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results],
        total=len(results),
        latency_ms=round(total_ms, 2),
    )

    if request.include_explanation:
        cfg = get_settings()
        if cfg.llm_enabled:
            try:
                from services.llm_service.llm import generate_explanation
                explanation = await generate_explanation(request.query, results)
                response.explanation = explanation
            except Exception as e:
                _log.warning("llm_failed", error=str(e))

    return response


@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = 10,
    score_threshold: float = 0.0,
) -> SearchResponse:
    """Upload an image and find visually similar items."""
    from PIL import Image

    total_start = time.perf_counter_ns()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    cfg = get_settings()
    loop = asyncio.get_event_loop()

    if cfg.unified_enabled:
        from services.embedding_service.unified_embedder import get_unified_embedder
        embedder = get_unified_embedder()

        def _encode_and_search():
            vec = embedder.encode_image(img)
            from services.retrieval_service.hybrid_retriever import get_hybrid_retriever
            hybrid = get_hybrid_retriever()
            return hybrid.search(
                query_vector=vec, top_k=top_k, score_threshold=score_threshold
            )
    else:
        embedder = _get_embedder()

        def _encode_and_search():
            preprocessed = embedder.preprocess(img)
            vec = embedder.encode_images([preprocessed])[0]
            from services.retrieval_service.retriever import get_retriever
            return get_retriever().search(
                query_vector=vec, top_k=top_k, score_threshold=score_threshold
            )

    with timed("image_search") as t:
        results = await loop.run_in_executor(_executor, _encode_and_search)

    total_ms = (time.perf_counter_ns() - total_start) / 1_000_000

    return SearchResponse(
        query=f"[image: {file.filename}]",
        results=[SearchResult(**r) for r in results],
        total=len(results),
        latency_ms=round(total_ms, 2),
    )
