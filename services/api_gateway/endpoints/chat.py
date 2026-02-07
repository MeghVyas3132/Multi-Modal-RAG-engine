"""
Chat/RAG endpoint — SSE streaming with retrieval + LLM.
"""

from __future__ import annotations

import asyncio
import json as _json
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from configs.settings import get_settings
from utils.logger import get_logger
from utils.metrics import metrics
from utils.timing import timed

from services.api_gateway.models import ChatRequest

_log = get_logger(__name__)
router = APIRouter(tags=["chat"])
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chat")


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    RAG chat with SSE streaming.

    V2 pipeline:
      1. Semantic cache check for query
      2. Modality routing (which collections to search)
      3. Unified/hybrid retrieval with graph expansion
      4. Cross-encoder reranking
      5. LLM response cache check
      6. Stream LLM answer (or serve from cache)
    """
    cfg = get_settings()

    async def event_generator():
        retrieval_start = time.perf_counter_ns()
        loop = asyncio.get_event_loop()

        text_results = []
        image_results = []

        # ── Modality routing ────────────────────────────────
        modality_weights = None
        if cfg.router_enabled:
            try:
                from services.routing_service.modality_router import (
                    route_query,
                    should_search_modality,
                )
                modality_weights = route_query(request.query)
            except Exception:
                pass

        # ── Unified search path ─────────────────────────────
        if cfg.unified_enabled:
            try:
                from services.embedding_service.unified_embedder import get_unified_embedder
                from services.retrieval_service.hybrid_retriever import get_hybrid_retriever

                embedder = get_unified_embedder()
                hybrid = get_hybrid_retriever()

                query_vec = await loop.run_in_executor(
                    _executor, lambda: embedder.encode_text(request.query)
                )

                all_results = await loop.run_in_executor(
                    _executor,
                    lambda: hybrid.search(
                        query_vector=query_vec,
                        top_k=request.top_k * 2,
                        score_threshold=request.score_threshold,
                        modality_weights=modality_weights,
                        expand_with_graph=cfg.graph_enabled,
                        query_text=request.query,
                    ),
                )

                # Rerank
                if cfg.reranker_enabled and all_results:
                    try:
                        from services.retrieval_service.reranker import get_reranker
                        reranker = get_reranker()
                        all_results = await loop.run_in_executor(
                            _executor,
                            lambda: reranker.rerank(
                                request.query, all_results, top_n=request.top_k
                            ),
                        )
                    except Exception:
                        all_results = all_results[:request.top_k]

                # Split by modality for backward-compatible response
                for r in all_results:
                    mod = r.get("modality", "text")
                    if mod == "image":
                        image_results.append(r)
                    else:
                        text_results.append(r)

            except Exception as e:
                _log.warning("unified_chat_fallback", error=str(e))
                # Fall through to legacy path
                text_results, image_results = await _legacy_retrieval(
                    request, loop
                )
        else:
            text_results, image_results = await _legacy_retrieval(
                request, loop
            )

        retrieval_ms = (time.perf_counter_ns() - retrieval_start) / 1_000_000

        # Emit Phase 1: retrieval results
        yield {
            "event": "retrieval",
            "data": _json.dumps({
                "text_results": text_results,
                "image_results": image_results,
                "latency_ms": round(retrieval_ms, 2),
            }),
        }

        # ── LLM response cache check ───────────────────────
        if cfg.semantic_cache_enabled:
            try:
                import hashlib
                from services.cache_service.semantic_cache import get_semantic_cache
                cache = get_semantic_cache()
                context_hash = hashlib.md5(
                    _json.dumps(text_results[:5], sort_keys=True).encode()
                ).hexdigest()
                cached_answer = cache.get_llm_response(request.query, context_hash)
                if cached_answer:
                    yield {
                        "event": "token",
                        "data": _json.dumps({"token": cached_answer}),
                    }
                    yield {
                        "event": "done",
                        "data": _json.dumps({"status": "complete", "cached": True}),
                    }
                    return
            except Exception:
                pass

        # ── Phase 2: Stream LLM answer ──────────────────────
        from services.llm_service.llm import stream_chat as stream_llm

        full_answer = []
        try:
            async for token in stream_llm(
                query=request.query,
                text_chunks=text_results,
                image_results=image_results,
            ):
                full_answer.append(token)
                yield {
                    "event": "token",
                    "data": _json.dumps({"token": token}),
                }
        except Exception as e:
            _log.error("llm_stream_failed", error=str(e))
            yield {
                "event": "error",
                "data": _json.dumps({"error": str(e)}),
            }

        # Cache the LLM response
        if cfg.semantic_cache_enabled and full_answer:
            try:
                import hashlib
                from services.cache_service.semantic_cache import get_semantic_cache
                cache = get_semantic_cache()
                context_hash = hashlib.md5(
                    _json.dumps(text_results[:5], sort_keys=True).encode()
                ).hexdigest()
                cache.put_llm_response(
                    request.query, context_hash, "".join(full_answer)
                )
            except Exception:
                pass

        yield {
            "event": "done",
            "data": _json.dumps({"status": "complete"}),
        }

    return EventSourceResponse(event_generator())


async def _legacy_retrieval(request, loop):
    """Legacy dual-collection retrieval."""
    from services.api_gateway.app import get_active_embedder, get_text_embedder
    from services.retrieval_service.retriever import get_retriever

    text_results = []
    image_results = []
    retriever = get_retriever()

    try:
        text_embedder = get_text_embedder()
        text_vector = await loop.run_in_executor(
            _executor, lambda: text_embedder.encode_text(request.query)
        )
        text_results = await loop.run_in_executor(
            _executor,
            lambda: retriever.search_text(
                query_vector=text_vector,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                filter_conditions=request.filters,
            ),
        )
    except Exception as e:
        _log.warning("text_search_failed", error=str(e))

    if request.include_images:
        try:
            embedder = get_active_embedder()
            img_vector = await loop.run_in_executor(
                _executor, lambda: embedder.encode_text(request.query)
            )
            image_results = await loop.run_in_executor(
                _executor,
                lambda: retriever.search(
                    query_vector=img_vector,
                    top_k=request.top_k,
                    score_threshold=request.score_threshold,
                    filter_conditions=request.filters,
                ),
            )
        except Exception as e:
            _log.warning("image_search_failed", error=str(e))

    return text_results, image_results
