"""
Health, stats, and system endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter

from configs.settings import get_settings
from utils.logger import get_logger
from utils.metrics import metrics

from services.api_gateway.models import HealthResponse, StatsResponse

_log = get_logger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + readiness probe."""
    from services.api_gateway.app import get_active_embedder, get_text_embedder
    from services.api_gateway.cache import is_available as redis_available

    cfg = get_settings()

    # CLIP / Unified embedder
    try:
        embedder = get_active_embedder()
        clip_ok = embedder is not None and embedder.is_ready
        device = str(embedder.device) if embedder else "unknown"
    except Exception:
        clip_ok = False
        device = "unknown"

    # Text embedder
    try:
        te = get_text_embedder()
        text_ok = te is not None and te.is_ready
    except Exception:
        text_ok = False

    # Qdrant
    try:
        from services.retrieval_service.retriever import get_retriever
        get_retriever().collection_info()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    # Redis
    redis_ok = redis_available()

    # V2 components
    unified_ok = False
    vlm_ok = False
    graph_ok = False

    if cfg.unified_enabled:
        try:
            from services.embedding_service.unified_embedder import get_unified_embedder
            ue = get_unified_embedder()
            unified_ok = ue is not None
        except Exception:
            pass

    if cfg.vlm_enabled:
        try:
            from services.vlm_service.local_vlm import get_vlm
            vlm = get_vlm()
            vlm_ok = vlm is not None
        except Exception:
            pass

    if cfg.graph_enabled:
        try:
            from services.graph_service.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            graph_ok = kg is not None
        except Exception:
            pass

    status = "healthy" if (clip_ok and qdrant_ok and text_ok) else "degraded"

    return HealthResponse(
        status=status,
        clip_loaded=clip_ok,
        text_embedder_loaded=text_ok,
        qdrant_connected=qdrant_ok,
        redis_connected=redis_ok,
        device=device,
        unified_embedder_loaded=unified_ok,
        vlm_loaded=vlm_ok,
        graph_loaded=graph_ok,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Runtime performance statistics."""
    try:
        from services.retrieval_service.retriever import get_retriever
        retriever = get_retriever()
        collection = retriever.collection_info()
        text_collection = retriever.text_collection_info()
    except Exception:
        collection = {}
        text_collection = {}

    unified_collection = {}
    cfg = get_settings()
    if cfg.unified_enabled:
        try:
            from services.retrieval_service.hybrid_retriever import get_hybrid_retriever
            unified_collection = get_hybrid_retriever().collection_info()
        except Exception:
            pass

    cache_stats = {}
    if cfg.semantic_cache_enabled:
        try:
            from services.cache_service.semantic_cache import get_semantic_cache
            cache_stats = get_semantic_cache().stats()
        except Exception:
            pass

    graph_stats = {}
    if cfg.graph_enabled:
        try:
            from services.graph_service.knowledge_graph import get_knowledge_graph
            graph_stats = get_knowledge_graph().stats()
        except Exception:
            pass

    return StatsResponse(
        metrics=metrics.snapshot(),
        collection=collection,
        text_collection=text_collection,
        unified_collection=unified_collection,
        cache_stats=cache_stats,
        graph_stats=graph_stats,
    )
