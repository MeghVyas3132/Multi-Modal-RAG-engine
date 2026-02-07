"""
Knowledge graph endpoints — entity queries, graph stats, management.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter

from configs.settings import get_settings
from utils.logger import get_logger

_log = get_logger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/stats")
async def graph_stats():
    """Return knowledge graph statistics."""
    cfg = get_settings()
    if not cfg.graph_enabled:
        return {"enabled": False}

    try:
        from services.graph_service.knowledge_graph import get_knowledge_graph
        graph = get_knowledge_graph()
        return {"enabled": True, **graph.stats()}
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@router.get("/related")
async def find_related(
    entity: str,
    max_hops: int = 2,
    max_results: int = 20,
):
    """Find entities related to a given entity in the graph."""
    cfg = get_settings()
    if not cfg.graph_enabled:
        return {"error": "Graph is disabled"}

    from services.graph_service.knowledge_graph import get_knowledge_graph
    graph = get_knowledge_graph()
    related = graph.find_related(
        entity, max_hops=max_hops, max_results=max_results
    )
    return {"entity": entity, "related": related}


@router.get("/expand")
async def expand_query(query: str):
    """
    Expand a query with graph entities — useful for debugging
    what the graph-augmented retriever will do.
    """
    cfg = get_settings()
    if not cfg.graph_enabled:
        return {"expansion_terms": []}

    from services.graph_service.knowledge_graph import get_knowledge_graph
    graph = get_knowledge_graph()

    entities = graph.find_entities_in_query(query)
    expansion = graph.expand_query(query)

    return {
        "query": query,
        "matched_entities": entities,
        "expansion_terms": expansion,
    }


@router.post("/save")
async def save_graph():
    """Persist the knowledge graph to disk."""
    from services.graph_service.knowledge_graph import get_knowledge_graph
    graph = get_knowledge_graph()
    graph.save()
    return {"status": "saved", **graph.stats()}
