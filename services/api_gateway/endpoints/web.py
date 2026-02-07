"""
Web ingestion endpoints — URL scraping, indexing, and search grounding.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed

from services.api_gateway.models import WebIndexRequest, WebIndexResponse

_log = get_logger(__name__)
router = APIRouter(prefix="/web", tags=["web"])
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="web")


@router.post("/index", response_model=WebIndexResponse)
async def index_url(request: WebIndexRequest):
    """
    Scrape a URL, chunk the content, embed, and index into the unified collection.
    Supports: web pages, YouTube transcripts, GitHub READMEs.
    """
    cfg = get_settings()
    if not cfg.web_scraping_enabled:
        raise HTTPException(status_code=400, detail="Web scraping is disabled")

    total_start = time.perf_counter_ns()

    # --- Phase 1: Scrape (async I/O — await directly) ---
    from services.web_service.web_scraper import scrape_url
    content = await scrape_url(request.url)
    if not content or not content.content:
        raise HTTPException(status_code=422, detail="Failed to scrape URL")

    # --- Phase 2: Chunk + Embed + Index (CPU-bound — run in executor) ---
    loop = asyncio.get_event_loop()

    def _chunk_embed_index():
        from services.document_service.semantic_chunker import SemanticChunker
        chunker = SemanticChunker()
        chunks = chunker.auto_chunk(
            content.content,
            source=request.url,
            modality="text",
        )

        if not chunks:
            return 0

        texts = [c.content for c in chunks]

        if cfg.unified_enabled:
            from services.embedding_service.unified_embedder import get_unified_embedder
            from services.retrieval_service.hybrid_retriever import get_hybrid_retriever

            embedder = get_unified_embedder()
            vectors = embedder.encode_text_batch(texts)

            ids = []
            payloads = []
            for i, (text, chunk) in enumerate(zip(texts, chunks)):
                chunk_id = f"web:{content.content_hash}:{i}"
                point_id = int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)
                ids.append(point_id)
                payloads.append({
                    "text": text,
                    "source_url": request.url,
                    "source_title": content.title,
                    "source_type": content.source_type,
                    "chunk_index": i,
                    "modality": "text",
                    "type": "web_chunk",
                    "language": chunk.language,
                })

            hybrid = get_hybrid_retriever()
            hybrid.upsert_unified_batch(ids, vectors, payloads)
            return len(ids)
        else:
            from services.embedding_service.text_embedder import create_text_embedder
            from services.retrieval_service.retriever import get_retriever

            embedder = create_text_embedder()
            vectors = embedder.encode_batch(texts)

            ids = []
            payloads = []
            for i, text in enumerate(texts):
                chunk_id = f"web:{content.content_hash}:{i}"
                point_id = int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)
                ids.append(point_id)
                payloads.append({
                    "text": text,
                    "source_url": request.url,
                    "type": "web_chunk",
                })

            retriever = get_retriever()
            retriever.upsert_text_batch(ids, vectors, payloads)
            return len(ids)

    chunks_indexed = await loop.run_in_executor(_executor, _chunk_embed_index)

    # --- Phase 3: Entity extraction (async — fire and forget) ---
    if cfg.graph_enabled and chunks_indexed > 0:
        asyncio.create_task(_extract_entities_async(content, request.url))

    elapsed_ms = (time.perf_counter_ns() - total_start) / 1_000_000

    return WebIndexResponse(
        status="indexed",
        url=request.url,
        title=content.title,
        source_type=content.source_type,
        chunks_indexed=chunks_indexed,
        latency_ms=round(elapsed_ms, 2),
    )


async def _extract_entities_async(content, url: str):
    """Extract entities from web content and add to the knowledge graph."""
    try:
        from services.document_service.semantic_chunker import SemanticChunker
        chunker = SemanticChunker()
        chunks = chunker.auto_chunk(content.content, source=url, modality="text")
        if not chunks:
            return

        texts = [c.content for c in chunks[:10]]

        from services.graph_service.entity_extractor import extract_entities_batch
        entities_list = await extract_entities_batch(texts)

        from services.graph_service.knowledge_graph import get_knowledge_graph
        graph = get_knowledge_graph()

        for entities_data in entities_list:
            if entities_data:
                graph.add_entities(
                    entities_data.get("entities", []),
                    source=url,
                )
                graph.add_relationships(
                    entities_data.get("relationships", [])
                )
        graph.save()
    except Exception as e:
        _log.debug("web_entity_extraction_failed", error=str(e))


@router.post("/search-grounding")
async def search_grounding(query: str, threshold: float = 0.65):
    """
    If retrieval score is below threshold, fall back to web search
    for grounding. Returns web-augmented context.
    """
    cfg = get_settings()
    if not cfg.web_scraping_enabled:
        return {"grounded": False, "reason": "Web scraping disabled"}

    loop = asyncio.get_event_loop()

    # Phase 1: Check retrieval quality (CPU-bound — executor)
    def _check_retrieval():
        if cfg.unified_enabled:
            from services.embedding_service.unified_embedder import get_unified_embedder
            from services.retrieval_service.hybrid_retriever import get_hybrid_retriever

            embedder = get_unified_embedder()
            vec = embedder.encode_text(query)
            hybrid = get_hybrid_retriever()
            results = hybrid.search(query_vector=vec, top_k=3)
        else:
            from services.embedding_service.text_embedder import create_text_embedder
            from services.retrieval_service.retriever import get_retriever

            embedder = create_text_embedder()
            vec = embedder.encode_text(query)
            results = get_retriever().search_text(query_vector=vec, top_k=3)

        return results

    results = await loop.run_in_executor(_executor, _check_retrieval)

    if results and results[0]["score"] >= threshold:
        return {"grounded": True, "source": "local", "results": results}

    # Phase 2: Fall back to web (async I/O — await directly)
    try:
        from services.web_service.web_scraper import scrape_jina
        import urllib.parse

        search_url = f"https://s.jina.ai/{urllib.parse.quote(query)}"
        web_content = await scrape_jina(search_url)
        if web_content:
            return {
                "grounded": True,
                "source": "web",
                "content": web_content.content[:2000],
                "title": web_content.title,
            }
    except Exception as e:
        _log.warning("web_grounding_failed", error=str(e))

    return {"grounded": False, "reason": "No quality results found"}
