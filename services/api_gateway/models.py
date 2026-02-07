"""
Request/Response models — strict validation at the API boundary.

Why Pydantic models instead of raw dicts?
  - Input validation is free (Pydantic V2 is compiled Rust).
  - Automatic OpenAPI schema generation for docs.
  - Type safety for downstream consumers.
  - Default values documented in one place.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Inbound search request from the client."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Natural language search query",
        examples=["a dog playing in the park"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters (e.g. {'category': 'nature'})",
    )
    include_explanation: bool = Field(
        default=False,
        description="If true and LLM is enabled, include an AI explanation",
    )


class SearchResult(BaseModel):
    """A single search result."""

    id: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Outbound search response."""

    query: str
    results: List[SearchResult]
    total: int
    latency_ms: float
    explanation: Optional[str] = None
    cached: bool = False


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    clip_loaded: bool
    text_embedder_loaded: bool
    qdrant_connected: bool
    redis_connected: bool
    device: str
    unified_embedder_loaded: bool = False
    vlm_loaded: bool = False
    graph_loaded: bool = False


class StatsResponse(BaseModel):
    """Runtime statistics response."""

    metrics: Dict[str, Any]
    collection: Dict[str, Any]
    text_collection: Optional[Dict[str, Any]] = None
    unified_collection: Optional[Dict[str, Any]] = None
    cache_stats: Optional[Dict[str, Any]] = None
    graph_stats: Optional[Dict[str, Any]] = None


# ── PDF RAG Models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    """Chat/RAG request with optional PDF context."""

    query: str
    top_k: int = 5
    include_images: bool = True
    score_threshold: float = 0.35
    filters: Optional[Dict[str, str]] = None


class PDFUploadResponse(BaseModel):
    """Response from PDF upload + indexing."""

    status: str
    filename: str
    total_pages: int
    chunks_indexed: int
    images_indexed: int
    latency_ms: float
    metadata: Dict[str, Any] = {}


# ── V2: Web Ingestion Models ───────────────────────────────

class WebIndexRequest(BaseModel):
    """Request to scrape and index a URL."""

    url: str = Field(
        ...,
        min_length=1,
        description="URL to scrape and index",
        examples=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"],
    )
    recursive: bool = Field(
        default=False,
        description="Crawl linked pages (up to max_pages)",
    )
    max_pages: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Max pages to crawl in recursive mode",
    )


class WebIndexResponse(BaseModel):
    """Response from web URL indexing."""

    status: str
    url: str
    title: str
    source_type: str
    chunks_indexed: int
    latency_ms: float


# ── V2: Graph Query Models ─────────────────────────────────

class GraphQueryRequest(BaseModel):
    """Request for graph traversal."""

    entity: str
    max_hops: int = Field(default=2, ge=1, le=5)
    max_results: int = Field(default=20, ge=1, le=100)


class GraphQueryResponse(BaseModel):
    """Graph traversal result."""

    entity: str
    related: List[Dict[str, Any]]


# ── V2: VLM Models ─────────────────────────────────────────

class VLMCaptionResponse(BaseModel):
    """Response from VLM image captioning."""

    caption: str
    confidence: float
    model: str
    latency_ms: float
