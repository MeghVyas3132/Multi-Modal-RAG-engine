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
    qdrant_connected: bool
    redis_connected: bool
    device: str


class StatsResponse(BaseModel):
    """Runtime statistics response."""

    metrics: Dict[str, Any]
    collection: Dict[str, Any]
