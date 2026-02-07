"""
Multi-tier caching — L1 in-process, L2 Redis, L3 semantic similarity.
"""

from services.cache_service.semantic_cache import (
    SemanticCache,
    get_semantic_cache,
)
from services.cache_service.deduplication import DeduplicationService

__all__ = ["SemanticCache", "get_semantic_cache", "DeduplicationService"]
