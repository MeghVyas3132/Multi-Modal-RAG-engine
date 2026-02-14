"""
Multi-tier Semantic Cache — L1 in-process, L2 Redis, L3 embedding similarity.

Architecture decisions:
  1. L1: OrderedDict LRU (500 entries, ~50MB RAM). Hit latency: 0.01ms.
  2. L2: Redis hash with TTL. Hit latency: 0.5ms over loopback.
  3. L3: Semantic — embed the query, compare against cached query embeddings.
     Uses Qdrant collection `cache_vectors` for fast ANN.
     Hit if cosine > 0.93. Latency: ~5ms (vector search).
  4. Cache key hierarchy:
     L1/L2: exact MD5 of normalized query → instant lookup.
     L3:  embed query → ANN search in cache collection → if top-1 score > 0.93, return cached.
  5. LLM response caching: cache the final LLM answer keyed by (query, context_hash).
     7-day TTL. Saves $$ on LLM API calls.
  6. All cache operations are fail-safe: errors never break the search path.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed

_log = get_logger(__name__)


class _LRUCache:
    """Thread-safe LRU cache backed by OrderedDict."""

    def __init__(self, maxsize: int = 500) -> None:
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
        }


class SemanticCache:
    """
    Multi-tier cache with semantic similarity matching.
    
    Lookup order:
      1. L1 in-process LRU (exact match)
      2. L2 Redis (exact match)  
      3. L3 Qdrant semantic ANN (cosine > threshold)
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._enabled = cfg.semantic_cache_enabled
        self._similarity_threshold = cfg.semantic_cache_similarity
        self._l2_ttl = cfg.semantic_cache_l2_ttl
        self._llm_ttl = cfg.llm_cache_ttl

        # L1: in-process LRU
        self._l1 = _LRUCache(maxsize=cfg.semantic_cache_l1_size)

        # L2: Redis
        self._redis = None
        self._redis_available = False
        self._init_redis()

        # L3: Qdrant semantic cache collection
        self._qdrant_client = None
        self._cache_collection = "cache_vectors"
        self._embedder = None
        self._init_semantic_layer()

    def _init_redis(self) -> None:
        """Initialize Redis connection for L2 cache."""
        cfg = get_settings()
        if not cfg.redis_enabled:
            return
        try:
            import redis
            self._redis = redis.Redis(
                host=cfg.redis_host,
                port=cfg.redis_port,
                db=cfg.redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=1,
            )
            self._redis.ping()
            self._redis_available = True
            _log.info("semantic_cache_redis_connected")
        except Exception as e:
            _log.warning("semantic_cache_redis_unavailable", error=str(e))

    def _init_semantic_layer(self) -> None:
        """Initialize Qdrant collection for L3 semantic cache."""
        cfg = get_settings()
        if not cfg.semantic_cache_enabled:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._qdrant_client = QdrantClient(
                host=cfg.qdrant_host, port=cfg.qdrant_port
            )

            # Determine vector dimension
            vector_dim = cfg.unified_vector_dim if cfg.unified_enabled else cfg.text_vector_dim

            # Create cache collection if needed
            collections = [
                c.name for c in self._qdrant_client.get_collections().collections
            ]
            if self._cache_collection not in collections:
                self._qdrant_client.create_collection(
                    collection_name=self._cache_collection,
                    vectors_config=VectorParams(
                        size=vector_dim,
                        distance=Distance.COSINE,
                    ),
                )
                _log.info("semantic_cache_collection_created", dim=vector_dim)

        except Exception as e:
            _log.warning("semantic_cache_qdrant_unavailable", error=str(e))
            self._qdrant_client = None

    def _get_embedder(self):
        """Lazy-load embedder for semantic cache."""
        if self._embedder is not None:
            return self._embedder
        cfg = get_settings()
        if cfg.unified_enabled:
            from services.embedding_service.unified_embedder import get_unified_embedder
            self._embedder = get_unified_embedder()
        else:
            from services.embedding_service.text_embedder import TextEmbedder
            self._embedder = TextEmbedder()
        return self._embedder

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query for consistent cache keys."""
        return " ".join(query.lower().strip().split())

    @staticmethod
    def _cache_key(query: str, prefix: str = "rag:sem") -> str:
        """Deterministic cache key."""
        normalized = SemanticCache._normalize_query(query)
        return f"{prefix}:{hashlib.md5(normalized.encode()).hexdigest()}"

    # ── Lookup ──────────────────────────────────────────────

    def get(
        self,
        query: str,
        search_type: str = "search",
    ) -> Optional[Dict[str, Any]]:
        """
        Multi-tier cache lookup.
        Returns cached response dict or None.
        """
        if not self._enabled:
            return None

        key = self._cache_key(query, prefix=f"rag:{search_type}")

        # L1: in-process exact match
        result = self._l1.get(key)
        if result is not None:
            _log.debug("cache_hit", tier="L1", query=query[:50])
            return result

        # L2: Redis exact match
        result = self._get_redis(key)
        if result is not None:
            _log.debug("cache_hit", tier="L2", query=query[:50])
            self._l1.put(key, result)  # Promote to L1
            return result

        # L3: Semantic similarity
        result = self._get_semantic(query, search_type)
        if result is not None:
            _log.debug("cache_hit", tier="L3", query=query[:50])
            self._l1.put(key, result)  # Promote to L1
            return result

        return None

    def _get_redis(self, key: str) -> Optional[Dict[str, Any]]:
        """L2 Redis lookup."""
        if not self._redis_available or self._redis is None:
            return None
        try:
            data = self._redis.get(key)
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    def _get_semantic(
        self,
        query: str,
        search_type: str,
    ) -> Optional[Dict[str, Any]]:
        """L3 Qdrant semantic similarity lookup."""
        if self._qdrant_client is None:
            return None

        try:
            embedder = self._get_embedder()
            cfg = get_settings()

            # Embed query
            if cfg.unified_enabled:
                query_vec = embedder.encode_text(query).tolist()
            else:
                query_vec = embedder.encode_text(query).tolist()

            # ANN search in cache collection
            results = self._qdrant_client.search(
                collection_name=self._cache_collection,
                query_vector=query_vec,
                limit=1,
                score_threshold=self._similarity_threshold,
            )

            if results and results[0].score >= self._similarity_threshold:
                payload = results[0].payload
                if payload.get("search_type") == search_type:
                    cached_response = json.loads(payload.get("response", "{}"))
                    if cached_response:
                        return cached_response
        except Exception as e:
            _log.debug("semantic_cache_lookup_error", error=str(e))

        return None

    # ── Store ───────────────────────────────────────────────

    def put(
        self,
        query: str,
        response: Dict[str, Any],
        search_type: str = "search",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store response in all cache tiers.
        """
        if not self._enabled:
            return

        key = self._cache_key(query, prefix=f"rag:{search_type}")

        # L1
        self._l1.put(key, response)

        # L2: Redis
        self._put_redis(key, response, ttl)

        # L3: Semantic (async-safe, fire and forget)
        self._put_semantic(query, response, search_type)

    def _put_redis(
        self,
        key: str,
        response: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Store in Redis with TTL."""
        if not self._redis_available or self._redis is None:
            return
        try:
            self._redis.setex(
                key,
                ttl or self._l2_ttl,
                json.dumps(response),
            )
        except Exception:
            pass

    def _put_semantic(
        self,
        query: str,
        response: Dict[str, Any],
        search_type: str,
    ) -> None:
        """Store query embedding + response in Qdrant for semantic matching."""
        if self._qdrant_client is None:
            return

        try:
            from qdrant_client.models import PointStruct
            import uuid

            embedder = self._get_embedder()
            cfg = get_settings()

            if cfg.unified_enabled:
                query_vec = embedder.encode_text(query).tolist()
            else:
                query_vec = embedder.encode_text(query).tolist()

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=query_vec,
                payload={
                    "query": query,
                    "search_type": search_type,
                    "response": json.dumps(response),
                    "timestamp": time.time(),
                },
            )

            self._qdrant_client.upsert(
                collection_name=self._cache_collection,
                points=[point],
            )

        except Exception as e:
            _log.debug("semantic_cache_store_error", error=str(e))

    # ── LLM Response Cache ──────────────────────────────────

    def get_llm_response(
        self,
        query: str,
        context_hash: str,
    ) -> Optional[str]:
        """
        Check cache for a previously generated LLM response.
        Key = hash(query + context_hash) to ensure context consistency.
        """
        if not self._enabled:
            return None

        key = f"rag:llm:{hashlib.md5(f'{query}|{context_hash}'.encode()).hexdigest()}"

        # L1
        result = self._l1.get(key)
        if result is not None:
            return result

        # L2
        result = self._get_redis(key)
        if result is not None:
            self._l1.put(key, result)
            return result.get("answer") if isinstance(result, dict) else result

        return None

    def put_llm_response(
        self,
        query: str,
        context_hash: str,
        answer: str,
    ) -> None:
        """Cache an LLM response."""
        if not self._enabled:
            return

        key = f"rag:llm:{hashlib.md5(f'{query}|{context_hash}'.encode()).hexdigest()}"
        response = {"answer": answer, "timestamp": time.time()}

        self._l1.put(key, answer)
        self._put_redis(key, response, self._llm_ttl)

    # ── Management ──────────────────────────────────────────

    def clear(self) -> None:
        """Clear all cache tiers."""
        self._l1.clear()
        if self._redis_available and self._redis:
            try:
                # Delete all rag:* keys
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match="rag:*", count=100)
                    if keys:
                        self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception:
                pass

        if self._qdrant_client:
            try:
                self._qdrant_client.delete_collection(self._cache_collection)
                self._init_semantic_layer()
            except Exception:
                pass

        _log.info("cache_cleared")

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        redis_size = 0
        if self._redis_available and self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match="rag:*", count=100)
                    redis_size += len(keys)
                    if cursor == 0:
                        break
            except Exception:
                pass

        qdrant_size = 0
        if self._qdrant_client:
            try:
                info = self._qdrant_client.get_collection(self._cache_collection)
                qdrant_size = info.points_count or 0
            except Exception:
                pass

        return {
            "l1": self._l1.stats,
            "l2_redis": {"size": redis_size, "available": self._redis_available},
            "l3_semantic": {"size": qdrant_size, "threshold": self._similarity_threshold},
        }


# ── Singleton ───────────────────────────────────────────────

_instance: Optional[SemanticCache] = None
_lock = threading.Lock()


def get_semantic_cache() -> SemanticCache:
    """Get or create the singleton SemanticCache."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = SemanticCache()
        return _instance
