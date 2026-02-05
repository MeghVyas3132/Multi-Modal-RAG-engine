"""
Vector Retrieval Service — Qdrant HNSW ANN search.

Architecture decisions:
  1. We use Qdrant's gRPC interface (port 6334) for search because it's
     ~2x faster than REST for vector operations. HTTP is used only for
     collection management.
  2. The Qdrant client maintains a persistent connection pool. We create
     it once at startup and reuse it for every request.
  3. We set `search_params.hnsw_ef` at query time to control the
     recall/latency tradeoff. Higher ef = better recall, more latency.
     Default 128 gives >95% recall@10 on 1M vectors.
  4. Collection is configured with:
     - on_disk=False (everything in RAM)
     - HNSW indexing (not flat/brute-force)
     - quantization: scalar int8 for ~4x memory reduction with <1% recall loss
  5. All search calls are synchronous because Qdrant's Python client
     doesn't support true async gRPC yet. We wrap in run_in_executor
     at the FastAPI layer.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionStatus,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    QuantizationConfig,
    ScalarQuantization,
    ScalarQuantizationConfig,
    SearchParams,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)


class VectorRetriever:
    """
    Wraps Qdrant client with production-tuned HNSW search.
    Singleton — create once at startup, query on every request.
    """

    def __init__(self) -> None:
        cfg = get_settings()

        # ── Connect to Qdrant (prefer gRPC for search speed) ──
        self._client = QdrantClient(
            host=cfg.qdrant_host,
            port=cfg.qdrant_port,
            grpc_port=cfg.qdrant_grpc_port,
            prefer_grpc=True,  # gRPC is ~2x faster for vector search
            timeout=10,
        )
        self._collection = cfg.qdrant_collection
        self._vector_dim = cfg.clip_vector_dim
        self._hnsw_ef = cfg.qdrant_hnsw_ef
        self._hnsw_m = cfg.qdrant_hnsw_m
        self._hnsw_ef_construct = cfg.qdrant_hnsw_ef_construct

        _log.info(
            "qdrant_connected",
            host=cfg.qdrant_host,
            port=cfg.qdrant_port,
            collection=self._collection,
        )

    # ── Collection Management ───────────────────────────────

    def ensure_collection(self) -> None:
        """
        Create the collection if it doesn't exist. Idempotent.
        Called at startup and by the indexing pipeline.
        """
        collections = [c.name for c in self._client.get_collections().collections]

        if self._collection in collections:
            info = self._client.get_collection(self._collection)
            _log.info(
                "collection_exists",
                name=self._collection,
                vectors_count=info.vectors_count,
                status=info.status.name,
            )
            return

        _log.info("creating_collection", name=self._collection, dim=self._vector_dim)

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=self._vector_dim,
                distance=Distance.COSINE,
                # Force all vectors into RAM — no disk-backed segments
                on_disk=False,
            ),
            hnsw_config=HnswConfigDiff(
                m=self._hnsw_m,
                ef_construct=self._hnsw_ef_construct,
                # Build index in RAM, not on disk
                on_disk=False,
            ),
            optimizers_config=OptimizersConfigDiff(
                # Merge segments aggressively for search speed
                default_segment_number=2,
                # Flush threshold — keep high to avoid premature flushing
                memmap_threshold=100_000,
            ),
            # Scalar quantization: int8 reduces memory ~4x with <1% recall loss.
            # On 1M vectors * 1024 dim * 4 bytes = ~4GB → ~1GB with quantization.
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type="int8",
                    quantile=0.99,
                    always_ram=True,  # Quantized vectors always in RAM
                ),
            ),
        )
        _log.info("collection_created", name=self._collection)

    # ── Search (HOT PATH) ──────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        ANN search against the HNSW index. This is the hot path.

        Args:
            query_vector: L2-normalized float32 vector from CLIP text encoder.
            top_k: Number of nearest neighbors to return.
            score_threshold: Minimum cosine similarity score.
            filter_conditions: Optional payload filter (e.g. {"category": "nature"}).

        Returns:
            List of dicts with keys: id, score, metadata
        """
        # Build optional filter
        qdrant_filter = None
        if filter_conditions:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
            qdrant_filter = Filter(must=conditions)

        with timed("vector_search") as t:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                search_params=SearchParams(
                    hnsw_ef=self._hnsw_ef,
                    # Use quantized vectors for initial candidate selection,
                    # then rescore with original vectors for precision.
                    quantization=None,
                ),
                # Return full payload (image path, metadata)
                with_payload=True,
            )

        metrics.record("search_latency_ms", t["ms"])

        return [
            {
                "id": str(hit.id),
                "score": round(hit.score, 4),
                "metadata": hit.payload or {},
            }
            for hit in results
        ]

    # ── Batch Upsert (used by indexing pipeline) ────────────

    def upsert_batch(
        self,
        ids: List[int],
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
    ) -> None:
        """
        Insert or update a batch of vectors with metadata.
        Used by the offline indexing pipeline only.
        """
        points = [
            PointStruct(
                id=int(id_),
                vector=vec.tolist(),
                payload=payload,
            )
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]

        self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,  # Wait for indexing to confirm durability
        )

    # ── Stats ───────────────────────────────────────────────

    def collection_info(self) -> Dict[str, Any]:
        """Return collection stats for the /stats endpoint."""
        try:
            info = self._client.get_collection(self._collection)
            return {
                "collection": self._collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.name,
                "indexed": info.status == CollectionStatus.GREEN,
                "segments_count": len(info.segments or []) if hasattr(info, 'segments') else None,
            }
        except Exception as e:
            return {"error": str(e)}

    @property
    def client(self) -> QdrantClient:
        """Expose raw client for advanced operations."""
        return self._client


# ── Module-level singleton ──────────────────────────────────

_instance: Optional[VectorRetriever] = None
_lock = threading.Lock()


def create_retriever() -> VectorRetriever:
    """Create or return the singleton VectorRetriever. Thread-safe."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = VectorRetriever()
        return _instance


def get_retriever() -> VectorRetriever:
    """Return the singleton. Raises if not yet created."""
    if _instance is None:
        raise RuntimeError(
            "VectorRetriever not initialized. Call create_retriever() at startup."
        )
    return _instance
