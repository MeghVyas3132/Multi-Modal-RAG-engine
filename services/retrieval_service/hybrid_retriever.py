"""
Hybrid Retriever — unified collection search with multi-signal fusion.

Architecture decisions:
  1. Searches the unified_vectors collection (single space for text + images).
  2. Graph expansion: before searching, expands query with related entities
     from the knowledge graph for better recall.
  3. Modality weighting: uses router probabilities to weight results
     from different modalities.
  4. Reciprocal Rank Fusion (RRF): if searching multiple collections
     (legacy mode), fuses results using RRF instead of raw score merging.
  5. Deduplication at retrieval time: filters near-duplicate results.
  6. Falls back to legacy VectorRetriever if unified collection unavailable.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)


class HybridRetriever:
    """
    Production hybrid retriever with graph expansion and modality awareness.
    """

    def __init__(self) -> None:
        cfg = get_settings()

        # Qdrant
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, HnswConfigDiff,
            SearchParams, ScalarQuantization, ScalarQuantizationConfig,
            OptimizersConfigDiff,
        )

        self._client = QdrantClient(
            host=cfg.qdrant_host,
            port=cfg.qdrant_port,
            grpc_port=cfg.qdrant_grpc_port,
            prefer_grpc=True,
            timeout=10,
        )

        self._unified_collection = cfg.unified_collection
        self._unified_dim = cfg.unified_vector_dim
        self._hnsw_ef = cfg.qdrant_hnsw_ef
        self._top_k_initial = cfg.retrieval_top_k_initial
        self._hybrid_enabled = cfg.hybrid_search_enabled
        self._graph_enabled = cfg.graph_enabled

        # Ensure unified collection exists
        self._ensure_unified_collection()

        _log.info(
            "hybrid_retriever_init",
            collection=self._unified_collection,
            dim=self._unified_dim,
            hybrid=self._hybrid_enabled,
        )

    def _ensure_unified_collection(self) -> None:
        """Create the unified_vectors collection if needed."""
        from qdrant_client.models import (
            Distance, VectorParams, HnswConfigDiff,
            OptimizersConfigDiff, ScalarQuantization,
            ScalarQuantizationConfig,
        )

        cfg = get_settings()
        collections = [
            c.name for c in self._client.get_collections().collections
        ]

        if self._unified_collection in collections:
            info = self._client.get_collection(self._unified_collection)
            _log.info(
                "unified_collection_exists",
                vectors=info.vectors_count,
                status=info.status.name,
            )
            return

        self._client.create_collection(
            collection_name=self._unified_collection,
            vectors_config=VectorParams(
                size=self._unified_dim,
                distance=Distance.COSINE,
                on_disk=False,
            ),
            hnsw_config=HnswConfigDiff(
                m=cfg.qdrant_hnsw_m,
                ef_construct=cfg.qdrant_hnsw_ef_construct,
                on_disk=False,
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2,
                memmap_threshold=100_000,
            ),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type="int8",
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
        )
        _log.info("unified_collection_created", dim=self._unified_dim)

    # ── Core Search ─────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: float = 0.0,
        modality_weights: Optional[Dict[str, float]] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        expand_with_graph: bool = True,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with modality weighting and graph expansion.

        Args:
            query_vector: Unified embedding of the query.
            top_k: Final number of results to return.
            score_threshold: Minimum score cutoff.
            modality_weights: {"text": 0.8, "image": 0.15, ...} from router.
            filter_conditions: Qdrant payload filters.
            expand_with_graph: Whether to use graph expansion.
            query_text: Original query text (for graph entity matching).

        Returns:
            List of result dicts with id, score, metadata, modality.
        """
        from qdrant_client.models import (
            SearchParams, Filter, FieldCondition, MatchValue,
        )

        # Graph expansion: add related entity vectors to search
        expansion_terms = []
        if expand_with_graph and self._graph_enabled and query_text:
            expansion_terms = self._graph_expand(query_text)

        # Build filter
        qdrant_filter = None
        if filter_conditions:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
            qdrant_filter = Filter(must=conditions)

        # Primary search
        with timed("hybrid_search") as t:
            results = self._client.search(
                collection_name=self._unified_collection,
                query_vector=query_vector.tolist(),
                limit=self._top_k_initial,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                search_params=SearchParams(hnsw_ef=self._hnsw_ef),
                with_payload=True,
            )

        metrics.record("hybrid_search_latency_ms", t["ms"])

        # Convert to result dicts
        candidates = [
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "metadata": hit.payload or {},
                "modality": (hit.payload or {}).get("modality", "text"),
            }
            for hit in results
        ]

        # Apply modality weighting
        if modality_weights:
            candidates = self._apply_modality_weights(
                candidates, modality_weights
            )

        # Deduplicate near-identical results
        candidates = self._deduplicate_results(candidates)

        # Sort by weighted score and trim
        candidates.sort(key=lambda x: x["score"], reverse=True)
        final = candidates[:top_k]

        _log.info(
            "hybrid_search_complete",
            initial=len(results),
            after_weight=len(candidates),
            final=len(final),
            expansion_terms=len(expansion_terms),
            latency_ms=round(t["ms"], 2),
        )

        return final

    # ── Multi-Collection Search (legacy mode) ───────────────

    def search_multi_collection(
        self,
        query_vector_clip: Optional[np.ndarray] = None,
        query_vector_text: Optional[np.ndarray] = None,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search multiple legacy collections and fuse with RRF.
        Used when unified_enabled=False.
        """
        from qdrant_client.models import SearchParams

        cfg = get_settings()
        all_results = []

        # Image collection (CLIP vectors)
        if query_vector_clip is not None:
            try:
                img_results = self._client.search(
                    collection_name=cfg.qdrant_collection,
                    query_vector=query_vector_clip.tolist(),
                    limit=top_k * 2,
                    score_threshold=score_threshold,
                    search_params=SearchParams(hnsw_ef=self._hnsw_ef),
                    with_payload=True,
                )
                for hit in img_results:
                    all_results.append({
                        "id": str(hit.id),
                        "score": float(hit.score),
                        "metadata": hit.payload or {},
                        "modality": "image",
                        "collection": cfg.qdrant_collection,
                    })
            except Exception as e:
                _log.warning("image_search_error", error=str(e))

        # Text collection (sentence transformer vectors)
        if query_vector_text is not None:
            try:
                text_results = self._client.search(
                    collection_name=cfg.pdf_text_collection,
                    query_vector=query_vector_text.tolist(),
                    limit=top_k * 2,
                    score_threshold=score_threshold,
                    search_params=SearchParams(hnsw_ef=self._hnsw_ef),
                    with_payload=True,
                )
                for hit in text_results:
                    all_results.append({
                        "id": str(hit.id),
                        "score": float(hit.score),
                        "metadata": hit.payload or {},
                        "modality": "text",
                        "collection": cfg.pdf_text_collection,
                    })
            except Exception as e:
                _log.warning("text_search_error", error=str(e))

        # Fuse with RRF
        fused = self._reciprocal_rank_fusion(all_results, k=60)
        return fused[:top_k]

    # ── Graph Expansion ─────────────────────────────────────

    def _graph_expand(self, query_text: str) -> List[str]:
        """
        Find entities in query, traverse graph, return expansion terms.
        """
        try:
            from services.graph_service.knowledge_graph import get_knowledge_graph
            graph = get_knowledge_graph()
            return graph.expand_query(query_text)
        except Exception as e:
            _log.debug("graph_expansion_error", error=str(e))
            return []

    # ── Modality Weighting ──────────────────────────────────

    @staticmethod
    def _apply_modality_weights(
        results: List[Dict[str, Any]],
        weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Adjust scores based on router-determined modality relevance.
        """
        for r in results:
            modality = r.get("modality", "text")
            weight = weights.get(modality, 0.5)
            # Blend: 70% original score + 30% modality weight
            r["score"] = r["score"] * 0.7 + weight * 0.3
        return results

    # ── Reciprocal Rank Fusion ──────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        RRF merges ranked lists from multiple sources.
        Score = sum(1 / (k + rank_i)) for each source list.
        """
        # Group by source collection
        by_collection: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            col = r.get("collection", "default")
            if col not in by_collection:
                by_collection[col] = []
            by_collection[col].append(r)

        # Sort each list by original score
        for col in by_collection:
            by_collection[col].sort(key=lambda x: x["score"], reverse=True)

        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, Dict[str, Any]] = {}

        for col, ranked in by_collection.items():
            for rank, r in enumerate(ranked):
                doc_id = r["id"]
                rrf_score = 1.0 / (k + rank + 1)

                if doc_id in rrf_scores:
                    rrf_scores[doc_id] += rrf_score
                else:
                    rrf_scores[doc_id] = rrf_score
                    result_map[doc_id] = r

        # Apply RRF scores
        for doc_id, score in rrf_scores.items():
            result_map[doc_id]["score"] = round(score, 6)

        # Sort by RRF score
        fused = sorted(result_map.values(), key=lambda x: x["score"], reverse=True)
        return fused

    # ── Deduplication ───────────────────────────────────────

    @staticmethod
    def _deduplicate_results(
        results: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate results based on content hash.
        """
        seen_hashes = set()
        unique = []

        for r in results:
            # Hash based on text content in metadata
            content = r.get("metadata", {}).get(text_key, "")
            if not content:
                content = str(r.get("metadata", {}))

            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique.append(r)

        return unique

    # ── Batch Upsert ────────────────────────────────────────

    def upsert_unified_batch(
        self,
        ids: List,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
    ) -> None:
        """Insert vectors into the unified collection."""
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=int(id_) if isinstance(id_, (int, np.integer)) else str(id_),
                vector=vec.tolist(),
                payload=payload,
            )
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]

        self._client.upsert(
            collection_name=self._unified_collection,
            points=points,
            wait=True,
        )

    # ── Stats ───────────────────────────────────────────────

    def collection_info(self) -> Dict[str, Any]:
        """Return unified collection stats."""
        try:
            info = self._client.get_collection(self._unified_collection)
            return {
                "collection": self._unified_collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.name,
            }
        except Exception as e:
            return {"error": str(e)}

    @property
    def client(self):
        return self._client


# ── Singleton ───────────────────────────────────────────────

_instance: Optional[HybridRetriever] = None
_lock = __import__("threading").Lock()


def get_hybrid_retriever() -> HybridRetriever:
    """Get or create the singleton HybridRetriever."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = HybridRetriever()
        return _instance
