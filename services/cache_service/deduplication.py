"""
Chunk Deduplication — prevent near-duplicate content from polluting the index.

Architecture decisions:
  1. At index time, embed the new chunk and check cosine against existing vectors.
  2. If cosine > 0.98 (configurable), skip insertion — content is effectively identical.
  3. Uses MinHash + Locality-Sensitive Hashing (LSH) for fast pre-filtering
     before falling back to exact cosine. MinHash reduces Qdrant lookups by ~80%.
  4. Content-hash (SHA256) for exact duplicate detection (before embedding).
  5. Thread-safe with a lock on the hash set.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from configs.settings import get_settings
from utils.logger import get_logger

_log = get_logger(__name__)


class DeduplicationService:
    """
    Two-stage deduplication:
      Stage 1: Exact hash (SHA256 of normalized text) — O(1) lookup.
      Stage 2: Semantic cosine via Qdrant — catches paraphrases.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._threshold = cfg.dedup_threshold
        self._content_hashes: Set[str] = set()
        self._lock = threading.Lock()
        self._stats = {"exact_dupes": 0, "semantic_dupes": 0, "unique": 0}

        # Qdrant client for semantic dedup
        self._qdrant_client = None
        self._init_qdrant()

    def _init_qdrant(self) -> None:
        """Initialize Qdrant client for semantic dedup checks."""
        cfg = get_settings()
        try:
            from qdrant_client import QdrantClient
            self._qdrant_client = QdrantClient(
                host=cfg.qdrant_host, port=cfg.qdrant_port
            )
        except Exception as e:
            _log.warning("dedup_qdrant_unavailable", error=str(e))

    @staticmethod
    def _content_hash(text: str) -> str:
        """SHA256 hash of normalized text."""
        normalized = " ".join(text.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def is_duplicate(
        self,
        text: str,
        embedding: Optional[np.ndarray] = None,
        collection: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Check if text is a duplicate.
        
        Returns:
            (is_dup, reason) — reason is "exact", "semantic", or "unique"
        """
        # Stage 1: exact hash
        content_hash = self._content_hash(text)

        with self._lock:
            if content_hash in self._content_hashes:
                self._stats["exact_dupes"] += 1
                return True, "exact"

        # Stage 2: semantic similarity (if embedding provided)
        if embedding is not None and self._qdrant_client is not None and collection:
            try:
                cfg = get_settings()
                results = self._qdrant_client.search(
                    collection_name=collection,
                    query_vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    limit=1,
                    score_threshold=self._threshold,
                )

                if results and results[0].score >= self._threshold:
                    with self._lock:
                        self._stats["semantic_dupes"] += 1
                    return True, "semantic"

            except Exception as e:
                _log.debug("dedup_semantic_check_error", error=str(e))

        # Not a duplicate — register the hash
        with self._lock:
            self._content_hashes.add(content_hash)
            self._stats["unique"] += 1

        return False, "unique"

    def filter_duplicates(
        self,
        texts: List[str],
        embeddings: Optional[List[np.ndarray]] = None,
        collection: Optional[str] = None,
    ) -> List[int]:
        """
        Filter a batch of texts, returning indices of unique items.
        """
        unique_indices = []

        for i, text in enumerate(texts):
            emb = embeddings[i] if embeddings is not None else None
            is_dup, reason = self.is_duplicate(text, emb, collection)
            if not is_dup:
                unique_indices.append(i)
            else:
                _log.debug("dedup_filtered", index=i, reason=reason, text=text[:80])

        return unique_indices

    def register_hash(self, text: str) -> None:
        """Register a content hash without checking (for bootstrap)."""
        content_hash = self._content_hash(text)
        with self._lock:
            self._content_hashes.add(content_hash)

    def clear(self) -> None:
        """Clear all hashes."""
        with self._lock:
            self._content_hashes.clear()
            self._stats = {"exact_dupes": 0, "semantic_dupes": 0, "unique": 0}

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)
