"""
Text Embedding Service — sentence-transformers for PDF RAG.

Architecture decisions:
  1. Uses all-MiniLM-L6-v2 (384-dim) — fast, small, excellent for
     semantic similarity on text chunks. Loads in ~1s on CPU.
  2. Separate from CLIP embedder because text-to-text retrieval needs
     a model trained on text pairs, not CLIP's text-image contrastive.
  3. Singleton pattern matches CLIPEmbedder for consistency.
  4. Batch encoding is the primary path since PDF upload processes
     many chunks at once.
  5. Vectors are L2-normalized for cosine similarity in Qdrant.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)


class TextEmbedder:
    """
    Sentence-transformer text embedder for PDF RAG.
    Loads all-MiniLM-L6-v2 at startup, encodes text chunks to 384-dim vectors.
    """

    def __init__(self) -> None:
        cfg = get_settings()

        with timed("text_model_load") as t:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(cfg.text_model_name)

        self._vector_dim = cfg.text_vector_dim
        self._ready = True

        _log.info(
            "text_embedder_ready",
            model=cfg.text_model_name,
            dim=self._vector_dim,
            load_ms=round(t["ms"], 2),
        )

        # Warm up
        self._warmup()

    def _warmup(self) -> None:
        """Run a throwaway inference to warm caches."""
        with timed("text_warmup"):
            _ = self.encode_text("warmup query")
        _log.info("text_warmup_complete")

    # ── Public API ──────────────────────────────────────────

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text string to a normalized 384-dim vector.

        Returns:
            np.ndarray of shape (384,), dtype float32, L2-normalized.
        """
        with timed("text_encode") as t:
            vector = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        metrics.record("text_embedding_latency_ms", t["ms"])
        return vector.astype(np.float32)

    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Batch encode multiple text strings.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.

        Returns:
            np.ndarray of shape (len(texts), 384), dtype float32, L2-normalized.
        """
        with timed("text_batch_encode") as t:
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )

        metrics.record("text_batch_encode_ms", t["ms"])
        _log.info(
            "text_batch_encoded",
            count=len(texts),
            ms=round(t["ms"], 2),
        )
        return vectors.astype(np.float32)

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── Module-level singleton ──────────────────────────────────

_instance: Optional[TextEmbedder] = None
_lock = threading.Lock()


def create_text_embedder() -> TextEmbedder:
    """Create or return the singleton TextEmbedder. Thread-safe."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = TextEmbedder()
        return _instance


def get_text_embedder() -> TextEmbedder:
    """Return the singleton. Raises if not yet created."""
    if _instance is None:
        raise RuntimeError(
            "TextEmbedder not initialized. Call create_text_embedder() at startup."
        )
    return _instance
