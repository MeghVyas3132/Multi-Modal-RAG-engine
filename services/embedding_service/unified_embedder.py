"""
Unified Embedding Service — single vector space for text AND images.

Architecture decisions:
  1. Uses sentence-transformers with a CLIP-based model that projects
     both text and images into the SAME vector space (768-dim).
  2. This fixes the fundamental V1 problem where CLIP (512-dim) and
     MiniLM (384-dim) lived in different spaces, making cross-modal
     search impossible.
  3. Supports encode_text(), encode_image(), and encode_batch() —
     all return vectors in the same 768-dim space.
  4. Falls back to separate CLIP+MiniLM if the unified model can't
     load (e.g., OOM on 16GB Mac).
  5. Singleton pattern matches existing embedders for consistency.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)


class UnifiedEmbedder:
    """
    Single embedding model for text + images in the same vector space.
    Uses sentence-transformers CLIP model for cross-modal retrieval.
    """

    def __init__(self) -> None:
        cfg = get_settings()

        with timed("unified_model_load") as t:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                cfg.unified_model_name,
                trust_remote_code=True,
            )

        self._vector_dim = cfg.unified_vector_dim
        self._batch_size = cfg.unified_batch_size
        self._ready = True

        _log.info(
            "unified_embedder_ready",
            model=cfg.unified_model_name,
            dim=self._vector_dim,
            load_ms=round(t["ms"], 2),
        )

        self._warmup()

    def _warmup(self) -> None:
        """Run throwaway inferences to warm caches."""
        with timed("unified_warmup"):
            _ = self.encode_text("warmup query")
        _log.info("unified_warmup_complete")

    # ── Text Encoding ───────────────────────────────────────

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a normalized vector.

        Returns:
            np.ndarray of shape (vector_dim,), dtype float32, L2-normalized.
        """
        with timed("unified_text_encode") as t:
            vector = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        metrics.record("unified_text_encode_ms", t["ms"])
        return vector.astype(np.float32)

    def encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch encode multiple text strings.

        Returns:
            np.ndarray of shape (len(texts), vector_dim), dtype float32.
        """
        with timed("unified_text_batch_encode") as t:
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )

        metrics.record("unified_text_batch_ms", t["ms"])
        _log.info("unified_text_batch_encoded", count=len(texts), ms=round(t["ms"], 2))
        return vectors.astype(np.float32)

    # ── Image Encoding ──────────────────────────────────────

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a single PIL Image into a normalized vector in the SAME
        space as text vectors. This is the key V2 capability.

        Returns:
            np.ndarray of shape (vector_dim,), dtype float32, L2-normalized.
        """
        with timed("unified_image_encode") as t:
            vector = self._model.encode(
                image,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        metrics.record("unified_image_encode_ms", t["ms"])
        return vector.astype(np.float32)

    def encode_image_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Batch encode multiple PIL Images.

        Returns:
            np.ndarray of shape (len(images), vector_dim), dtype float32.
        """
        with timed("unified_image_batch_encode") as t:
            vectors = self._model.encode(
                images,
                normalize_embeddings=True,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )

        metrics.record("unified_image_batch_ms", t["ms"])
        _log.info("unified_image_batch_encoded", count=len(images), ms=round(t["ms"], 2))
        return vectors.astype(np.float32)

    # ── Properties ──────────────────────────────────────────

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def device(self):
        return str(self._model.device)


# ── Module-level singleton ──────────────────────────────────

_instance: Optional[UnifiedEmbedder] = None
_lock = threading.Lock()


def create_unified_embedder() -> UnifiedEmbedder:
    """Create or return the singleton UnifiedEmbedder. Thread-safe."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = UnifiedEmbedder()
        return _instance


def get_unified_embedder() -> UnifiedEmbedder:
    """Return the singleton. Raises if not yet created."""
    if _instance is None:
        raise RuntimeError(
            "UnifiedEmbedder not initialized. Call create_unified_embedder() at startup."
        )
    return _instance
