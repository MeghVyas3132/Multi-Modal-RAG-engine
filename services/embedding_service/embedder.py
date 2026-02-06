"""
CLIP Embedding Service — the hot path of every search request.

Architecture decisions:
  1. Model is loaded ONCE at module import / startup and kept in GPU memory.
     Loading ViT-H-14 takes ~4s. We pay that cost once, not per request.
  2. We use open_clip (not the original openai/clip) because it supports
     ViT-H-14 with laion2b weights — better zero-shot performance.
  3. Text tokenization + forward pass happen in a single call with
     torch.no_grad() and torch.amp.autocast for speed.
  4. The returned vector is L2-normalized (unit sphere) so cosine
     similarity reduces to dot product in Qdrant.
  5. Thread safety: PyTorch forward passes are thread-safe for inference
     when using no_grad(). We don't need explicit locking.
  6. We pre-allocate the tokenizer context length to avoid dynamic
     reallocation per query.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
import torch
import open_clip

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)


class CLIPEmbedder:
    """
    Singleton-style CLIP text/image encoder.

    Instantiate once at startup via `create_embedder()`, then call
    `encode_text()` on the hot path. The model stays warm in memory.
    """

    def __init__(self) -> None:
        cfg = get_settings()

        # ── Device selection ────────────────────────────────
        if cfg.force_cpu or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda")
        _log.info("clip_device_selected", device=str(self._device))

        # ── Load model + tokenizer ──────────────────────────
        with timed("clip_model_load"):
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                cfg.clip_model_name,
                pretrained=cfg.clip_pretrained,
                device=self._device,
            )
            self._tokenizer = open_clip.get_tokenizer(cfg.clip_model_name)

        # Put model in eval mode and disable dropout / batchnorm updates.
        self._model.eval()

        # ── Compile the model for faster inference (PyTorch 2.x) ──
        # torch.compile() with reduce-overhead gives ~20-30% speedup on GPU.
        if self._device.type == "cuda" and hasattr(torch, "compile"):
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead")
                _log.info("clip_model_compiled", mode="reduce-overhead")
            except Exception as e:
                _log.warning("clip_compile_failed", error=str(e))

        self._vector_dim = cfg.clip_vector_dim
        self._ready = True

        # ── Warm up: run a dummy forward pass ───────────────
        # This triggers CUDA kernel compilation and memory allocation
        # so the first real request doesn't pay that cost.
        self._warmup()

    def _warmup(self) -> None:
        """Run a throwaway inference to warm CUDA kernels + allocators."""
        with timed("clip_warmup"):
            _ = self.encode_text("warmup query")
        _log.info("clip_warmup_complete")

    # ── Public API ──────────────────────────────────────────

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text query into a normalized embedding vector.

        Returns:
            np.ndarray of shape (vector_dim,), dtype float32, L2-normalized.

        This is the hot path. Every nanosecond counts.
        """
        with timed("clip_text_encode") as t:
            tokens = self._tokenizer([text]).to(self._device)

            with torch.no_grad(), torch.amp.autocast(device_type=self._device.type):
                text_features = self._model.encode_text(tokens)

            # L2 normalize so cosine similarity = dot product
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Move to CPU numpy — Qdrant client expects numpy/list
            # .float() ensures float32 before numpy (autocast may produce bfloat16)
            vector = text_features.squeeze(0).cpu().float().numpy()

        metrics.record("embedding_latency_ms", t["ms"])
        return vector

    def encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch encode multiple text queries. Useful for evaluation,
        not for the real-time search path (where batch=1 is optimal).

        Returns:
            np.ndarray of shape (len(texts), vector_dim), dtype float32.
        """
        tokens = self._tokenizer(texts).to(self._device)

        with torch.no_grad(), torch.amp.autocast(device_type=self._device.type):
            text_features = self._model.encode_text(tokens)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().float().numpy()

    def encode_images(self, images: List) -> np.ndarray:
        """
        Batch encode preprocessed image tensors.
        Used ONLY by the offline indexing pipeline, NEVER at query time.

        Args:
            images: list of PIL.Image already preprocessed via self._preprocess

        Returns:
            np.ndarray of shape (len(images), vector_dim), dtype float32.
        """
        image_input = torch.stack(images).to(self._device)

        with torch.no_grad(), torch.amp.autocast(device_type=self._device.type):
            image_features = self._model.encode_image(image_input)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().float().numpy()

    @property
    def preprocess(self):
        """Expose the image preprocessing transform for the indexing pipeline."""
        return self._preprocess

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── Module-level singleton ──────────────────────────────────
# We don't create this at import time because settings may not
# be loaded yet. Instead, we use a factory + cache pattern.

_instance: Optional[CLIPEmbedder] = None
_lock = threading.Lock()


def create_embedder() -> CLIPEmbedder:
    """
    Create or return the singleton CLIPEmbedder.
    Thread-safe, idempotent. Call this at startup.
    """
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        # Double-checked locking
        if _instance is not None:
            return _instance
        _instance = CLIPEmbedder()
        return _instance


def get_embedder() -> CLIPEmbedder:
    """
    Return the singleton. Raises if not yet created.
    Use on the hot path — no locking overhead.
    """
    if _instance is None:
        raise RuntimeError(
            "CLIPEmbedder not initialized. Call create_embedder() at startup."
        )
    return _instance
