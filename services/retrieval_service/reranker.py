"""
Cross-Encoder Reranker — precision layer on top of ANN retrieval.

Architecture decisions:
  1. Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, ~2ms per pair).
     At 50 candidates × 2ms = 100ms reranking. Acceptable for <200ms total.
  2. Lazy-loaded: model only loaded on first rerank call.
  3. Batch scoring: all query-document pairs scored in one forward pass.
  4. Only reranks text modality. Image results keep original scores.
  5. Top-N after reranking defaults to 10 (configurable).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import numpy as np

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranker for precision refinement.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._model_name = cfg.reranker_model
        self._top_n = cfg.reranker_top_n
        self._enabled = cfg.reranker_enabled
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self) -> None:
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
                _log.info("reranker_loaded", model=self._model_name)
            except Exception as e:
                _log.error("reranker_load_failed", error=str(e))
                self._enabled = False

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder scoring.

        Args:
            query: The user query.
            results: List of result dicts from retriever.
            top_n: How many to return after reranking.
            text_key: Metadata key containing the text to score against.

        Returns:
            Reranked results with updated scores.
        """
        if not self._enabled or not results:
            return results

        top_n = top_n or self._top_n
        self._load_model()

        if self._model is None:
            return results

        # Separate text and non-text results
        text_results = []
        non_text_results = []

        for r in results:
            content = r.get("metadata", {}).get(text_key, "")
            if content and r.get("modality", "text") == "text":
                text_results.append(r)
            else:
                non_text_results.append(r)

        if not text_results:
            return results[:top_n]

        # Build query-document pairs for cross-encoder
        pairs = [
            [query, r.get("metadata", {}).get(text_key, "")]
            for r in text_results
        ]

        with timed("reranking") as t:
            scores = self._model.predict(pairs)

        metrics.record("rerank_latency_ms", t["ms"])

        # Update scores
        for r, score in zip(text_results, scores):
            r["original_score"] = r["score"]
            r["score"] = float(score)
            r["reranked"] = True

        # Sort text results by reranker score
        text_results.sort(key=lambda x: x["score"], reverse=True)

        # Merge: top text results + non-text results
        # Interleave non-text based on their original position ratio
        merged = []
        text_idx = 0
        non_text_idx = 0

        # Simple merge: take top_n from reranked text, then fill with non-text
        for r in text_results[:top_n]:
            merged.append(r)

        remaining = top_n - len(merged)
        if remaining > 0:
            for r in non_text_results[:remaining]:
                merged.append(r)

        _log.info(
            "reranked",
            query=query[:60],
            candidates=len(results),
            text_scored=len(text_results),
            returned=len(merged),
            latency_ms=round(t["ms"], 2),
        )

        return merged

    def is_loaded(self) -> bool:
        """Check if reranker model is loaded."""
        return self._model is not None


# ── Singleton ───────────────────────────────────────────────

_instance: Optional[Reranker] = None
_lock = threading.Lock()


def get_reranker() -> Reranker:
    """Get or create the singleton Reranker."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = Reranker()
        return _instance
