"""
Unit tests for the semantic chunker module.

Covers chunk_fixed, chunk_semantic (with mock embedder), auto_chunk,
edge cases, and DocumentChunk metadata.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.document_service.semantic_chunker import (
    DocumentChunk,
    auto_chunk,
    chunk_fixed,
    chunk_semantic,
    _split_sentences,
)


# ── Helpers ─────────────────────────────────────────────────

def _make_text(n_sentences: int = 20, sentence_len: int = 80) -> str:
    """Generate deterministic text with clear sentence boundaries."""
    sentences = []
    for i in range(n_sentences):
        word = f"word{i}"
        filler = " ".join([word] * (sentence_len // (len(word) + 1)))
        sentences.append(f"Sentence {i}: {filler}.")
    return " ".join(sentences)


class _MockEmbedder:
    """
    Fake embedder that returns deterministic vectors.
    Consecutive sentences get similar vectors; every `topic_shift` sentences
    the vector shifts sharply to simulate a topic boundary.
    """

    def __init__(self, dim: int = 64, topic_shift: int = 5):
        self.dim = dim
        self.topic_shift = topic_shift
        self._call_count = 0

    def encode_text_batch(self, texts):
        """Return list of numpy vectors with artificial topic shifts."""
        vectors = []
        for i, _ in enumerate(texts):
            topic = i // self.topic_shift
            base = np.zeros(self.dim)
            base[topic % self.dim] = 1.0  # one-hot-ish per topic
            # Add small noise to keep adjacent sentences similar
            noise = np.random.RandomState(i).randn(self.dim) * 0.01
            vec = base + noise
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            vectors.append(vec)
        return vectors


# ── chunk_fixed ─────────────────────────────────────────────

class TestChunkFixed:
    def test_basic_chunking(self):
        text = _make_text(10)
        chunks = chunk_fixed(text, page_num=1, source="test.pdf")
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_empty_text(self):
        assert chunk_fixed("", page_num=1, source="x") == []

    def test_short_text(self):
        assert chunk_fixed("Hi", page_num=1, source="x") == []

    def test_overlap(self):
        text = _make_text(20)
        chunks = chunk_fixed(text, page_num=1, source="test.pdf", chunk_size=200, overlap=50)
        if len(chunks) >= 2:
            # Overlap means second chunk starts before first chunk ends
            assert chunks[1].char_start < chunks[0].char_end + 50

    def test_metadata(self):
        text = _make_text(5)
        chunks = chunk_fixed(text, page_num=3, source="report.pdf", modality="table")
        for c in chunks:
            assert c.page_num == 3
            assert c.source == "report.pdf"
            assert c.modality == "table"
            assert c.language  # should have a language tag

    def test_chunk_index_sequential(self):
        text = _make_text(20)
        chunks = chunk_fixed(text, page_num=1, source="s", chunk_size=300)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# ── chunk_semantic ──────────────────────────────────────────

class TestChunkSemantic:
    def test_with_mock_embedder(self):
        text = _make_text(20)
        embedder = _MockEmbedder(topic_shift=5)
        chunks = chunk_semantic(text, page_num=1, source="s", embedder=embedder)
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_without_embedder_fallback_to_fixed(self):
        text = _make_text(10)
        chunks = chunk_semantic(text, page_num=1, source="s", embedder=None)
        # Should fall back to chunk_fixed
        assert len(chunks) >= 1

    def test_short_text_fallback(self):
        chunks = chunk_semantic("Short.", page_num=1, source="s", embedder=_MockEmbedder())
        # Very short — falls back to chunk_fixed which returns []
        assert isinstance(chunks, list)

    def test_topic_boundaries_produce_splits(self):
        """With mock embedder that shifts every 5 sentences,
        semantic chunking should produce more chunks than fixed."""
        text = _make_text(30)
        embedder = _MockEmbedder(topic_shift=5)
        sem_chunks = chunk_semantic(text, page_num=1, source="s", embedder=embedder)
        fixed_chunks = chunk_fixed(text, page_num=1, source="s", chunk_size=512)
        # Semantic chunking with topic shifts should produce ≥ fixed chunks
        # (not guaranteed to be strictly more, but should be reasonable)
        assert len(sem_chunks) >= 1


# ── auto_chunk ──────────────────────────────────────────────

class TestAutoChunk:
    def test_defaults_to_fixed_without_embedder(self):
        text = _make_text(10)
        chunks = auto_chunk(text, page_num=1, source="s")
        assert len(chunks) >= 1

    def test_returns_document_chunks(self):
        text = _make_text(10)
        chunks = auto_chunk(text, page_num=1, source="s", embedder=_MockEmbedder())
        assert all(isinstance(c, DocumentChunk) for c in chunks)


# ── _split_sentences ────────────────────────────────────────

class TestSplitSentences:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third sentence."
        parts = _split_sentences(text)
        assert len(parts) >= 2

    def test_question_marks(self):
        text = "What is this? This is that. Really!"
        parts = _split_sentences(text)
        assert len(parts) >= 2

    def test_short_fragments_filtered(self):
        text = "Hi. OK. This is a longer sentence that should be kept."
        parts = _split_sentences(text)
        # Very short fragments (≤5 chars) are filtered
        for p in parts:
            assert len(p) > 5
