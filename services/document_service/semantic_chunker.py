"""
Semantic Chunker — context-aware document splitting.

Architecture decisions:
  1. Three strategies: fixed (V1 legacy), semantic (embedding-based
     topic boundary detection), and hierarchical (respects document
     structure from MinerU/PyMuPDF).
  2. Semantic chunking embeds consecutive sentences and splits where
     cosine similarity drops below threshold (topic shift).
  3. Hierarchical chunking uses section headers as natural boundaries,
     with recursive subdivision for long sections.
  4. All strategies produce DocumentChunk objects with rich metadata
     including modality tag, section hierarchy, and language.
  5. Variable chunk sizes (128-1024 chars) based on content density.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed

_log = get_logger(__name__)


@dataclass
class DocumentChunk:
    """
    A typed chunk from any document source.
    V2 replacement for the V1 TextChunk — carries modality and hierarchy.
    """
    content: str
    modality: str  # text, table, equation, figure_caption, code
    page_num: int
    chunk_index: int
    source: str  # filename or URL
    section_hierarchy: List[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    language: str = "en"
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences at . ! ? boundaries."""
    # Match sentence endings followed by space or end of string
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip() and len(s.strip()) > 5]


def _detect_language(text: str) -> str:
    """Detect language of text. Returns ISO 639-1 code."""
    cfg = get_settings()
    if not cfg.multilingual_enabled:
        return cfg.default_language

    try:
        from langdetect import detect
        return detect(text[:500])  # Sample first 500 chars for speed
    except Exception:
        return cfg.default_language


def chunk_fixed(
    text: str,
    page_num: int,
    source: str,
    chunk_size: int = 512,
    overlap: int = 64,
    modality: str = "text",
) -> List[DocumentChunk]:
    """
    V1-compatible fixed-window chunking with sentence boundary snapping.
    Kept for backward compatibility and simple documents.
    """
    if not text or len(text) < 20:
        return []

    lang = _detect_language(text)
    chunks: List[DocumentChunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Snap to sentence boundary
        if end < len(text):
            for boundary in range(end, max(start + chunk_size // 2, start), -1):
                if text[boundary - 1] in '.!?\n':
                    end = boundary
                    break

        chunk_text = text[start:end].strip()
        if len(chunk_text) >= 20:
            chunks.append(DocumentChunk(
                content=chunk_text,
                modality=modality,
                page_num=page_num,
                chunk_index=idx,
                source=source,
                char_start=start,
                char_end=end,
                language=lang,
            ))
            idx += 1

        if end < len(text):
            new_start = end - overlap
            # Guarantee forward progress — always advance at least 1 char
            start = max(new_start, start + 1)
        else:
            start = len(text)

    return chunks


def chunk_semantic(
    text: str,
    page_num: int,
    source: str,
    modality: str = "text",
    embedder=None,
) -> List[DocumentChunk]:
    """
    Semantic chunking — split at topic boundaries detected by
    embedding similarity drops between consecutive sentences.

    Args:
        text: Full text to chunk.
        page_num: Page number (1-indexed).
        source: Source document name.
        modality: Content type tag.
        embedder: Embedding model with encode_text_batch() or encode_batch().

    Returns:
        List of DocumentChunks with variable sizes.
    """
    cfg = get_settings()
    min_size = cfg.semantic_chunk_min_size
    max_size = cfg.semantic_chunk_max_size
    threshold = cfg.semantic_chunk_threshold

    if not text or len(text) < min_size:
        return chunk_fixed(text, page_num, source, modality=modality)

    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return chunk_fixed(text, page_num, source, modality=modality)

    lang = _detect_language(text)

    # Embed all sentences
    if embedder is None:
        # Fall back to fixed chunking if no embedder available
        return chunk_fixed(text, page_num, source, modality=modality)

    try:
        # Use unified embedder's batch encode
        if hasattr(embedder, 'encode_text_batch'):
            vectors = embedder.encode_text_batch(sentences)
        elif hasattr(embedder, 'encode_batch'):
            vectors = embedder.encode_batch(sentences)
        else:
            return chunk_fixed(text, page_num, source, modality=modality)
    except Exception as e:
        _log.warning("semantic_chunk_embed_failed", error=str(e))
        return chunk_fixed(text, page_num, source, modality=modality)

    # Compute cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(vectors) - 1):
        cos_sim = np.dot(vectors[i], vectors[i + 1])
        similarities.append(cos_sim)

    # Find split points where similarity drops below threshold
    split_indices = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            split_indices.append(i + 1)
    split_indices.append(len(sentences))

    # Build chunks from sentence groups
    chunks: List[DocumentChunk] = []
    char_pos = 0
    idx = 0

    for i in range(len(split_indices) - 1):
        start_sent = split_indices[i]
        end_sent = split_indices[i + 1]
        chunk_sentences = sentences[start_sent:end_sent]
        chunk_text = " ".join(chunk_sentences).strip()

        # Enforce min/max size constraints
        if len(chunk_text) < min_size and chunks:
            # Merge with previous chunk
            prev = chunks[-1]
            chunks[-1] = DocumentChunk(
                content=prev.content + " " + chunk_text,
                modality=modality,
                page_num=page_num,
                chunk_index=prev.chunk_index,
                source=source,
                char_start=prev.char_start,
                char_end=char_pos + len(chunk_text),
                language=lang,
            )
        elif len(chunk_text) > max_size:
            # Split oversized chunk with fixed strategy
            sub_chunks = chunk_fixed(
                chunk_text, page_num, source,
                chunk_size=max_size, overlap=64, modality=modality,
            )
            for sc in sub_chunks:
                sc.chunk_index = idx
                sc.char_start = char_pos + sc.char_start
                sc.char_end = char_pos + sc.char_end
                sc.language = lang
                chunks.append(sc)
                idx += 1
        elif len(chunk_text) >= 20:
            chunks.append(DocumentChunk(
                content=chunk_text,
                modality=modality,
                page_num=page_num,
                chunk_index=idx,
                source=source,
                char_start=char_pos,
                char_end=char_pos + len(chunk_text),
                language=lang,
            ))
            idx += 1

        char_pos += len(chunk_text) + 1

    return chunks


def chunk_hierarchical(
    sections: List[dict],
    source: str,
    embedder=None,
) -> List[DocumentChunk]:
    """
    Hierarchical chunking — respects document structure.
    Expects sections as list of dicts with 'title', 'content', 'page_num', 'level'.

    Falls back to semantic chunking for content within each section.
    """
    cfg = get_settings()
    max_size = cfg.semantic_chunk_max_size
    chunks: List[DocumentChunk] = []
    global_idx = 0

    for section in sections:
        title = section.get("title", "")
        content = section.get("content", "")
        page_num = section.get("page_num", 1)
        level = section.get("level", 0)
        hierarchy = section.get("hierarchy", [])

        if not content or len(content.strip()) < 20:
            continue

        # Prepend section title for context
        full_text = f"{title}\n{content}" if title else content

        if len(full_text) <= max_size:
            # Section fits in one chunk
            chunks.append(DocumentChunk(
                content=full_text.strip(),
                modality="text",
                page_num=page_num,
                chunk_index=global_idx,
                source=source,
                section_hierarchy=hierarchy + ([title] if title else []),
                language=_detect_language(content),
            ))
            global_idx += 1
        else:
            # Section too large — use semantic chunking
            sub_chunks = chunk_semantic(
                full_text, page_num, source,
                modality="text", embedder=embedder,
            )
            for sc in sub_chunks:
                sc.chunk_index = global_idx
                sc.section_hierarchy = hierarchy + ([title] if title else [])
                chunks.append(sc)
                global_idx += 1

    return chunks


def auto_chunk(
    text: str,
    page_num: int,
    source: str,
    modality: str = "text",
    embedder=None,
) -> List[DocumentChunk]:
    """
    Auto-select chunking strategy based on settings.
    """
    cfg = get_settings()

    if cfg.chunking_strategy == "semantic" and embedder is not None:
        return chunk_semantic(text, page_num, source, modality=modality, embedder=embedder)
    elif cfg.chunking_strategy == "hierarchical":
        # Hierarchical requires structured input; fall back to semantic
        return chunk_semantic(text, page_num, source, modality=modality, embedder=embedder)
    else:
        return chunk_fixed(text, page_num, source, modality=modality)
