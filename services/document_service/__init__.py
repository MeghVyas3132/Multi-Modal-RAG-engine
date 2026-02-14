"""
Document Service Package — intelligent document parsing and chunking.
"""

from services.document_service.semantic_chunker import (
    DocumentChunk,
    auto_chunk,
    chunk_fixed,
    chunk_hierarchical,
    chunk_semantic,
)

__all__ = [
    "DocumentChunk",
    "auto_chunk",
    "chunk_fixed",
    "chunk_hierarchical",
    "chunk_semantic",
]
