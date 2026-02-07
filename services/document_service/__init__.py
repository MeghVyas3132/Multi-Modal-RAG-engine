"""
Document Service Package — intelligent document parsing and chunking.
"""

from services.document_service.semantic_chunker import SemanticChunker, get_semantic_chunker

__all__ = [
    "SemanticChunker",
    "get_semantic_chunker",
]
