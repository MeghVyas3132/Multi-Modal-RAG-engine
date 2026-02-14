"""
Graph Service Package — knowledge graph construction and querying.
"""

from services.graph_service.entity_extractor import extract_entities, extract_entities_batch
from services.graph_service.knowledge_graph import KnowledgeGraph, get_knowledge_graph

__all__ = [
    "extract_entities",
    "extract_entities_batch",
    "KnowledgeGraph",
    "get_knowledge_graph",
]
