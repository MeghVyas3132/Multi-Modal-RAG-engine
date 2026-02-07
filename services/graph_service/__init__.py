"""
Graph Service Package — knowledge graph construction and querying.
"""

from services.graph_service.entity_extractor import EntityExtractor, get_entity_extractor
from services.graph_service.knowledge_graph import KnowledgeGraph, get_knowledge_graph

__all__ = [
    "EntityExtractor",
    "get_entity_extractor",
    "KnowledgeGraph",
    "get_knowledge_graph",
]
