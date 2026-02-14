"""
Knowledge Graph — schema-less graph for multi-hop reasoning.

Architecture decisions:
  1. NetworkX DiGraph — in-memory, fast, no external DB needed.
     At 100K nodes + 500K edges: ~200MB RAM. Acceptable for our scale.
  2. Schema-less: entities are strings, types are discovered automatically.
     No predefined ontology — works across all domains.
  3. Incremental updates: add nodes/edges from new documents without
     rebuilding the entire graph.
  4. Persistence: serialize to JSON on disk, load at startup.
  5. Graph queries: find_related(entity, max_hops) returns connected
     entities for query expansion in the retriever.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed

_log = get_logger(__name__)


class KnowledgeGraph:
    """
    In-memory knowledge graph using NetworkX.
    Thread-safe for concurrent reads, serialized writes.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        cfg = get_settings()
        self._persist_path = Path(persist_path or cfg.graph_persist_path)
        self._graph = nx.DiGraph()
        self._lock = threading.Lock()
        self._max_hops = cfg.graph_max_hops
        self._dirty = False

        # Load persisted graph
        self._load()

    def _load(self) -> None:
        """Load graph from JSON file if exists."""
        if self._persist_path.exists():
            try:
                with timed("graph_load") as t:
                    data = json.loads(self._persist_path.read_text())
                    for node in data.get("nodes", []):
                        self._graph.add_node(
                            node["id"],
                            **{k: v for k, v in node.items() if k != "id"},
                        )
                    for edge in data.get("edges", []):
                        self._graph.add_edge(
                            edge["from"],
                            edge["to"],
                            **{k: v for k, v in edge.items() if k not in ("from", "to")},
                        )
                _log.info(
                    "graph_loaded",
                    nodes=self._graph.number_of_nodes(),
                    edges=self._graph.number_of_edges(),
                    ms=round(t["ms"], 2),
                )
            except Exception as e:
                _log.warning("graph_load_failed", error=str(e))
                self._graph = nx.DiGraph()

    def save(self) -> None:
        """Persist graph to JSON file."""
        if not self._dirty:
            return

        with self._lock:
            try:
                self._persist_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "nodes": [
                        {"id": n, **self._graph.nodes[n]}
                        for n in self._graph.nodes
                    ],
                    "edges": [
                        {"from": u, "to": v, **d}
                        for u, v, d in self._graph.edges(data=True)
                    ],
                    "metadata": {
                        "saved_at": time.time(),
                        "node_count": self._graph.number_of_nodes(),
                        "edge_count": self._graph.number_of_edges(),
                    },
                }
                self._persist_path.write_text(json.dumps(data, indent=2))
                self._dirty = False
                _log.info(
                    "graph_saved",
                    nodes=self._graph.number_of_nodes(),
                    edges=self._graph.number_of_edges(),
                )
            except Exception as e:
                _log.warning("graph_save_failed", error=str(e))

    # ── Graph Mutation ──────────────────────────────────────

    def add_entities(
        self,
        entities: List[str],
        source: str = "",
        chunk_id: str = "",
    ) -> None:
        """Add entity nodes to the graph."""
        with self._lock:
            for entity in entities:
                if self._graph.has_node(entity):
                    # Increment mention count
                    self._graph.nodes[entity]["count"] = (
                        self._graph.nodes[entity].get("count", 0) + 1
                    )
                    # Track all sources
                    sources = self._graph.nodes[entity].get("sources", [])
                    if source and source not in sources:
                        sources.append(source)
                        self._graph.nodes[entity]["sources"] = sources
                else:
                    self._graph.add_node(
                        entity,
                        count=1,
                        sources=[source] if source else [],
                        first_seen=time.time(),
                    )
            self._dirty = True

    def add_relationships(
        self,
        relationships: List[Dict[str, str]],
    ) -> None:
        """Add edges to the graph."""
        with self._lock:
            for rel in relationships:
                from_e = rel.get("from", "")
                to_e = rel.get("to", "")
                rel_type = rel.get("type", "relates_to")
                source = rel.get("source", "")

                if not from_e or not to_e:
                    continue

                # Ensure nodes exist
                if not self._graph.has_node(from_e):
                    self._graph.add_node(from_e, count=1, sources=[source])
                if not self._graph.has_node(to_e):
                    self._graph.add_node(to_e, count=1, sources=[source])

                if self._graph.has_edge(from_e, to_e):
                    # Increment weight
                    self._graph[from_e][to_e]["weight"] = (
                        self._graph[from_e][to_e].get("weight", 1) + 1
                    )
                else:
                    self._graph.add_edge(
                        from_e,
                        to_e,
                        type=rel_type,
                        weight=1,
                        source=source,
                    )
            self._dirty = True

    # ── Graph Queries ───────────────────────────────────────

    def find_related(
        self,
        entity: str,
        max_hops: Optional[int] = None,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity within max_hops.
        Returns list of {entity, relationship, distance, weight}.
        Used for query expansion in the retriever.
        """
        max_hops = max_hops or self._max_hops
        entity_lower = entity.lower().strip()

        # Find matching node(s) — case-insensitive
        matching_nodes = [
            n for n in self._graph.nodes
            if n.lower() == entity_lower
        ]

        if not matching_nodes:
            return []

        results = []
        seen: Set[str] = set()

        for start_node in matching_nodes:
            # BFS traversal up to max_hops — returns {node: distance}
            path_lengths = nx.single_source_shortest_path_length(
                self._graph, start_node, cutoff=max_hops
            )
            for target, distance in path_lengths.items():
                if target == start_node or target in seen:
                    continue
                seen.add(target)

                # Get relationship type from direct edge (if exists)
                edge_data = self._graph.get_edge_data(start_node, target) or {}
                rel_type = edge_data.get("type", "connected")
                weight = edge_data.get("weight", 1)

                results.append({
                    "entity": target,
                    "relationship": rel_type,
                    "distance": distance,
                    "weight": weight,
                    "sources": self._graph.nodes[target].get("sources", []),
                })

        # Sort by weight (descending) then distance (ascending)
        results.sort(key=lambda r: (-r["weight"], r["distance"]))
        return results[:max_results]

    def find_entities_in_query(self, query: str) -> List[str]:
        """
        Find graph entities mentioned in a query string.
        Simple substring matching — fast but not fuzzy.
        """
        query_lower = query.lower()
        found = []

        for node in self._graph.nodes:
            if len(node) >= 3 and node.lower() in query_lower:
                found.append(node)

        # Sort by entity length (longer = more specific = better)
        found.sort(key=len, reverse=True)
        return found[:10]

    def expand_query(self, query: str) -> List[str]:
        """
        Given a query, find mentioned entities, traverse graph,
        and return expansion terms for the retriever.
        """
        entities = self.find_entities_in_query(query)
        if not entities:
            return []

        expansion_terms = []
        for entity in entities[:3]:  # Limit to top 3 entities
            related = self.find_related(entity, max_hops=1, max_results=5)
            for r in related:
                if r["entity"] not in expansion_terms:
                    expansion_terms.append(r["entity"])

        return expansion_terms[:10]

    # ── Stats ───────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "density": round(nx.density(self._graph), 6) if self._graph.number_of_nodes() > 0 else 0,
            "components": nx.number_weakly_connected_components(self._graph)
            if self._graph.number_of_nodes() > 0 else 0,
        }


# ── Module-level singleton ──────────────────────────────────

_instance: Optional[KnowledgeGraph] = None
_lock = threading.Lock()


def get_knowledge_graph() -> KnowledgeGraph:
    """Get or create the singleton KnowledgeGraph."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = KnowledgeGraph()
        return _instance
