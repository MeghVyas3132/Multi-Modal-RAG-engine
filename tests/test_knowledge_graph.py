"""
Unit tests for the knowledge graph — in-memory graph operations,
entity lookup, multi-hop traversal, query expansion, and persistence.
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.graph_service.knowledge_graph import KnowledgeGraph


@pytest.fixture()
def graph(tmp_path):
    """Create a fresh KnowledgeGraph backed by a temp file."""
    path = str(tmp_path / "test_graph.json")
    return KnowledgeGraph(persist_path=path)


class TestAddEntities:
    def test_add_single(self, graph):
        graph.add_entities(["photosynthesis"], source="bio.pdf", chunk_id="c1")
        assert graph._graph.has_node("photosynthesis")
        assert graph._graph.nodes["photosynthesis"]["count"] == 1

    def test_add_duplicate_increments_count(self, graph):
        graph.add_entities(["cell"], source="a.pdf")
        graph.add_entities(["cell"], source="b.pdf")
        assert graph._graph.nodes["cell"]["count"] == 2
        assert "a.pdf" in graph._graph.nodes["cell"]["sources"]
        assert "b.pdf" in graph._graph.nodes["cell"]["sources"]

    def test_add_multiple(self, graph):
        graph.add_entities(["mitosis", "meiosis", "cell cycle"])
        assert graph._graph.number_of_nodes() == 3


class TestAddRelationships:
    def test_add_single_edge(self, graph):
        graph.add_relationships([
            {"from": "sun", "to": "photosynthesis", "type": "causes", "source": "bio.pdf"},
        ])
        assert graph._graph.has_edge("sun", "photosynthesis")
        assert graph._graph["sun"]["photosynthesis"]["type"] == "causes"

    def test_duplicate_edge_increments_weight(self, graph):
        rel = {"from": "A", "to": "B", "type": "relates_to"}
        graph.add_relationships([rel])
        graph.add_relationships([rel])
        assert graph._graph["A"]["B"]["weight"] == 2

    def test_skip_empty_entities(self, graph):
        graph.add_relationships([{"from": "", "to": "B", "type": "x"}])
        assert graph._graph.number_of_edges() == 0


class TestFindRelated:
    def test_single_hop(self, graph):
        graph.add_relationships([
            {"from": "A", "to": "B", "type": "relates_to"},
            {"from": "A", "to": "C", "type": "causes"},
        ])
        results = graph.find_related("A", max_hops=1)
        names = [r["entity"] for r in results]
        assert "B" in names
        assert "C" in names

    def test_multi_hop(self, graph):
        graph.add_relationships([
            {"from": "A", "to": "B", "type": "x"},
            {"from": "B", "to": "C", "type": "x"},
            {"from": "C", "to": "D", "type": "x"},
        ])
        results = graph.find_related("A", max_hops=3)
        names = [r["entity"] for r in results]
        assert "D" in names

    def test_max_hops_limit(self, graph):
        graph.add_relationships([
            {"from": "A", "to": "B", "type": "x"},
            {"from": "B", "to": "C", "type": "x"},
            {"from": "C", "to": "D", "type": "x"},
        ])
        results = graph.find_related("A", max_hops=1)
        names = [r["entity"] for r in results]
        assert "B" in names
        assert "D" not in names

    def test_missing_entity(self, graph):
        results = graph.find_related("nonexistent")
        assert results == []

    def test_case_insensitive(self, graph):
        graph.add_relationships([
            {"from": "Earth", "to": "Moon", "type": "orbits"},
        ])
        results = graph.find_related("earth")  # lowercase query
        names = [r["entity"] for r in results]
        assert "Moon" in names

    def test_max_results(self, graph):
        for i in range(30):
            graph.add_relationships([
                {"from": "center", "to": f"node_{i}", "type": "x"},
            ])
        results = graph.find_related("center", max_results=5)
        assert len(results) <= 5


class TestFindEntitiesInQuery:
    def test_substring_match(self, graph):
        graph.add_entities(["photosynthesis", "chlorophyll"])
        found = graph.find_entities_in_query("How does photosynthesis use chlorophyll?")
        assert "photosynthesis" in found
        assert "chlorophyll" in found

    def test_short_entity_skipped(self, graph):
        graph.add_entities(["AI", "machine learning"])
        found = graph.find_entities_in_query("AI and machine learning")
        # "AI" is only 2 chars — below the len >= 3 threshold
        assert "AI" not in found
        assert "machine learning" in found


class TestExpandQuery:
    def test_expansion(self, graph):
        graph.add_entities(["photosynthesis", "chlorophyll", "light"])
        graph.add_relationships([
            {"from": "photosynthesis", "to": "chlorophyll", "type": "uses"},
            {"from": "photosynthesis", "to": "light", "type": "requires"},
        ])
        terms = graph.expand_query("What is photosynthesis?")
        assert "chlorophyll" in terms or "light" in terms

    def test_no_entities_returns_empty(self, graph):
        terms = graph.expand_query("something completely unrelated 12345")
        assert terms == []


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "persist_test.json")

        # Build and save
        g1 = KnowledgeGraph(persist_path=path)
        g1.add_entities(["A", "B"])
        g1.add_relationships([{"from": "A", "to": "B", "type": "x"}])
        g1.save()

        # Load into new instance
        g2 = KnowledgeGraph(persist_path=path)
        assert g2._graph.has_node("A")
        assert g2._graph.has_node("B")
        assert g2._graph.has_edge("A", "B")

    def test_save_only_when_dirty(self, tmp_path):
        path = str(tmp_path / "dirty_test.json")
        g = KnowledgeGraph(persist_path=path)
        g.save()  # Not dirty — should not create file
        assert not os.path.exists(path)

        g.add_entities(["X"])
        g.save()  # Now dirty — should persist
        assert os.path.exists(path)


class TestStats:
    def test_empty_graph(self, graph):
        s = graph.stats()
        assert s["nodes"] == 0
        assert s["edges"] == 0
        assert s["density"] == 0

    def test_populated_graph(self, graph):
        graph.add_relationships([
            {"from": "A", "to": "B", "type": "x"},
            {"from": "B", "to": "C", "type": "x"},
        ])
        s = graph.stats()
        assert s["nodes"] == 3
        assert s["edges"] == 2
