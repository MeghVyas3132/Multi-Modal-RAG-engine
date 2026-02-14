"""
Unit tests for the modality router — heuristic routing and LRU cache.
"""
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.routing_service.modality_router import (
    route_heuristic,
    get_primary_modality,
    should_search_modality,
    _LRUCache,
)


class TestRouteHeuristic:
    """Test the regex-based heuristic router."""

    def test_text_query(self):
        probs = route_heuristic("What is photosynthesis?")
        assert probs["text"] > 0.5
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_image_query(self):
        probs = route_heuristic("show me a diagram of a cell")
        assert probs["image"] > 0.3
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_table_query(self):
        probs = route_heuristic("show me a table of population statistics")
        # Should match both image and table patterns
        assert probs["table"] > 0.1
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_code_query(self):
        probs = route_heuristic("show me the code for quicksort algorithm")
        assert probs["code"] > 0.1
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_empty_query(self):
        probs = route_heuristic("")
        assert probs["text"] > 0.5

    def test_probabilities_sum_to_one(self):
        queries = [
            "normal text query",
            "show me image of cat",
            "table of data for countries",
            "implement a function in python",
            "show me a chart of graph data for code algorithm",
        ]
        for q in queries:
            probs = route_heuristic(q)
            assert abs(sum(probs.values()) - 1.0) < 0.01, f"Failed for: {q}"


class TestGetPrimaryModality:
    def test_text_primary(self):
        assert get_primary_modality({"text": 0.8, "image": 0.1, "table": 0.05, "code": 0.05}) == "text"

    def test_image_primary(self):
        assert get_primary_modality({"text": 0.1, "image": 0.7, "table": 0.1, "code": 0.1}) == "image"


class TestShouldSearchModality:
    def test_above_threshold(self):
        assert should_search_modality({"text": 0.8, "image": 0.2}, "text", 0.1)

    def test_below_threshold(self):
        assert not should_search_modality({"text": 0.8, "image": 0.05}, "image", 0.1)


class TestLRUCacheThreadSafety:
    """Test the thread-safe LRU cache."""

    def test_basic_operations(self):
        cache = _LRUCache(maxsize=3)
        cache.put("a", {"text": 1.0})
        cache.put("b", {"text": 0.5})
        assert cache.get("a") == {"text": 1.0}
        assert cache.get("c") is None

    def test_eviction(self):
        cache = _LRUCache(maxsize=2)
        cache.put("a", {"text": 1.0})
        cache.put("b", {"text": 0.5})
        cache.put("c", {"text": 0.3})  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == {"text": 0.5}
        assert cache.get("c") == {"text": 0.3}

    def test_concurrent_access(self):
        """Hammer the cache from multiple threads to detect race conditions."""
        cache = _LRUCache(maxsize=100)
        errors = []

        def writer(prefix: str, count: int):
            try:
                for i in range(count):
                    cache.put(f"{prefix}:{i}", {"text": float(i)})
            except Exception as e:
                errors.append(e)

        def reader(prefix: str, count: int):
            try:
                for i in range(count):
                    cache.get(f"{prefix}:{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for t_id in range(8):
            threads.append(threading.Thread(target=writer, args=(f"w{t_id}", 200)))
            threads.append(threading.Thread(target=reader, args=(f"w{t_id}", 200)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
