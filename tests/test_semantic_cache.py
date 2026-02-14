"""
Unit tests for the semantic cache — L1 LRU, key normalisation,
and cache-key determinism.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.cache_service.semantic_cache import SemanticCache, _LRUCache


# ── _LRUCache (L1 in-process) ──────────────────────────────

class TestLRUCache:
    def test_put_get(self):
        c = _LRUCache(maxsize=5)
        c.put("k1", {"answer": "hello"})
        assert c.get("k1") == {"answer": "hello"}

    def test_miss(self):
        c = _LRUCache(maxsize=5)
        assert c.get("missing") is None

    def test_eviction(self):
        c = _LRUCache(maxsize=3)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        c.put("d", 4)  # evicts "a"
        assert c.get("a") is None
        assert c.get("d") == 4

    def test_access_refreshes_order(self):
        c = _LRUCache(maxsize=3)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        c.get("a")  # touch "a" — now b is the oldest
        c.put("d", 4)  # evicts "b"
        assert c.get("b") is None
        assert c.get("a") == 1

    def test_update_existing_key(self):
        c = _LRUCache(maxsize=5)
        c.put("k", "v1")
        c.put("k", "v2")
        assert c.get("k") == "v2"

    def test_clear(self):
        c = _LRUCache(maxsize=5)
        c.put("a", 1)
        c.clear()
        assert c.get("a") is None
        assert c.stats["size"] == 0
        assert c.stats["hits"] == 0

    def test_stats(self):
        c = _LRUCache(maxsize=5)
        c.put("a", 1)
        c.get("a")  # hit
        c.get("b")  # miss
        s = c.stats
        assert s["size"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5


# ── Static helpers on SemanticCache ─────────────────────────

class TestNormalizeQuery:
    def test_lowercase(self):
        assert SemanticCache._normalize_query("Hello World") == "hello world"

    def test_strip_whitespace(self):
        assert SemanticCache._normalize_query("  foo  bar  ") == "foo bar"

    def test_collapse_spaces(self):
        assert SemanticCache._normalize_query("a   b     c") == "a b c"

    def test_empty(self):
        assert SemanticCache._normalize_query("") == ""

    def test_idempotent(self):
        q = "some query text"
        assert SemanticCache._normalize_query(q) == SemanticCache._normalize_query(q)


class TestCacheKey:
    def test_deterministic(self):
        k1 = SemanticCache._cache_key("hello world")
        k2 = SemanticCache._cache_key("hello world")
        assert k1 == k2

    def test_case_insensitive(self):
        k1 = SemanticCache._cache_key("Hello World")
        k2 = SemanticCache._cache_key("hello world")
        assert k1 == k2

    def test_different_queries_different_keys(self):
        k1 = SemanticCache._cache_key("query one")
        k2 = SemanticCache._cache_key("query two")
        assert k1 != k2

    def test_prefix(self):
        k = SemanticCache._cache_key("q", prefix="rag:search")
        assert k.startswith("rag:search:")

    def test_whitespace_normalization(self):
        k1 = SemanticCache._cache_key("  hello   world  ")
        k2 = SemanticCache._cache_key("hello world")
        assert k1 == k2
