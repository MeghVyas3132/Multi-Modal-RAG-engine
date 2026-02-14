"""
Unit tests for entity extraction input/output contract.

These tests mock the LLM call to validate that:
  1. extract_entities returns (List[str], List[Dict]) tuple.
  2. extract_entities_batch expects List[Dict] with 'text', 'source', 'chunk_id'.
  3. Relationship validation filters out invalid edges.
  4. Entity normalisation is consistent.
"""
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.graph_service.entity_extractor import (
    _normalize_entity,
    extract_entities,
    extract_entities_batch,
)


# ── Helpers ─────────────────────────────────────────────────

def _make_llm_response(entities, relationships):
    """Build a mock OpenAI ChatCompletion response."""
    payload = json.dumps({"entities": entities, "relationships": relationships})
    choice = MagicMock()
    choice.message.content = payload
    response = MagicMock()
    response.choices = [choice]
    return response


# ── _normalize_entity ───────────────────────────────────────

class TestNormalizeEntity:
    def test_lowercase(self):
        assert _normalize_entity("  Photosynthesis  ") == "photosynthesis"

    def test_collapse_spaces(self):
        assert _normalize_entity("  machine  learning  ") == "machine learning"

    def test_empty(self):
        assert _normalize_entity("") == ""


# ── extract_entities ────────────────────────────────────────

class TestExtractEntities:
    @pytest.mark.asyncio
    async def test_returns_tuple(self):
        """Return type must be (entities, relationships) tuple."""
        mock_resp = _make_llm_response(
            entities=["Sun", "Photosynthesis"],
            relationships=[
                {"from": "Sun", "to": "Photosynthesis", "type": "causes"},
            ],
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("services.graph_service.entity_extractor.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            # Ensure cfg has a key
            with patch("services.graph_service.entity_extractor.get_settings") as mock_cfg:
                cfg = MagicMock()
                cfg.cerebras_api_key = "test-key"
                cfg.cerebras_base_url = "https://example.com"
                cfg.llm_model = "test-model"
                mock_cfg.return_value = cfg

                entities, rels = await extract_entities(
                    text="The sun drives photosynthesis.",
                    source="bio.pdf",
                    chunk_id="c1",
                )

        assert isinstance(entities, list)
        assert isinstance(rels, list)
        assert all(isinstance(e, str) for e in entities)
        assert all(isinstance(r, dict) for r in rels)

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        """Without any API key, should return ([], [])."""
        with patch("services.graph_service.entity_extractor.get_settings") as mock_cfg:
            cfg = MagicMock()
            cfg.cerebras_api_key = ""
            cfg.groq_api_key = ""
            mock_cfg.return_value = cfg

            entities, rels = await extract_entities("Some text")

        assert entities == []
        assert rels == []

    @pytest.mark.asyncio
    async def test_relationship_validation_filters_invalid(self):
        """Relationships referencing non-extracted entities should be filtered."""
        mock_resp = _make_llm_response(
            entities=["A", "B"],
            relationships=[
                {"from": "A", "to": "B", "type": "x"},
                {"from": "A", "to": "UNKNOWN", "type": "x"},  # should be filtered
            ],
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("services.graph_service.entity_extractor.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            with patch("services.graph_service.entity_extractor.get_settings") as mock_cfg:
                cfg = MagicMock()
                cfg.cerebras_api_key = "key"
                cfg.cerebras_base_url = "https://example.com"
                cfg.llm_model = "model"
                mock_cfg.return_value = cfg

                entities, rels = await extract_entities("text")

        # Only A→B should survive
        assert len(rels) == 1
        assert rels[0]["from"] == "a"
        assert rels[0]["to"] == "b"


# ── extract_entities_batch ──────────────────────────────────

class TestExtractEntitiesBatch:
    @pytest.mark.asyncio
    async def test_batch_accepts_dict_list(self):
        """extract_entities_batch must accept List[Dict] with text/source/chunk_id."""
        mock_resp = _make_llm_response(entities=[], relationships=[])
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("services.graph_service.entity_extractor.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_client
            with patch("services.graph_service.entity_extractor.get_settings") as mock_cfg:
                cfg = MagicMock()
                cfg.cerebras_api_key = "key"
                cfg.cerebras_base_url = "https://example.com"
                cfg.llm_model = "model"
                mock_cfg.return_value = cfg

                chunks = [
                    {"text": "The sun is hot.", "source": "a.pdf", "chunk_id": "c0"},
                    {"text": "Plants make food.", "source": "a.pdf", "chunk_id": "c1"},
                ]
                results = await extract_entities_batch(chunks)

        assert len(results) == 2
        for entities, rels in results:
            assert isinstance(entities, list)
            assert isinstance(rels, list)

    @pytest.mark.asyncio
    async def test_batch_empty_input(self):
        """Empty input should return empty results."""
        results = await extract_entities_batch([])
        assert results == []
