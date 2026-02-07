"""
Entity Extractor — LLM-powered entity and relationship extraction.

Architecture decisions:
  1. Uses Cerebras (fast, cheap) to extract entities and relationships
     from text chunks during indexing.
  2. Structured JSON output format ensures clean graph construction.
  3. Batch processing — send multiple chunks in one LLM call to
     reduce API overhead.
  4. Entity deduplication via fuzzy matching (lowercase + strip).
  5. Extraction is optional — fails silently if LLM is unavailable.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import openai

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)

# ── Extraction Prompt ───────────────────────────────────────

_EXTRACTION_PROMPT = """Extract named entities and their relationships from the text below.

Return ONLY valid JSON in this exact format:
{
  "entities": ["entity1", "entity2", "entity3"],
  "relationships": [
    {"from": "entity1", "to": "entity2", "type": "relates_to"},
    {"from": "entity2", "to": "entity3", "type": "part_of"}
  ]
}

Rules:
- Extract people, organizations, concepts, locations, events, technical terms
- Relationship types: relates_to, part_of, causes, follows, contradicts, supports, mentions, defines
- Keep entities concise (1-4 words)
- Max 15 entities, max 20 relationships per chunk
- If no entities found, return {"entities": [], "relationships": []}

Text:
"""


def _normalize_entity(entity: str) -> str:
    """Normalize entity for deduplication."""
    return entity.strip().lower().replace("  ", " ")


async def extract_entities(
    text: str,
    source: str = "",
    chunk_id: str = "",
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Extract entities and relationships from a single text chunk.

    Args:
        text: Text content to analyze.
        source: Source document name (for metadata).
        chunk_id: Chunk identifier (for metadata).

    Returns:
        (entities, relationships) tuple.
        entities: List of entity name strings.
        relationships: List of {from, to, type} dicts.
    """
    cfg = get_settings()

    # Get LLM client (Cerebras primary)
    client = None
    if cfg.cerebras_api_key:
        client = openai.AsyncOpenAI(
            api_key=cfg.cerebras_api_key,
            base_url=cfg.cerebras_base_url,
            timeout=15.0,
        )
    elif cfg.groq_api_key:
        client = openai.AsyncOpenAI(
            api_key=cfg.groq_api_key,
            base_url=cfg.groq_base_url,
            timeout=15.0,
        )

    if client is None:
        return [], []

    with timed("entity_extraction") as t:
        try:
            response = await client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": "You are an entity extraction system. Return ONLY valid JSON."},
                    {"role": "user", "content": _EXTRACTION_PROMPT + text[:2000]},
                ],
                max_tokens=512,
                temperature=0.0,
            )

            raw = response.choices[0].message.content or ""

            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                return [], []

            data = json.loads(json_match.group())
            entities = [_normalize_entity(e) for e in data.get("entities", []) if e.strip()]
            relationships = data.get("relationships", [])

            # Validate relationships
            valid_rels = []
            entity_set = set(entities)
            for rel in relationships:
                from_e = _normalize_entity(rel.get("from", ""))
                to_e = _normalize_entity(rel.get("to", ""))
                rel_type = rel.get("type", "relates_to")
                if from_e and to_e and from_e in entity_set and to_e in entity_set:
                    valid_rels.append({
                        "from": from_e,
                        "to": to_e,
                        "type": rel_type,
                        "source": source,
                        "chunk_id": chunk_id,
                    })

            metrics.record("entity_extraction_ms", t["ms"])
            return entities, valid_rels

        except Exception as e:
            _log.warning("entity_extraction_failed", error=str(e))
            return [], []


async def extract_entities_batch(
    chunks: List[Dict[str, Any]],
) -> List[Tuple[List[str], List[Dict[str, str]]]]:
    """
    Extract entities from multiple chunks. Processes sequentially
    to avoid rate limiting.

    Args:
        chunks: List of dicts with 'text', 'source', 'chunk_id' keys.

    Returns:
        List of (entities, relationships) tuples, one per chunk.
    """
    results = []
    for i, chunk in enumerate(chunks):
        entities, rels = await extract_entities(
            text=chunk.get("text", ""),
            source=chunk.get("source", ""),
            chunk_id=chunk.get("chunk_id", str(i)),
        )
        results.append((entities, rels))

        if (i + 1) % 10 == 0:
            _log.info("entity_extraction_progress", done=i + 1, total=len(chunks))

    return results
