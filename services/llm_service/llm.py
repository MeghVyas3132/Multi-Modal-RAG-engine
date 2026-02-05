"""
Optional LLM Explanation Service

Architecture decisions:
  1. This service is NEVER on the critical path. Search results are returned
     immediately. LLM explanation is generated asynchronously and can be
     fetched separately or streamed.
  2. We use OpenAI's async client to avoid blocking the event loop.
  3. If LLM is disabled (default), this module is a no-op.
  4. The prompt is kept minimal to reduce token cost and latency.
  5. We cache LLM responses keyed by (query, result_ids) to avoid
     re-generating explanations for identical searches.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)

# Lazy import — don't load openai if LLM is disabled
_openai_client = None


def _get_openai_client():
    """Lazy-load the async OpenAI client."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    import openai
    cfg = get_settings()
    _openai_client = openai.AsyncOpenAI(api_key=cfg.openai_api_key)
    return _openai_client


_SYSTEM_PROMPT = """You are an image search assistant. Given a user's search query 
and a list of retrieved image results with metadata, provide a brief 2-3 sentence 
explanation of why these images match the query. Be concise and factual."""


async def generate_explanation(
    query: str,
    results: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Generate an LLM explanation for search results.
    Returns None if LLM is disabled or on error.

    This is ASYNC and should be called with asyncio.create_task()
    so it never blocks the search response.
    """
    cfg = get_settings()
    if not cfg.llm_enabled:
        return None

    try:
        # Build a compact representation of results for the prompt
        result_summary = [
            {
                "rank": i + 1,
                "score": r.get("score", 0),
                "file_name": r.get("metadata", {}).get("file_name", "unknown"),
            }
            for i, r in enumerate(results[:5])  # Cap at 5 to limit token usage
        ]

        user_prompt = (
            f"Search query: \"{query}\"\n\n"
            f"Top results:\n{json.dumps(result_summary, indent=2)}\n\n"
            f"Why do these images match the query?"
        )

        client = _get_openai_client()

        with timed("llm_generation") as t:
            response = await client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.3,  # Low temperature for factual responses
            )

        metrics.record("llm_latency_ms", t["ms"])

        explanation = response.choices[0].message.content.strip()
        _log.info("llm_explanation_generated", query=query, tokens=response.usage.total_tokens)
        return explanation

    except Exception as e:
        _log.error("llm_generation_failed", error=str(e), query=query)
        return None
