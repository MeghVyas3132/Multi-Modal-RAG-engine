"""
LLM Service — Cerebras (primary) + Groq (fallback) with streaming.

Architecture decisions:
  1. Both Cerebras and Groq expose OpenAI-compatible APIs, so we use
     the openai SDK with different base_url values. Zero custom code.
  2. Cerebras Llama 3.3 70B: ~2100 tok/s, TTFT ~70ms — used as primary.
     Groq Llama 3.3 70B: ~300 tok/s, TTFT ~100ms — used as fallback.
  3. Streaming via async generators so the API layer can pipe directly
     to SSE. Each yield is a text delta, not a full response.
  4. Auto-failover: if Cerebras fails (timeout, 5xx, rate limit),
     we immediately retry on Groq. The caller never knows.
  5. The system prompt is RAG-aware: it instructs the model to cite
     retrieved context and stay grounded.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)

# ── Lazy-loaded clients ─────────────────────────────────────

_cerebras_client: Optional[openai.AsyncOpenAI] = None
_groq_client: Optional[openai.AsyncOpenAI] = None


def _get_cerebras_client() -> Optional[openai.AsyncOpenAI]:
    """Lazy-init Cerebras async client."""
    global _cerebras_client
    if _cerebras_client is not None:
        return _cerebras_client

    cfg = get_settings()
    if not cfg.cerebras_api_key:
        _log.warning("cerebras_api_key_not_set")
        return None

    _cerebras_client = openai.AsyncOpenAI(
        api_key=cfg.cerebras_api_key,
        base_url=cfg.cerebras_base_url,
        timeout=30.0,
    )
    _log.info("cerebras_client_ready", base_url=cfg.cerebras_base_url)
    return _cerebras_client


def _get_groq_client() -> Optional[openai.AsyncOpenAI]:
    """Lazy-init Groq async client."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    cfg = get_settings()
    if not cfg.groq_api_key:
        _log.warning("groq_api_key_not_set")
        return None

    _groq_client = openai.AsyncOpenAI(
        api_key=cfg.groq_api_key,
        base_url=cfg.groq_base_url,
        timeout=30.0,
    )
    _log.info("groq_client_ready", base_url=cfg.groq_base_url)
    return _groq_client


# ── System prompts ──────────────────────────────────────────

_RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on retrieved context from documents and images. 

Rules:
1. Base your answer ONLY on the provided context. If the context doesn't contain relevant information, say so.
2. Cite page numbers when referencing specific information (e.g. "According to page 3...").
3. Be concise but thorough. Use bullet points for lists.
4. Never fabricate information not present in the context."""

_IMAGE_SYSTEM_PROMPT = """You are an image search assistant. Given a user's search query 
and a list of retrieved image results with metadata, provide a brief 2-3 sentence 
explanation of why these images match the query. Be concise and factual."""


def _build_rag_prompt(
    query: str,
    text_chunks: List[Dict[str, Any]],
    image_results: List[Dict[str, Any]],
) -> str:
    """Build a RAG prompt with retrieved context."""
    parts = [f'Question: "{query}"\n']

    if text_chunks:
        parts.append("Retrieved text context:")
        for i, chunk in enumerate(text_chunks[:8], 1):
            meta = chunk.get("metadata", {})
            page = meta.get("page_num", "?")
            source = meta.get("source_pdf", "unknown")
            text = meta.get("text", chunk.get("text", ""))
            score = chunk.get("score", 0)
            parts.append(f"  [{i}] (page {page}, {source}, score={score:.2f}): {text}")
        parts.append("")

    if image_results:
        parts.append("Retrieved images:")
        for i, img in enumerate(image_results[:5], 1):
            meta = img.get("metadata", {})
            name = meta.get("file_name", "unknown")
            score = img.get("score", 0)
            parts.append(f"  [{i}] {name} (score={score:.2f})")
        parts.append("")

    parts.append("Answer the question based on the context above.")
    return "\n".join(parts)


# ── Streaming generation ────────────────────────────────────

async def stream_chat(
    query: str,
    text_chunks: Optional[List[Dict[str, Any]]] = None,
    image_results: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response tokens. Tries Cerebras first, falls back to Groq.

    Yields:
        Individual text deltas as they arrive from the LLM.
    """
    cfg = get_settings()
    if not cfg.llm_enabled:
        yield "LLM is disabled. Set LLM_ENABLED=true and provide API keys."
        return

    # Build prompt with RAG context
    has_context = bool(text_chunks or image_results)
    if has_context:
        system_prompt = _RAG_SYSTEM_PROMPT
        user_prompt = _build_rag_prompt(
            query,
            text_chunks or [],
            image_results or [],
        )
    else:
        system_prompt = _IMAGE_SYSTEM_PROMPT
        user_prompt = query

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Try Cerebras first, then Groq
    providers = []
    cerebras = _get_cerebras_client()
    groq = _get_groq_client()

    if cerebras:
        providers.append(("cerebras", cerebras))
    if groq:
        providers.append(("groq", groq))

    if not providers:
        yield "No LLM provider configured. Set CEREBRAS_API_KEY or GROQ_API_KEY."
        return

    last_error = None
    for provider_name, client in providers:
        try:
            start_time = time.perf_counter_ns()
            first_token_time = None
            token_count = 0

            stream = await client.chat.completions.create(
                model=cfg.llm_model,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter_ns()
                        ttft_ms = (first_token_time - start_time) / 1_000_000
                        metrics.record("llm_ttft_ms", ttft_ms)

                    token_count += 1
                    yield delta.content

            total_ms = (time.perf_counter_ns() - start_time) / 1_000_000
            metrics.record("llm_total_ms", total_ms)
            metrics.record("llm_tokens", token_count)

            _log.info(
                "llm_stream_complete",
                provider=provider_name,
                tokens=token_count,
                ttft_ms=round(ttft_ms, 1) if first_token_time else None,
                total_ms=round(total_ms, 1),
            )
            return  # Success — don't try fallback

        except Exception as e:
            last_error = e
            _log.warning(
                "llm_provider_failed",
                provider=provider_name,
                error=str(e),
            )
            continue  # Try next provider

    # All providers failed
    yield f"All LLM providers failed. Last error: {last_error}"


# ── Non-streaming generation (for backward compat) ──────────

async def generate_explanation(
    query: str,
    results: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Generate a non-streaming LLM explanation for search results.
    Backward-compatible with the old /search endpoint.
    """
    cfg = get_settings()
    if not cfg.llm_enabled:
        return None

    # Collect all tokens from the stream
    tokens = []
    async for token in stream_chat(
        query=query,
        image_results=results,
    ):
        tokens.append(token)

    return "".join(tokens) if tokens else None
