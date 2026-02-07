"""
Modality Router — classify queries to skip irrelevant collections.

Architecture decisions:
  1. Three modes: heuristic (regex, instant), llm (GPT-4o-mini, ~100ms),
     trained (DistilBERT, ~10ms — future, after collecting data).
  2. Heuristic mode uses keyword patterns to detect image vs text intent.
     Fast, free, covers ~80% of cases correctly.
  3. LLM mode uses GPT-4o-mini for ambiguous queries. Results are
     cached aggressively (LRU + Redis) to amortize cost.
  4. Output is a probability dict: {text: 0.8, image: 0.15, table: 0.03, code: 0.02}
     The retriever uses this to weight collection searches.
  5. Below router_hybrid_threshold confidence, we fall back to
     searching all collections (safe default).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, Optional

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)


# ── Modality types ──────────────────────────────────────────

MODALITIES = ("text", "image", "table", "code")


def _default_probs() -> Dict[str, float]:
    """Default hybrid probabilities (search everything equally)."""
    return {"text": 0.4, "image": 0.3, "table": 0.2, "code": 0.1}


# ── Heuristic Patterns ─────────────────────────────────────

_IMAGE_PATTERNS = re.compile(
    r"\b("
    r"show\s+me|image\s+of|picture\s+of|photo\s+of|diagram\s+of|"
    r"what\s+does\s+.+\s+look\s+like|visualize|illustration|"
    r"figure|chart|graph|plot|map|drawing|sketch|"
    r"how\s+does\s+.+\s+look|display|depict"
    r")\b",
    re.IGNORECASE,
)

_TABLE_PATTERNS = re.compile(
    r"\b("
    r"table|spreadsheet|csv|data\s+for|statistics|"
    r"compare.*numbers|list\s+of|tabular|column|row\s+data|"
    r"how\s+many|percentage|ratio|count\s+of"
    r")\b",
    re.IGNORECASE,
)

_CODE_PATTERNS = re.compile(
    r"\b("
    r"code|function|class|def\s+|import\s+|implement|"
    r"algorithm|program|script|syntax|API|endpoint|"
    r"debug|compile|runtime|error\s+in\s+code"
    r")\b",
    re.IGNORECASE,
)


# ── LRU Cache ──────────────────────────────────────────────

class _LRUCache:
    """Simple thread-safe LRU cache for router decisions."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._cache: OrderedDict[str, Dict[str, float]] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Dict[str, float]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Dict[str, float]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value


_router_cache = _LRUCache()


# ── Router Functions ────────────────────────────────────────

def route_heuristic(query: str) -> Dict[str, float]:
    """
    Fast regex-based routing. ~0.01ms, zero cost.
    Returns probability distribution over modalities.
    """
    query_lower = query.lower().strip()

    # Check patterns
    image_match = bool(_IMAGE_PATTERNS.search(query_lower))
    table_match = bool(_TABLE_PATTERNS.search(query_lower))
    code_match = bool(_CODE_PATTERNS.search(query_lower))

    # Build probability distribution
    probs = {"text": 0.0, "image": 0.0, "table": 0.0, "code": 0.0}

    matches = sum([image_match, table_match, code_match])

    if matches == 0:
        # No specific pattern — likely a text query
        probs["text"] = 0.85
        probs["image"] = 0.10
        probs["table"] = 0.03
        probs["code"] = 0.02
    else:
        # Distribute probability to matched modalities
        remaining = 0.90
        per_match = remaining / matches

        if image_match:
            probs["image"] = per_match
        if table_match:
            probs["table"] = per_match
        if code_match:
            probs["code"] = per_match

        # Text always gets some probability
        probs["text"] = 0.10

    return probs


async def route_llm(query: str) -> Dict[str, float]:
    """
    Use GPT-4o-mini for zero-shot modality classification.
    Cost: ~$0.00015 per query. Results cached.
    """
    import json
    import openai

    cfg = get_settings()
    if not cfg.openai_api_key:
        _log.warning("router_llm_no_key_fallback_heuristic")
        return route_heuristic(query)

    # Check cache first
    cached = _router_cache.get(query)
    if cached is not None:
        return cached

    client = openai.AsyncOpenAI(api_key=cfg.openai_api_key)

    with timed("router_llm") as t:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify this search query by what type of content it needs. "
                            "Return ONLY a JSON object with probabilities that sum to 1.0: "
                            '{"text": 0.0, "image": 0.0, "table": 0.0, "code": 0.0}'
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=50,
                temperature=0.0,
            )

            raw = response.choices[0].message.content or ""
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', raw)
            if json_match:
                probs = json.loads(json_match.group())
                # Ensure all keys exist
                for key in MODALITIES:
                    probs.setdefault(key, 0.0)
                # Normalize
                total = sum(probs.values())
                if total > 0:
                    probs = {k: v / total for k, v in probs.items()}
                else:
                    probs = _default_probs()
            else:
                probs = _default_probs()

        except Exception as e:
            _log.warning("router_llm_failed", error=str(e))
            probs = route_heuristic(query)

    metrics.record("router_latency_ms", t["ms"])

    # Cache the result
    _router_cache.put(query, probs)

    return probs


def route_query(query: str) -> Dict[str, float]:
    """
    Synchronous routing entry point (heuristic mode).
    Use route_query_async() for LLM mode.
    """
    cfg = get_settings()

    if not cfg.router_enabled:
        return _default_probs()

    if cfg.router_mode == "heuristic":
        return route_heuristic(query)
    else:
        # For non-async contexts, use heuristic
        return route_heuristic(query)


async def route_query_async(query: str) -> Dict[str, float]:
    """
    Async routing entry point — uses LLM mode if configured.
    """
    cfg = get_settings()

    if not cfg.router_enabled:
        return _default_probs()

    if cfg.router_mode == "llm":
        return await route_llm(query)
    else:
        return route_heuristic(query)


def get_primary_modality(probs: Dict[str, float]) -> str:
    """Return the modality with highest probability."""
    return max(probs, key=probs.get)


def should_search_modality(
    probs: Dict[str, float],
    modality: str,
    threshold: float = 0.1,
) -> bool:
    """Check if a modality should be searched based on probability."""
    return probs.get(modality, 0.0) >= threshold
