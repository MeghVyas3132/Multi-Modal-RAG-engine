# Architecture

This document describes the system architecture of the Multi-Modal RAG Engine V2, including the data flow, component interactions, and key design decisions.

---

## System Overview

The engine is a monolithic FastAPI application that runs as a single process with a single worker. All service interactions are in-process function calls, not HTTP or gRPC. This eliminates serialization overhead and simplifies deployment. Horizontal scaling is achieved by running multiple container replicas behind a load balancer.

### Core Subsystems

1. **API Gateway** -- FastAPI application with endpoint routers, middleware, and lifespan hooks
2. **Embedding Service** -- Encodes text and images into vector representations
3. **Retrieval Service** -- Searches Qdrant collections and reranks results
4. **Cache Service** -- Multi-tier caching and chunk deduplication
5. **Document Service** -- PDF parsing and semantic text chunking
6. **Graph Service** -- Entity extraction and knowledge graph management
7. **VLM Service** -- Image captioning with a local Vision-Language Model
8. **Routing Service** -- Query modality classification
9. **Web Service** -- Web content scraping and ingestion
10. **LLM Service** -- Chat response generation via external LLM providers

---

## Data Flow

### Search Request

```
Client POST /search {query, top_k}
    |
    v
[Semantic Cache L1] -- in-process OrderedDict LRU
    | miss
    v
[Semantic Cache L2] -- Redis hash lookup
    | miss
    v
[Semantic Cache L3] -- Qdrant ANN cosine > 0.93
    | miss
    v
[Modality Router] -- heuristic keyword/pattern analysis
    |                  returns {text: 0.8, image: 0.2}
    v
[Unified Embedder] -- Jina-CLIP v2, 768-dim vector
    |
    v
[Knowledge Graph Expansion] -- BFS traversal, max 2 hops
    |                           adds related entity terms
    v
[Qdrant HNSW Search] -- unified_vectors collection
    |                     cosine similarity, top_k * 2
    v
[Reciprocal Rank Fusion] -- merge results from graph-expanded queries
    |
    v
[Modality Weighting] -- boost text or image results per router output
    |
    v
[Cross-Encoder Reranker] -- ms-marco-MiniLM-L-6-v2
    |                        re-scores top 50 to top 10
    v
[Cache Store] -- fire-and-forget write to L1 + L2 + L3
    |
    v
Response {results, total, latency_ms, cached}
```

### Chat Request

```
Client POST /chat {query, top_k}
    |
    v
[LLM Response Cache] -- check if this query was answered before (7-day TTL)
    | miss
    v
[Retrieval Pipeline] -- same as search above
    |
    v
[Context Assembly] -- top text chunks + image captions + source attribution
    |
    v
[LLM Streaming] -- Cerebras primary (2100 tok/s), Groq fallback (300 tok/s)
    |                SSE: event:retrieval -> event:token -> event:done
    v
[LLM Response Cache Store]
    |
    v
SSE Stream to Client
```

### PDF Upload

```
Client POST /upload/pdf {file}
    |
    v
[PDF Parser] -- PyMuPDF: extract text chunks + embedded images
    |
    v
[Semantic Chunker] -- topic-boundary splitting via embedding similarity
    |                  min 128 chars, max 1024 chars per chunk
    v
[Deduplication] -- SHA256 hash + cosine > 0.98 against existing vectors
    |
    v
[Unified Embedder] -- encode all unique chunks
    |
    v
[Qdrant Upsert] -- unified_vectors collection
    |
    v
[VLM Captioning] -- SmolVLM-500M on extracted images (background)
    |
    v
[Entity Extraction] -- LLM extracts entities/relationships (background)
    |
    v
[Knowledge Graph Update] -- add nodes and edges
```

### Web URL Indexing

```
Client POST /web/index {url}
    |
    v
[Source Detection] -- YouTube, GitHub, Twitter, or generic web
    |
    v
[Scraper Chain] -- Jina Reader (free) -> Firecrawl (JS) -> httpx (fallback)
    |
    v
[Semantic Chunker] -- same as PDF pipeline
    |
    v
[Unified Embedder + Qdrant Upsert]
    |
    v
[Entity Extraction] -- background async task
```

---

## Design Decisions

### Single Worker Architecture

The application runs with `workers=1` because embedding models (CLIP, Jina-CLIP, SmolVLM, cross-encoder) are loaded into memory. Multiple workers would duplicate these models, consuming 2-6 GB of RAM each. Instead, we scale horizontally with container replicas behind a load balancer.

### In-Process Service Calls

All services (embedding, retrieval, caching, graph) are called as direct Python function calls within the same process. This avoids:
- HTTP/gRPC serialization overhead (2-5ms per call)
- Network latency between microservices
- Complexity of service discovery and circuit breakers

The trade-off is that all services must be co-located in the same process, which is acceptable given the single-worker model.

### Unified Vector Space

V2 uses Jina-CLIP v2 to encode both text and images into the same 768-dimensional vector space. This enables cross-modal search (text query finds images and vice versa) using a single Qdrant collection (`unified_vectors`). Legacy collections (`image_vectors`, `pdf_text_vectors`) are preserved for backward compatibility.

### Multi-Tier Caching

The three-tier cache avoids redundant computation:

| Tier | Backend | Latency | Match Type |
|------|---------|---------|------------|
| L1 | In-process OrderedDict | ~0.01ms | Exact string match |
| L2 | Redis | ~0.5ms | Exact string match |
| L3 | Qdrant ANN | ~5ms | Semantic similarity (cosine > 0.93) |

L3 is the most powerful tier: it can return cached results for semantically similar but textually different queries (e.g., "chocolate cake" matches "cocoa dessert").

### Knowledge Graph for Multi-Hop Reasoning

The NetworkX DiGraph stores entities and relationships extracted from indexed documents. During search, the graph expands the query with related entities via BFS traversal (max 2 hops). This enables multi-hop reasoning: a query about "mitochondria" can find documents about "cellular respiration" and "ATP synthesis" through graph edges.

### Fail-Open Cache Design

Redis and the semantic cache are fail-open: if Redis is down or a cache operation fails, the request falls through to the live pipeline. Search never fails due to a cache error.

### Cross-Encoder Reranking

The initial ANN retrieval returns the top 50 candidates using fast but approximate cosine similarity. The cross-encoder (ms-marco-MiniLM-L-6-v2, 22M parameters) then re-scores each candidate with a full attention pass over the (query, document) pair, returning the top 10. This two-stage approach balances speed and precision.

---

## Component Lifecycle

### Startup Sequence

The lifespan hook in `app.py` loads components in dependency order:

1. CLIP embedding model (legacy, always loaded)
2. Text embedder (sentence-transformers)
3. Qdrant connection + collection creation
4. Redis cache initialization
5. Unified embedder (Jina-CLIP v2) -- conditional on `UNIFIED_ENABLED`
6. Hybrid retriever -- conditional on `UNIFIED_ENABLED`
7. Knowledge graph -- conditional on `GRAPH_ENABLED`
8. Semantic cache -- conditional on `SEMANTIC_CACHE_ENABLED`
9. OpenTelemetry -- conditional on `OTEL_ENABLED`
10. Auth + rate limiting middleware

Each V2 component is wrapped in a try/except so that failures are non-fatal. The system degrades gracefully to V1 behavior if a V2 component fails to initialize.

### Shutdown Sequence

1. Knowledge graph is persisted to JSON
2. Thread pool executor is shut down

---

## Thread Model

| Thread Pool | Workers | Purpose |
|-------------|---------|---------|
| `qdrant` | 4 | Qdrant search and upsert operations |
| `search` | 4 | Search endpoint CPU-bound work |
| `chat` | 4 | Chat endpoint CPU-bound work |
| `upload` | 4 | Upload endpoint CPU-bound work |
| `web` | 2 | Web scraping CPU-bound work |

Async I/O (web scraping via httpx, SSE streaming) runs on the asyncio event loop. CPU-bound work (embedding, reranking) runs in ThreadPoolExecutors to avoid blocking the event loop.
