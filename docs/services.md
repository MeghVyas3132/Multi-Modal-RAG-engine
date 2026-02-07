# Services

This document provides a technical deep-dive into each service module in the Multi-Modal RAG Engine.

---

## Embedding Service

**Location**: `services/embedding_service/`

The embedding service encodes text and images into fixed-dimensional vector representations for similarity search.

### Factory (`__init__.py`)

The factory function `create_embedder_auto()` selects the backend at startup:
- If `USE_ONNX=true` and the ONNX model file exists, returns `ONNXCLIPEmbedder`
- Otherwise, returns `CLIPEmbedder` (PyTorch)

### CLIPEmbedder (`embedder.py`)

PyTorch-based CLIP encoder using OpenCLIP. Supports text encoding, image encoding, and batch operations. The model is loaded once at startup and kept in memory.

Key methods:
- `encode_text(text: str) -> np.ndarray` -- Single text to 512/1024-dim vector
- `encode_text_batch(texts: List[str]) -> np.ndarray` -- Batch text encoding
- `encode_images(images: List) -> np.ndarray` -- Pre-processed image encoding
- `preprocess(image: PIL.Image) -> tensor` -- Image preprocessing for CLIP

### ONNXCLIPEmbedder (`onnx_embedder.py`)

Drop-in replacement for CLIPEmbedder using ONNX Runtime. Provides 2-3x faster CPU inference by replacing the PyTorch forward pass with an optimized ONNX graph. The tokenizer remains in Python (OpenCLIP).

Configuration:
- Thread counts for intra/inter-op parallelism
- Execution mode (parallel or sequential)
- Execution providers (CPU, CUDA, TensorRT)
- Warmup passes at initialization to pre-allocate buffers

### TextEmbedder (`text_embedder.py`)

Sentence-transformers model (all-MiniLM-L6-v2, 384-dim) for text chunk encoding. Used for the legacy `pdf_text_vectors` collection.

### UnifiedEmbedder (`unified_embedder.py`)

Jina-CLIP v2 model that encodes both text and images into the same 768-dimensional vector space. This is the primary embedder in V2.

Key methods:
- `encode_text(text: str) -> np.ndarray` -- Text to 768-dim vector
- `encode_text_batch(texts: List[str]) -> np.ndarray` -- Batch text encoding
- `encode_image(image: PIL.Image) -> np.ndarray` -- Image to 768-dim vector
- `encode_image_batch(images: List[PIL.Image]) -> np.ndarray` -- Batch image encoding

The unified vector space enables cross-modal search: a text query can find relevant images, and an image query can find relevant text.

---

## Retrieval Service

**Location**: `services/retrieval_service/`

### Legacy Retriever (`retriever.py`)

Direct Qdrant client for the legacy `image_vectors` and `pdf_text_vectors` collections. Handles collection creation, HNSW configuration, vector upsert, and similarity search.

### Hybrid Retriever (`hybrid_retriever.py`)

The V2 retriever operates on the `unified_vectors` collection and implements:

1. **Graph expansion**: Before searching, expands the query with related entities from the knowledge graph via BFS traversal
2. **Multi-query search**: Runs separate Qdrant searches for the original query and each expanded query
3. **Reciprocal Rank Fusion (RRF)**: Merges results from multiple searches using RRF scoring: `score = sum(1 / (k + rank))` where `k = 60`
4. **Modality weighting**: Adjusts scores based on the modality router output (e.g., boost image results for visual queries)
5. **Deduplication**: Removes near-duplicate results using payload hash matching

Key method: `search(query_vector, top_k, score_threshold, modality_weights, expand_graph) -> List[dict]`

### Reranker (`reranker.py`)

Cross-encoder precision layer using `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters). The model is lazy-loaded on first use.

Process:
1. Takes the top N candidates from ANN retrieval (default N=50)
2. Scores each (query, document) pair with full cross-attention
3. Returns the top K results (default K=10) sorted by cross-encoder score

Only text-modality results are reranked. Image results pass through unchanged.

---

## Cache Service

**Location**: `services/cache_service/`

### Semantic Cache (`semantic_cache.py`)

Three-tier cache with both exact and semantic matching:

| Tier | Backend | Match | TTL | Size |
|------|---------|-------|-----|------|
| L1 | OrderedDict LRU | Exact query hash | None (LRU eviction) | 500 entries |
| L2 | Redis | Exact query hash | 24 hours | Unlimited |
| L3 | Qdrant | Cosine > 0.93 | None | Unlimited |

The L3 tier is the key innovation: it stores query embeddings in a dedicated `cache_vectors` Qdrant collection. When a new query arrives, its embedding is compared against cached queries using ANN search. If the cosine similarity exceeds 0.93, the cached result is returned. This handles paraphrased queries.

The cache also stores LLM responses separately with a 7-day TTL, keyed by the query + context hash. This avoids re-running the LLM for identical RAG pipelines.

### Deduplication Service (`deduplication.py`)

Two-stage duplicate detection for document chunks:

1. **Exact dedup**: SHA256 hash of the chunk text. O(1) lookup.
2. **Semantic dedup**: Cosine similarity > 0.98 against existing vectors in the collection. Catches near-duplicates with minor formatting differences.

Used during PDF upload and web indexing to prevent redundant vectors.

---

## Document Service

**Location**: `services/document_service/`

### Semantic Chunker (`semantic_chunker.py`)

Splits text into chunks based on topic boundaries rather than fixed character counts.

Algorithm:
1. Split text into sentences
2. Encode each sentence with the text embedder
3. Compute cosine similarity between consecutive sentence embeddings
4. When similarity drops below the threshold (default 0.75), start a new chunk
5. Enforce minimum (128 chars) and maximum (1024 chars) chunk sizes

The `auto_chunk()` method accepts raw text and returns a list of `Chunk` objects with content, metadata, and detected language.

---

## Graph Service

**Location**: `services/graph_service/`

### Entity Extractor (`entity_extractor.py`)

Uses the LLM (Cerebras/Groq) to extract named entities and relationships from text chunks. The prompt instructs the LLM to return structured JSON with:
- `entities`: list of `{name, type, description}`
- `relationships`: list of `{source, target, relation}`

The `extract_entities_batch()` function processes multiple chunks in parallel.

### Knowledge Graph (`knowledge_graph.py`)

In-memory NetworkX DiGraph with the following capabilities:

- **Add entities**: nodes with type, description, and source metadata
- **Add relationships**: directed edges with relation labels
- **Find related**: BFS traversal up to N hops from a given entity
- **Expand query**: Find entities mentioned in a query string, then return related entities as expansion terms
- **Persistence**: Save/load to JSON file
- **Thread safety**: All mutations are protected by a threading lock

The graph is initialized at startup and persisted to disk on shutdown.

---

## VLM Service

**Location**: `services/vlm_service/`

### Local VLM (`local_vlm.py`)

SmolVLM-500M-Instruct model for image captioning. Loaded via HuggingFace Transformers.

Pipeline:
1. Image is preprocessed and passed through the VLM
2. Model generates a natural language caption (max 256 tokens)
3. Confidence is estimated from generation probabilities
4. If confidence < 0.85, falls back to GPT-4o-mini via OpenAI API

Captions are cached on disk (`data/vlm_cache/`) keyed by image content hash. The `CaptionCache` class manages this persistence.

---

## Routing Service

**Location**: `services/routing_service/`

### Modality Router (`modality_router.py`)

Classifies queries into modality probabilities: `{text: float, image: float, hybrid: float}`.

Heuristic mode (default) uses keyword patterns:
- Image indicators: "show me", "picture of", "what does X look like"
- Text indicators: "explain", "describe the process", "what is"
- Hybrid: mixed signals

Results are cached in an LRU cache (1000 entries). The `get_primary_modality()` helper returns the highest-probability modality.

---

## Web Service

**Location**: `services/web_service/`

### Web Scraper (`web_scraper.py`)

Multi-tier scraping with platform-specific handlers:

| Platform | Method | Details |
|----------|--------|---------|
| YouTube | `scrape_youtube()` | Transcript extraction via youtube-transcript-api |
| GitHub | `scrape_github()` | README and code file extraction |
| General web | `scrape_jina()` | Jina Reader API (free tier, returns markdown) |
| JS-rendered | `scrape_firecrawl()` | Firecrawl API (paid, handles SPAs) |
| Fallback | `scrape_httpx()` | Direct HTTP with HTML-to-text extraction |

The `scrape_url()` entry point auto-detects the source type and routes to the appropriate scraper with fallback chain.

### Change Detector

Tracks SHA256 hashes of previously scraped content. On re-index, compares the new hash against the stored hash. If unchanged, skips re-indexing.

---

## LLM Service

**Location**: `services/llm_service/`

### LLM (`llm.py`)

OpenAI-compatible SDK client for LLM providers:

- **Primary**: Cerebras Cloud (llama-3.3-70b, ~2100 tokens/sec)
- **Fallback**: Groq Cloud (llama-3.3-70b, ~300 tokens/sec)

Supports both streaming (SSE for /chat) and non-streaming (entity extraction) modes. Auto-failover: if Cerebras returns an error, the request is retried on Groq.

---

## PDF Service

**Location**: `services/pdf_service/`

### Parser (`parser.py`)

PyMuPDF-based PDF parsing. Extracts:
- Text blocks with page numbers
- Embedded images with coordinates
- Document metadata (title, author, page count)

Text is split into chunks with configurable size (default 512 chars) and overlap (default 64 chars) for fixed chunking, or passed to the semantic chunker for topic-boundary splitting.

---

## API Gateway

**Location**: `services/api_gateway/`

### Application (`app.py`)

Thin FastAPI application that:
1. Runs the lifespan hook (startup/shutdown)
2. Mounts CORS middleware
3. Mounts static file serving for images
4. Includes all endpoint routers

Module-level accessors (`get_active_embedder()`, `get_text_embedder()`) allow endpoint modules to access shared state without circular imports.

### Endpoints

Six router modules handle all API endpoints. Each module uses lazy imports and `run_in_executor` for CPU-bound operations to avoid blocking the async event loop.

### Middleware

API key authentication via `X-API-Key` header and per-key rate limiting via slowapi. Both are optional and disabled by default.

### Telemetry

OpenTelemetry instrumentation with:
- TracerProvider with OTLP gRPC exporter
- MeterProvider with Prometheus-compatible histograms
- FastAPI auto-instrumentation for HTTP spans
- Manual span creation for embedding, retrieval, and cache operations
