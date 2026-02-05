# Multi-Modal RAG Image Search Engine

**Ultra-low latency text-to-image retrieval with CLIP + HNSW**

A production-grade search engine backend that accepts natural language queries ("a dog playing in the park") and returns the most semantically relevant images from a corpus of 1M+ images in **under 50ms**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [System Components](#system-components)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Performance Targets](#performance-targets)
- [CLIP Model Details](#clip-model-details)
- [Qdrant Configuration](#qdrant-configuration)
- [Configuration Reference](#configuration-reference)
- [Scaling Strategy](#scaling-strategy)
- [Limitations and Constraints](#limitations-and-constraints)
- [License](#license)

---

## Project Overview

This system solves text-to-image retrieval at scale. A user submits a natural language query through the REST API. The query is encoded into a dense vector using a CLIP text encoder. That vector is searched against a prebuilt HNSW index of image embeddings stored entirely in RAM. The top-K most similar images are returned with their metadata and similarity scores.

**Core Properties:**

- **Dataset scale**: 1,000,000+ images (simulated or real)
- **Query type**: Text-to-image semantic search
- **Embedding model**: OpenCLIP ViT-H-14 (1024-dimensional vectors)
- **Index type**: HNSW approximate nearest neighbor (Qdrant)
- **Target retrieval latency**: Under 50ms end-to-end (excluding LLM)
- **All vectors reside in RAM** at all times
- **No image embedding occurs at query time**
- **CLIP model is loaded once at startup** and kept warm

---

## Architecture

The system is composed of four in-process services behind a single FastAPI gateway, backed by Qdrant for vector storage and Redis for optional caching.

```
                          +---------------------+
                          |   Client Request    |
                          |   POST /search      |
                          +----------+----------+
                                     |
                                     v
                  +------------------+------------------+
                  |     FastAPI Gateway (:8000)         |
                  |   - Input validation (Pydantic v2)  |
                  |   - Redis cache check (~0.3ms)      |
                  |   - Response assembly               |
                  |   - Latency instrumentation         |
                  +------------------+------------------+
                                     |
                              cache miss
                                     |
                                     v
                  +------------------+------------------+
                  |   CLIP Text Encoder (in-process)    |
                  |   ViT-H-14 / laion2b_s32b_b79k      |
                  |   1024-dim normalized vector        |
                  |   GPU: ~3-8ms / CPU: ~20-50ms       |
                  +------------------+------------------+
                                     |
                       normalized float32 vector
                                     |
                                     v
                  +------------------+------------------+
                  |   Qdrant Vector DB (HNSW, RAM)      |
                  |   gRPC interface (2x faster vs REST)|
                  |   int8 scalar quantization          |
                  |   ~2-10ms per search                |
                  +------------------+------------------+
                                     |
                       top-K results + metadata
                                     |
                                     v
                  +------------------+------------------+
                  |     Top-K Results + Metadata        |
                  |   Optional: async LLM explanation   |
                  +--------------------------------------+
```

### Design Principles

- **No image embedding at query time** — Images are pre-encoded offline. Only text is encoded on the hot path.
- **No disk reads during retrieval** — All vectors and payloads reside in RAM.
- **CLIP model is warm-loaded** — A single forward pass runs during startup to trigger CUDA kernel compilation and memory allocation.
- **In-process service calls** — The embedding and retrieval services are function calls within the same process, not network hops. This eliminates ~2-5ms of serialization overhead per request.
- **Single worker per container** — The CLIP model lives in GPU memory. Multiple workers would duplicate the model and risk OOM. Horizontal scaling is achieved through container replicas behind a load balancer.
- **Cache is fail-open** — Redis unavailability degrades to live search with no error propagation. Search never fails because of cache issues.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Single process, single worker** | CLIP model lives in GPU memory. Multiple workers = multiple model copies = OOM. Scale horizontally with containers. |
| **In-process service calls** | No HTTP/gRPC between embedding → retrieval. Saves ~2-5ms per request. Service boundaries are in code, not network. |
| **gRPC for Qdrant** | ~2x faster than REST for vector search operations. |
| **Int8 scalar quantization** | Reduces vector memory ~4x (4GB → 1GB for 1M vectors) with <1% recall loss. |
| **Redis cache is fail-open** | Cache miss = live search. Cache error = live search. Search NEVER fails because of cache. |
| **No image embedding at query time** | Images are pre-encoded offline. Only text is encoded on the hot path. |
| **Warm model loading** | A dummy forward pass executes during startup to pre-compile CUDA kernels and allocate memory. No request ever hits a cold model. |

---

## System Components

### 1. API Gateway

FastAPI application serving as the single entry point. Handles input validation, cache lookup, response assembly, and latency instrumentation.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search` | Text-to-image semantic search |
| `GET` | `/health` | Liveness and readiness probe |
| `GET` | `/stats` | Runtime latency percentiles and collection metadata |

**Responsibilities:**

- Validate inbound requests using Pydantic v2 schemas
- Check Redis for cached responses before invoking compute
- Coordinate the embedding and retrieval services
- Record latency metrics for every request
- Cache responses for repeated queries
- Optionally invoke the LLM service for explanations

### 2. Embedding Service

Singleton CLIP text encoder that remains loaded in GPU/CPU memory for the entire process lifetime.

**Key Properties:**

- **Model**: OpenCLIP ViT-H-14 with laion2b_s32b_b79k weights
- **Output**: L2-normalized 1024-dimensional float32 vector
- **Optimization**: `torch.compile()` with `reduce-overhead` mode, `torch.amp.autocast` for mixed precision
- **Thread safety**: PyTorch inference under `torch.no_grad()` is safe for concurrent reads
- **Warmup**: A dummy forward pass runs at initialization to pre-compile CUDA kernels

The model is never reinitialized. It is created once during application startup and reused for every request.

### 3. Retrieval Service

Qdrant client wrapper using gRPC for vector search operations.

**Collection Configuration:**

- **Vector size**: 1024 dimensions
- **Distance metric**: Cosine similarity
- **Index type**: HNSW with `m=16`, `ef_construct=200`
- **Search parameter**: `ef=128` (tunable per-query)
- **Storage**: All vectors and payloads in RAM (`on_disk=false`)
- **Quantization**: Scalar int8 (`quantile=0.99`, `always_ram=true`), reducing memory ~4x with <1% recall loss

**Memory Estimates:**

| Scale | Raw Vectors | With int8 Quantization | Total (incl. HNSW graph) |
|-------|-------------|------------------------|--------------------------|
| 1M images | ~4 GB | ~1 GB | ~2 GB |
| 10M images | ~40 GB | ~10 GB | ~20 GB |
| 100M images | Requires distributed mode | ~100 GB | ~200 GB |

### 4. LLM Service (Optional)

Async, non-blocking explanation generator. Accepts retrieved metadata and produces a brief natural language summary of why the results match the query.

- **Disabled by default** (`LLM_ENABLED=false`)
- Uses OpenAI async client
- **Never blocks the retrieval path** — explanation is appended after results are assembled
- Results are capped at top 5 for prompt construction to limit token usage
- Responses are generated with low temperature (0.3) for factual consistency

### 5. Indexing Pipeline (Offline)

Batch image encoding script that runs independently of the API server.

- Scans a directory recursively for image files (JPEG, PNG, WebP, BMP, TIFF)
- Encodes images through CLIP's vision encoder in configurable batch sizes (default 256)
- Generates deterministic integer IDs via MD5 hash of file path (idempotent re-runs)
- Upserts vectors and metadata into Qdrant
- Writes a JSON manifest for auditing
- Supports GPU acceleration and progress logging

### 6. Dataset Simulator

Generates random unit-norm vectors matching the CLIP embedding distribution for load testing and benchmarking without requiring real images or the CLIP model.

- Default: 1,000,000 vectors in batches of 10,000
- Includes synthetic metadata with categories for filter testing
- Vectors are drawn from N(0,1) and L2-normalized to the unit hypersphere
- Indexing time: approximately 5-10 minutes for one million vectors

### 7. Benchmark Tool

End-to-end latency benchmarking script that sends configurable numbers of search requests to the running API and reports percentile statistics.

- Cycles through 15 predefined natural language queries
- Reports P50, P90, P95, P99, min, max, mean, and standard deviation
- Validates results against the 50ms P95 target
- Supports configurable request count and top-K

### 8. Cache Layer

Redis-backed caching layer for repeated queries.

- **Cache key**: MD5 hash of (query + top_k + filters)
- **Default TTL**: 3600 seconds
- **Fail-open design**: All cache errors are silently swallowed
- **Connection pooling**: Up to 20 persistent connections
- **Serialization**: JSON

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- (Optional) NVIDIA GPU with CUDA for fast CLIP inference

### 1. Clone and Setup

```bash
cd rag-image-search
cp .env.example .env

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
make install
```

### 2. Start Infrastructure

```bash
make infra
# Starts Qdrant (:6333/:6334) and Redis (:6379)
```

### 3. Index Data

**Option A: Simulate 1M vectors** (no CLIP model needed, fast)

```bash
make simulate
# Generates 1M random unit-norm vectors → Qdrant
# Takes ~5-10 minutes depending on machine
```

**Option B: Index real images** (requires CLIP model download)

```bash
# Put images in ./data/images/
make index
```

### 4. Start the API Server

```bash
make serve
# Server starts on http://localhost:8000
# CLIP model loads and warms up (~4s)
```

### 5. Search!

```bash
# Quick test
make test-search

# Or use curl directly
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "a sunset over the ocean", "top_k": 5}'
```

### 6. Benchmark

```bash
python -m scripts.benchmark --num-requests 1000 --top-k 10
```

---

## API Reference

### `POST /search`

Search for images by natural language query.

**Request:**

```json
{
  "query": "a dog playing in the snow",
  "top_k": 10,
  "score_threshold": 0.2,
  "filters": {"category": "animals"},
  "include_explanation": false
}
```

**Request Schema:**

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `query` | `string` | yes | — | 1-512 characters | Natural language search query |
| `top_k` | `integer` | no | 10 | 1-100 | Number of results to return |
| `score_threshold` | `float` | no | 0.0 | 0.0-1.0 | Minimum cosine similarity score |
| `filters` | `object` | no | null | — | Metadata key-value filter pairs |
| `include_explanation` | `boolean` | no | false | — | Request LLM-generated explanation |

**Response:**

```json
{
  "query": "a dog playing in the snow",
  "results": [
    {
      "id": "8372651",
      "score": 0.8234,
      "metadata": {
        "file_path": "/data/images/img_00123.jpg",
        "file_name": "img_00123.jpg",
        "s3_key": "s3://image-bucket/images/img_00123.jpg",
        "category": "animals",
        "width": 640,
        "height": 480
      }
    }
  ],
  "total": 10,
  "latency_ms": 12.45,
  "cached": false,
  "explanation": null
}
```

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `query` | `string` | Echo of the input query |
| `results` | `array` | List of result objects |
| `total` | `integer` | Number of results returned |
| `latency_ms` | `float` | End-to-end server-side latency |
| `cached` | `boolean` | Whether the response was served from cache |
| `explanation` | `string` or `null` | LLM-generated explanation if requested |

**Result Object:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique vector point ID in Qdrant |
| `score` | `float` | Cosine similarity score (0.0 to 1.0) |
| `metadata` | `object` | Stored payload: file_path, file_name, s3_key, width, height, category |

---

### `GET /health`

Health check for load balancers and monitoring.

**Response:**

```json
{
  "status": "healthy",
  "clip_loaded": true,
  "qdrant_connected": true,
  "redis_connected": true,
  "device": "cuda"
}
```

---

### `GET /stats`

Runtime performance metrics with latency percentiles.

**Response:**

```json
{
  "metrics": {
    "uptime_seconds": 3600.0,
    "embedding_latency_ms": {
      "count": 10000,
      "p50_ms": 5.2,
      "p95_ms": 8.1,
      "p99_ms": 12.3,
      "mean_ms": 5.8
    },
    "search_latency_ms": {
      "count": 10000,
      "p50_ms": 3.1,
      "p95_ms": 7.2,
      "p99_ms": 11.0,
      "mean_ms": 3.5
    },
    "total_latency_ms": {
      "count": 10000,
      "p50_ms": 9.5,
      "p95_ms": 18.3,
      "p99_ms": 28.1,
      "mean_ms": 10.2
    }
  },
  "collection": {
    "collection": "image_vectors",
    "vectors_count": 1000000,
    "status": "GREEN",
    "indexed": true
  }
}
```

---

## Performance Targets

| Metric | Target | Typical (GPU) | Typical (CPU) |
|--------|--------|---------------|---------------|
| CLIP text encode | < 10ms | 3-8ms | 20-50ms |
| Qdrant HNSW search | < 10ms | 2-5ms | 2-5ms |
| Redis cache hit | < 1ms | 0.3ms | 0.3ms |
| **Total E2E (excl. LLM)** | **< 50ms** | **~12ms** | **~35ms** |
| P95 latency | < 50ms | ~18ms | ~45ms |
| P99 latency | < 100ms | ~28ms | ~60ms |

All latency metrics are tracked in-process via a rolling window of 10,000 observations and exposed through the `/stats` endpoint with P50, P95, and P99 percentiles.

---

## CLIP Model Details

| Property | Value |
|----------|-------|
| Model | ViT-H-14 (OpenCLIP) |
| Pretrained weights | laion2b_s32b_b79k |
| Vector dimension | 1024 |
| Text context length | 77 tokens |
| Model size | ~1GB |
| GPU VRAM required | ~2GB |

The model is loaded once at startup and kept warm in memory. Every search query only runs the text encoder (~3-8ms on GPU). Image encoding happens offline during indexing.

---

## Qdrant Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Vector size | 1024 | Must match CLIP ViT-H-14 output dimension |
| Distance metric | Cosine | Standard for normalized embeddings |
| HNSW m | 16 | Graph degree: balances recall, memory, and build time |
| HNSW ef_construct | 200 | Build-time beam width: higher values improve index quality |
| HNSW ef (search) | 128 | Search-time beam width: higher values improve recall at the cost of latency |
| on_disk (vectors) | false | All vectors must reside in RAM for target latency |
| on_disk (HNSW) | false | HNSW graph must reside in RAM |
| Quantization | Scalar int8 | Reduces memory ~4x with <1% recall loss |
| Quantization quantile | 0.99 | Calibration quantile for int8 range mapping |
| Quantization always_ram | true | Quantized vectors are never paged to disk |

---

## Configuration Reference

All configuration is managed through environment variables. See `.env.example` for the complete reference.

### CLIP Model

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIP_MODEL_NAME` | `ViT-H-14` | OpenCLIP model architecture |
| `CLIP_PRETRAINED` | `laion2b_s32b_b79k` | Pretrained weights identifier |
| `CLIP_VECTOR_DIM` | `1024` | Output embedding dimension (must match model) |
| `FORCE_CPU` | `false` | Disable GPU even if CUDA is available |

### Qdrant

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | REST API port |
| `QDRANT_GRPC_PORT` | `6334` | gRPC port (used for search) |
| `QDRANT_COLLECTION` | `image_vectors` | Collection name |
| `QDRANT_HNSW_M` | `16` | HNSW graph degree (higher = better recall, more RAM) |
| `QDRANT_HNSW_EF_CONSTRUCT` | `200` | HNSW build-time beam width |
| `QDRANT_HNSW_EF` | `128` | HNSW search-time beam width |

### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_ENABLED` | `true` | Enable/disable caching |
| `REDIS_CACHE_TTL` | `3600` | Cache entry TTL in seconds |

### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Bind port |
| `API_WORKERS` | `1` | Uvicorn worker count (keep at 1) |

### LLM Service

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENABLED` | `false` | Enable the LLM explanation service |
| `LLM_MODEL` | `gpt-4` | OpenAI model identifier |
| `OPENAI_API_KEY` | (empty) | OpenAI API key |

---

## Scaling Strategy

### Vertical Scaling (up to 10M images)

- Increase machine RAM. Qdrant handles 10M vectors on a single node with 32 GB RAM.
- Increase `QDRANT_HNSW_M` to 32 for better recall at higher cardinality.
- Use dedicated GPU with 8+ GB VRAM for faster CLIP inference.

### Horizontal Scaling (10M-100M images)

- Deploy Qdrant in cluster mode with sharding across nodes.
- Add read replicas for search throughput.
- Run multiple API containers behind a load balancer, each with its own GPU.
- Use Redis Cluster for distributed caching.

### Beyond 100M images

- Switch from scalar quantization to product quantization (PQ) for further memory reduction.
- Consider IVFPQ indexing for billion-scale datasets (trades latency for memory).
- Shard the CLIP inference layer across multiple GPUs with model parallelism.
- Pre-warm cache with known high-traffic queries.

---

## Project Structure

```
rag-image-search/
├── configs/
│   └── settings.py            # Pydantic settings from .env
├── services/
│   ├── api_gateway/
│   │   ├── app.py             # FastAPI application + endpoints
│   │   ├── models.py          # Request/Response schemas
│   │   └── cache.py           # Redis caching layer
│   ├── embedding_service/
│   │   └── embedder.py        # CLIP text/image encoder (warm-loaded)
│   ├── retrieval_service/
│   │   └── retriever.py       # Qdrant HNSW search client
│   └── llm_service/
│       └── llm.py             # Optional async LLM explanations
├── indexing/
│   └── index_images.py        # Offline image indexing pipeline
├── scripts/
│   ├── simulate_dataset.py    # Generate 1M synthetic vectors
│   └── benchmark.py           # Latency benchmarking tool
├── infra/
│   └── docker-compose.yml     # Qdrant + Redis local stack
├── utils/
│   ├── logger.py              # Structured logging (structlog)
│   ├── timing.py              # High-precision latency measurement
│   └── metrics.py             # In-process P50/P95/P99 tracker
├── .env.example               # Environment variable template
├── requirements.txt           # Pinned Python dependencies
└── Makefile                   # Common development commands
```

---

## Limitations and Constraints

1. **Single-node deployment baseline** — The default configuration targets a single machine. The architecture supports horizontal scaling, but distributed deployment requires manual configuration of Qdrant cluster mode and a load balancer.

2. **CLIP is the only embedding model** — The embedding service is hardcoded to OpenCLIP ViT-H-14. Swapping to SigLIP, EVA-CLIP, or another model requires modifying the embedder and adjusting the vector dimension configuration.

3. **Images are not served by this system** — The API returns file paths and S3 keys. Actual image delivery is the responsibility of a CDN, object storage service, or static file server.

4. **Metadata storage is limited** — Image metadata is stored in Qdrant payloads, which is sufficient for lightweight data (path, dimensions, category). Complex metadata queries or joins require a relational database sidecar such as PostgreSQL.

5. **No authentication or authorization** — The API is completely open. API key validation, OAuth, or mTLS must be added before any non-local deployment.

6. **No rate limiting** — The API does not enforce request rate limits. In production, this should be handled by a reverse proxy or API gateway (NGINX, Kong, Envoy).

7. **LangChain and LlamaIndex are explicitly excluded** — All orchestration uses direct function calls. This is a deliberate decision to avoid abstraction overhead and maintain full control over the hot path.

8. **Single worker constraint** — The CLIP model occupies GPU memory. Running multiple workers within the same container would duplicate the model and risk OOM. Scaling must be done through additional containers, not additional workers.

9. **HNSW recall is approximate** — HNSW search is not exact nearest neighbor search. With the default parameters (m=16, ef=128), recall at top-10 exceeds 95% on typical distributions, but some relevant results may be missed compared to brute-force search.

10. **Simulated dataset limitations** — The synthetic vector generator produces random unit-norm vectors that match the statistical distribution of CLIP embeddings but do not carry semantic meaning. Benchmark results with simulated data validate latency and throughput but not retrieval quality.

---

## License

MIT
