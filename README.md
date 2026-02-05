# Multi-Modal RAG Image Search Engine# 🔍 Multi-Modal RAG Image Search Engine



Ultra-low latency text-to-image retrieval system built on CLIP and HNSW approximate nearest neighbor search. Accepts natural language queries and returns semantically relevant images from a corpus of 1M+ images with end-to-end latency under 50ms.**Ultra-low latency text-to-image retrieval with CLIP + HNSW**



---A production-grade backend that takes a natural language query ("a dog playing in the park") and returns the most semantically relevant images from a corpus of 1M+ images in **under 50ms**.



## Architecture Overview---



```## 🏗️ Architecture

                          +---------------------+

                          |    Client Request    |```

                          |   POST /search       |┌──────────────────────────────────────────────────────────────────┐

                          +----------+----------+│                         Client (curl / UI)                       │

                                     |└──────────────────────────┬───────────────────────────────────────┘

                                     v                           │ POST /search

                  +------------------+------------------+                           ▼

                  |          FastAPI Gateway (:8000)     |┌──────────────────────────────────────────────────────────────────┐

                  |   Input validation (Pydantic v2)     |│                     FastAPI Gateway (:8000)                       │

                  |   Redis cache check (~0.3ms)         |│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │

                  +------------------+------------------+│  │ Input Valid. │→ │ Redis Cache  │→ │ Response Assembly       │ │

                                     |│  │ (Pydantic)  │  │ (optional)   │  │ + Latency Tracking      │ │

                              cache miss│  └─────────────┘  └──────┬───────┘  └─────────────────────────┘ │

                                     |└──────────────────────────┼──────────────────────────────────────┘

                                     v                  cache miss│

                  +------------------+------------------+                           ▼

                  |     CLIP Text Encoder (in-process)   |┌──────────────────────────────────────────────────────────────────┐

                  |   ViT-H-14 / laion2b_s32b_b79k      |│              CLIP Text Encoder (in-process, GPU)                 │

                  |   1024-dim normalized vector          |│  • Model warm-loaded at startup (~4s one-time cost)              │

                  |   GPU: ~3-8ms / CPU: ~20-50ms        |│  • ViT-H-14 + laion2b weights → 1024-dim vector                 │

                  +------------------+------------------+│  • torch.compile() + autocast for speed                          │

                                     |│  • ~3-8ms GPU / ~20-50ms CPU per query                           │

                                     v└──────────────────────────┬───────────────────────────────────────┘

                  +------------------+------------------+                           │ normalized float32 vector

                  |     Qdrant Vector DB (HNSW, RAM)     |                           ▼

                  |   gRPC interface (2x faster vs REST) |┌──────────────────────────────────────────────────────────────────┐

                  |   int8 scalar quantization            |│                 Qdrant Vector DB (HNSW, RAM)                     │

                  |   ~2-10ms per search                  |│  • 1M vectors in-memory, int8 scalar quantization                │

                  +------------------+------------------+│  • HNSW index: m=16, ef_construct=200, ef_search=128             │

                                     |│  • gRPC interface (~2x faster than REST)                         │

                                     v│  • ~2-10ms per search                                            │

                  +------------------+------------------+└──────────────────────────┬───────────────────────────────────────┘

                  |       Top-K Results + Metadata       |                           │ Top-K results + metadata

                  |   Optional: async LLM explanation    |                           ▼

                  +--------------------------------------+┌──────────────────────────────────────────────────────────────────┐

```│             (Optional) LLM Explanation Service                   │

│  • Async, non-blocking — never delays search results             │

### Design Principles│  • OpenAI GPT-4 via async client                                 │

│  • Disabled by default                                           │

- **No image embedding at query time.** Images are pre-encoded offline. Only text is encoded on the hot path.└──────────────────────────────────────────────────────────────────┘

- **No disk reads during retrieval.** All vectors and payloads reside in RAM.```

- **CLIP model is warm-loaded.** A single forward pass runs during startup to trigger CUDA kernel compilation and memory allocation. No request ever hits a cold model.

- **In-process service calls.** The embedding and retrieval services are function calls within the same process, not network hops. This eliminates ~2-5ms of serialization overhead per request.### Key Design Decisions

- **Single worker per container.** The CLIP model lives in GPU memory. Multiple workers would duplicate the model and risk OOM. Horizontal scaling is achieved through container replicas behind a load balancer.

- **Cache is fail-open.** Redis unavailability degrades to live search with no error propagation. Search never fails because of cache issues.| Decision | Rationale |

|----------|-----------|

---| **Single process, single worker** | CLIP model lives in GPU memory. Multiple workers = multiple model copies = OOM. Scale horizontally with containers. |

| **In-process service calls** | No HTTP/gRPC between embedding → retrieval. Saves ~2-5ms per request. Service boundaries are in code, not network. |

## System Components| **gRPC for Qdrant** | ~2x faster than REST for vector search operations. |

| **Int8 scalar quantization** | Reduces vector memory ~4x (4GB → 1GB for 1M vectors) with <1% recall loss. |

### 1. API Gateway| **Redis cache is fail-open** | Cache miss = live search. Cache error = live search. Search NEVER fails because of cache. |

| **No image embedding at query time** | Images are pre-encoded offline. Only text is encoded on the hot path. |

FastAPI application serving as the single entry point. Handles input validation, cache lookup, response assembly, and latency instrumentation.

---

**Endpoints:**

## 📁 Project Structure

| Method | Path | Description |

|--------|------|-------------|```

| `POST` | `/search` | Text-to-image semantic search |rag-image-search/

| `GET` | `/health` | Liveness and readiness probe |├── configs/

| `GET` | `/stats` | Runtime latency percentiles and collection metadata |│   └── settings.py            # Pydantic settings from .env

├── services/

**Search Request Schema:**│   ├── api_gateway/

│   │   ├── app.py             # FastAPI application + endpoints

| Field | Type | Default | Constraints |│   │   ├── models.py          # Request/Response schemas

|-------|------|---------|-------------|│   │   └── cache.py           # Redis caching layer

| `query` | `string` | required | 1-512 characters |│   ├── embedding_service/

| `top_k` | `integer` | 10 | 1-100 |│   │   └── embedder.py        # CLIP text/image encoder (warm-loaded)

| `score_threshold` | `float` | 0.0 | 0.0-1.0 |│   ├── retrieval_service/

| `filters` | `object` | null | Metadata key-value filter |│   │   └── retriever.py       # Qdrant HNSW search client

| `include_explanation` | `boolean` | false | Triggers optional LLM path |│   └── llm_service/

│       └── llm.py             # Optional async LLM explanations

**Search Response Schema:**├── indexing/

│   └── index_images.py        # Offline image indexing pipeline

| Field | Type | Description |├── scripts/

|-------|------|-------------|│   ├── simulate_dataset.py    # Generate 1M synthetic vectors

| `query` | `string` | Echo of the input query |│   └── benchmark.py           # Latency benchmarking tool

| `results` | `array` | List of `{id, score, metadata}` objects |├── infra/

| `total` | `integer` | Number of results returned |│   └── docker-compose.yml     # Qdrant + Redis local stack

| `latency_ms` | `float` | End-to-end server-side latency |├── utils/

| `cached` | `boolean` | Whether the response was served from cache |│   ├── logger.py              # Structured logging (structlog)

| `explanation` | `string` | LLM-generated explanation (if requested) |│   ├── timing.py              # High-precision latency measurement

│   └── metrics.py             # In-process P50/P95/P99 tracker

### 2. Embedding Service├── .env.example               # Environment variable template

├── requirements.txt           # Pinned Python dependencies

Singleton CLIP text encoder that remains loaded in GPU/CPU memory for the entire process lifetime.└── Makefile                   # Common development commands

```

**Key properties:**

- Model: OpenCLIP ViT-H-14 with laion2b_s32b_b79k weights---

- Output: L2-normalized 1024-dimensional float32 vector

- Optimization: `torch.compile()` with `reduce-overhead` mode on GPU, `torch.amp.autocast` for mixed precision inference## 🚀 Quick Start

- Thread safety: PyTorch inference under `torch.no_grad()` is safe for concurrent reads

- Warmup: A dummy forward pass runs at initialization to pre-compile CUDA kernels### Prerequisites



### 3. Retrieval Service- Python 3.10+

- Docker & Docker Compose

Qdrant client wrapper using gRPC for vector search operations.- (Optional) NVIDIA GPU with CUDA for fast CLIP inference



**Collection configuration:**### 1. Clone and Setup

- Vector size: 1024 dimensions

- Distance metric: Cosine similarity```bash

- Index type: HNSW with `m=16`, `ef_construct=200`cd rag-image-search

- Search parameter: `ef=128` (tunable per-query)cp .env.example .env

- Storage: All vectors and payloads in RAM (`on_disk=false`)

- Quantization: Scalar int8 (`quantile=0.99`, `always_ram=true`), reducing memory ~4x with less than 1% recall loss# Create and activate virtual environment

python3 -m venv venv

**Memory estimates:**source venv/bin/activate



| Scale | Raw Vectors | With int8 Quantization | Total (incl. HNSW graph) |# Install dependencies

|-------|-------------|------------------------|--------------------------|make install

| 1M images | ~4 GB | ~1 GB | ~2 GB |```

| 10M images | ~40 GB | ~10 GB | ~20 GB |

| 100M images | Requires distributed mode | ~100 GB | ~200 GB |### 2. Start Infrastructure



### 4. LLM Service (Optional)```bash

make infra

Async, non-blocking explanation generator. Accepts retrieved metadata and produces a brief natural language summary of why the results match the query.# Starts Qdrant (:6333/:6334) and Redis (:6379)

```

- Disabled by default (`LLM_ENABLED=false`)

- Uses OpenAI async client### 3. Index Data

- Never blocks the retrieval path -- explanation is appended after results are assembled

- Results are capped at top 5 for prompt construction to limit token usage**Option A: Simulate 1M vectors** (no CLIP model needed, fast)

```bash

### 5. Indexing Pipeline (Offline)make simulate

# Generates 1M random unit-norm vectors → Qdrant

Batch image encoding script that runs independently of the API server.# Takes ~5-10 minutes depending on machine

```

- Scans a directory recursively for image files (JPEG, PNG, WebP, BMP, TIFF)

- Encodes images through CLIP's vision encoder in configurable batch sizes**Option B: Index real images** (requires CLIP model download)

- Generates deterministic integer IDs via MD5 hash of file path (idempotent re-runs)```bash

- Upserts vectors and metadata into Qdrant# Put images in ./data/images/

- Writes a JSON manifest for auditingmake index

- Supports GPU acceleration and progress logging```



### 6. Dataset Simulator### 4. Start the API Server



Generates random unit-norm vectors matching the CLIP embedding distribution for load testing and benchmarking without requiring real images or the CLIP model.```bash

make serve

- Default: 1,000,000 vectors in batches of 10,000# Server starts on http://localhost:8000

- Includes synthetic metadata with categories for filter testing# CLIP model loads and warms up (~4s)

- Vectors are drawn from N(0,1) and L2-normalized to the unit hypersphere```



---### 5. Search!



## Performance Targets```bash

# Quick test

| Metric | Target | Typical (GPU) | Typical (CPU) |make test-search

|--------|--------|---------------|---------------|

| CLIP text encode | < 10 ms | 3-8 ms | 20-50 ms |# Or use curl directly

| Qdrant HNSW search | < 10 ms | 2-5 ms | 2-5 ms |curl -X POST http://localhost:8000/search \

| Redis cache hit | < 1 ms | 0.3 ms | 0.3 ms |  -H "Content-Type: application/json" \

| **Total end-to-end (excl. LLM)** | **< 50 ms** | **~12 ms** | **~35 ms** |  -d '{"query": "a sunset over the ocean", "top_k": 5}'

| P95 latency | < 50 ms | ~18 ms | ~45 ms |```

| P99 latency | < 100 ms | ~28 ms | ~60 ms |

### 6. Benchmark

All latency metrics are tracked in-process via a rolling window of 10,000 observations and exposed through the `/stats` endpoint with P50, P95, and P99 percentiles.

```bash

---python -m scripts.benchmark --num-requests 1000 --top-k 10

```

## Infrastructure

---

### Docker Compose Services

## 📡 API Reference

| Service | Image | Ports | Purpose |

|---------|-------|-------|---------|### `POST /search`

| `app` | Custom (Dockerfile) | 8000 | FastAPI + CLIP inference |

| `qdrant` | qdrant/qdrant:v1.13.2 | 6333, 6334 | Vector DB (REST + gRPC) |Search for images by natural language query.

| `redis` | redis:7-alpine | 6379 | Query result cache |

**Request:**

### Dockerfile```json

{

Multi-stage build:  "query": "a dog playing in the snow",

1. **deps** stage: Installs Python packages into a cached layer. Rebuilds only when `requirements.txt` changes.  "top_k": 10,

2. **runtime** stage: Copies installed packages and application code. Thin layer for fast iteration.  "score_threshold": 0.2,

  "filters": {"category": "animals"},

The container runs a single uvicorn worker. Health checks poll `/health` every 30 seconds with a 60-second startup grace period (CLIP model loading).  "include_explanation": false

}

### Resource Limits```



| Service | Memory Limit | Notes |**Response:**

|---------|-------------|-------|```json

| app | 4 GB | CLIP model (~2 GB) + PyTorch overhead |{

| qdrant | 6 GB | 1M vectors quantized + HNSW graph + payloads |  "query": "a dog playing in the snow",

| redis | 512 MB | LRU eviction, no persistence |  "results": [

    {

---      "id": "8372651",

      "score": 0.8234,

## Project Structure      "metadata": {

        "file_path": "/data/images/img_00123.jpg",

```        "file_name": "img_00123.jpg",

.        "s3_key": "s3://image-bucket/images/img_00123.jpg",

+-- configs/        "category": "animals",

|   +-- settings.py              # Pydantic settings from environment        "width": 640,

+-- services/        "height": 480

|   +-- api_gateway/      }

|   |   +-- app.py               # FastAPI application and endpoints    }

|   |   +-- models.py            # Request/response Pydantic schemas  ],

|   |   +-- cache.py             # Redis caching layer (fail-open)  "total": 10,

|   +-- embedding_service/  "latency_ms": 12.45,

|   |   +-- embedder.py          # CLIP text/image encoder singleton  "cached": false,

|   +-- retrieval_service/  "explanation": null

|   |   +-- retriever.py         # Qdrant HNSW search client}

|   +-- llm_service/```

|       +-- llm.py               # Optional async LLM explanations

+-- indexing/### `GET /health`

|   +-- index_images.py          # Offline batch image indexing

+-- scripts/Health check for load balancers and monitoring.

|   +-- simulate_dataset.py      # Synthetic vector generation (1M)

|   +-- benchmark.py             # End-to-end latency benchmark```json

+-- utils/{

|   +-- logger.py                # Structured logging (structlog)  "status": "healthy",

|   +-- timing.py                # Nanosecond-precision timing  "clip_loaded": true,

|   +-- metrics.py               # In-process percentile tracker  "qdrant_connected": true,

+-- docker-compose.yml           # Full stack orchestration  "redis_connected": true,

+-- Dockerfile                   # Multi-stage app container  "device": "cuda"

+-- Makefile                     # Development task runner}

+-- requirements.txt             # Pinned Python dependencies```

+-- .env.example                 # Environment variable reference

```### `GET /stats`



---Runtime performance metrics with latency percentiles.



## Configuration Reference```json

{

All configuration is managed through environment variables. See `.env.example` for the complete reference.  "metrics": {

    "uptime_seconds": 3600.0,

### CLIP Model    "embedding_latency_ms": {

      "count": 10000,

| Variable | Default | Description |      "p50_ms": 5.2,

|----------|---------|-------------|      "p95_ms": 8.1,

| `CLIP_MODEL_NAME` | `ViT-H-14` | OpenCLIP model architecture |      "p99_ms": 12.3,

| `CLIP_PRETRAINED` | `laion2b_s32b_b79k` | Pretrained weights identifier |      "mean_ms": 5.8

| `CLIP_VECTOR_DIM` | `1024` | Output embedding dimension (must match model) |    },

| `FORCE_CPU` | `false` | Disable GPU even if CUDA is available |    "search_latency_ms": {

      "count": 10000,

### Qdrant      "p50_ms": 3.1,

      "p95_ms": 7.2,

| Variable | Default | Description |      "p99_ms": 11.0,

|----------|---------|-------------|      "mean_ms": 3.5

| `QDRANT_HOST` | `localhost` | Qdrant server hostname |    },

| `QDRANT_PORT` | `6333` | REST API port |    "total_latency_ms": {

| `QDRANT_GRPC_PORT` | `6334` | gRPC port (used for search) |      "count": 10000,

| `QDRANT_COLLECTION` | `image_vectors` | Collection name |      "p50_ms": 9.5,

| `QDRANT_HNSW_M` | `16` | HNSW graph degree (higher = better recall, more RAM) |      "p95_ms": 18.3,

| `QDRANT_HNSW_EF_CONSTRUCT` | `200` | HNSW build-time beam width |      "p99_ms": 28.1,

| `QDRANT_HNSW_EF` | `128` | HNSW search-time beam width |      "mean_ms": 10.2

    }

### Redis  },

  "collection": {

| Variable | Default | Description |    "collection": "image_vectors",

|----------|---------|-------------|    "vectors_count": 1000000,

| `REDIS_HOST` | `localhost` | Redis server hostname |    "status": "GREEN",

| `REDIS_PORT` | `6379` | Redis port |    "indexed": true

| `REDIS_ENABLED` | `true` | Enable/disable caching |  }

| `REDIS_CACHE_TTL` | `3600` | Cache entry TTL in seconds |}

```

### API Server

---

| Variable | Default | Description |

|----------|---------|-------------|## ⚡ Performance Targets

| `API_HOST` | `0.0.0.0` | Bind address |

| `API_PORT` | `8000` | Bind port || Metric | Target | Typical (GPU) | Typical (CPU) |

| `API_WORKERS` | `1` | Uvicorn worker count (keep at 1) ||--------|--------|---------------|---------------|

| CLIP text encode | < 10ms | 3-8ms | 20-50ms |

---| Qdrant HNSW search | < 10ms | 2-5ms | 2-5ms |

| Redis cache hit | < 1ms | 0.3ms | 0.3ms |

## Scaling Considerations| **Total E2E (excl. LLM)** | **< 50ms** | **~12ms** | **~35ms** |

| P95 latency | < 50ms | ~18ms | ~45ms |

### Vertical Scaling (up to 10M images)| P99 latency | < 100ms | ~28ms | ~60ms |



- Increase machine RAM. Qdrant handles 10M vectors on a single node with 32 GB RAM.---

- Increase `QDRANT_HNSW_M` to 32 for better recall at higher cardinality.

- Use dedicated GPU with 8+ GB VRAM for faster CLIP inference.## 🧠 CLIP Model Details



### Horizontal Scaling (10M-100M images)| Property | Value |

|----------|-------|

- Deploy Qdrant in cluster mode with sharding across nodes.| Model | ViT-H-14 (OpenCLIP) |

- Add read replicas for search throughput.| Pretrained weights | laion2b_s32b_b79k |

- Run multiple API containers behind a load balancer, each with its own GPU.| Vector dimension | 1024 |

- Use Redis Cluster for distributed caching.| Text context length | 77 tokens |

| Model size | ~1GB |

### Beyond 100M images| GPU VRAM required | ~2GB |



- Switch from scalar quantization to product quantization (PQ) for further memory reduction.The model is loaded once at startup and kept warm in memory. Every search

- Consider IVFPQ indexing for billion-scale datasets (trades latency for memory).query only runs the text encoder (~3-8ms on GPU). Image encoding happens

- Shard the CLIP inference layer across multiple GPUs with model parallelism.offline during indexing.

- Pre-warm cache with known high-traffic queries.

---

---

## 🔧 Configuration

## Technology Stack

All configuration is via environment variables. See `.env.example` for the complete list.

| Component | Technology | Version |

|-----------|-----------|---------|Key tuning parameters:

| Language | Python | 3.10+ |

| API Framework | FastAPI | 0.115.6 || Variable | Default | Effect |

| ASGI Server | Uvicorn | 0.34.0 ||----------|---------|--------|

| ML Framework | PyTorch | 2.1+ || `QDRANT_HNSW_M` | 16 | Higher = better recall, more RAM |

| CLIP Implementation | OpenCLIP | 2.29.0 || `QDRANT_HNSW_EF_CONSTRUCT` | 200 | Higher = better index quality, slower build |

| Vector Database | Qdrant | 1.13.2 || `QDRANT_HNSW_EF` | 128 | Higher = better search recall, slower search |

| Cache | Redis | 7.x || `CLIP_VECTOR_DIM` | 1024 | Must match the model |

| Serialization | Pydantic | 2.10.4 || `SEARCH_TOP_K` | 10 | Default results per query |

| Logging | structlog | 24.x || `REDIS_CACHE_TTL` | 3600 | Cache expiry in seconds |

| Containerization | Docker Compose | v2 |

---

---

## 📈 Scaling to 10M+ Images

## Constraints and Assumptions

### Current limits (single node)

1. **Single-node deployment baseline.** The architecture supports horizontal scaling but the default configuration targets a single machine.

2. **CLIP is the sole embedding model.** The embedding service interface is simple enough to swap for SigLIP or EVA-CLIP, but no abstraction layer exists for model-agnostic operation.- **1M vectors**: ~2GB Qdrant RAM (with int8 quantization)

3. **Images are stored externally.** The system returns file paths and S3 keys. Image serving is the responsibility of a CDN or object storage layer.- **10M vectors**: ~20GB Qdrant RAM

4. **Metadata is stored in Qdrant payloads.** Sufficient for lightweight metadata (path, dimensions, category). For rich metadata with complex queries, a relational sidecar (PostgreSQL) should be introduced.- **100M vectors**: Requires Qdrant distributed mode

5. **No authentication or authorization.** API security (API keys, OAuth, mTLS) must be added before any non-local deployment.

6. **LangChain and LlamaIndex are explicitly excluded.** All orchestration is direct function calls to avoid unnecessary abstraction overhead.### Scaling strategy



---1. **Vertical (10M)**: Bigger machine, more RAM. Qdrant handles 10M vectors on a single node with 32GB RAM.



## License2. **Horizontal (100M+)**:

   - Deploy Qdrant in cluster mode (sharding across nodes)

MIT   - Add replicas for read throughput

   - Use Qdrant's built-in distributed search

3. **Multi-GPU**:
   - Run multiple API server containers, each with its own GPU
   - Load balance with NGINX/Envoy
   - Each container loads its own CLIP model

4. **Index optimization**:
   - Increase `HNSW_M` to 32-48 for higher recall at 100M scale
   - Use product quantization (PQ) instead of scalar for 100M+
   - Consider IVFPQ for billion-scale (trade latency for memory)

5. **Caching**:
   - Redis Cluster for distributed caching
   - Increase TTL for popular queries
   - Pre-warm cache with known popular queries

---

## 🧪 Development

```bash
# Run with auto-reload (dev mode)
make serve-dev

# Run tests
make test

# Check health
make health

# View stats
make stats

# Tear down infrastructure
make infra-down

# Clean generated files
make clean
```

---

## 🔍 Assumptions

1. **Single-node deployment** for now. The architecture is designed to be horizontally scalable but the current implementation runs on one machine.

2. **CLIP is the only embedding model**. The embedding service interface is simple enough to swap for other models (SigLIP, EVA-CLIP) but the current code is hardcoded to OpenCLIP.

3. **Images are stored externally**. The system returns image paths/S3 keys. It does not serve image files — that's the job of a CDN or object store.

4. **Metadata is stored in Qdrant payloads**. For simple metadata (file path, size, category), this is sufficient. For rich metadata, add a PostgreSQL sidecar.

5. **No authentication**. Add an API key middleware or OAuth for production.

---

## License

MIT
