# Multi-Modal RAG Image Search Engine# Multi-Modal RAG Image Search Engine



A production-grade, ultra-low latency text-to-image retrieval system. It accepts natural language queries such as "a dog playing in the park" and returns the most semantically relevant images from a corpus of over one million images. End-to-end retrieval latency targets are under 50 milliseconds.



This is not a prototype. It is structured as a real search engine backend with clear service boundaries, precomputed embeddings, warm-loaded models, and zero disk access on the hot path.A production-grade, ultra-low latency text-to-image retrieval system. It accepts natural language queries such as "a dog playing in the park" and returns the most semantically relevant images from a corpus of over one million images. The system is built on OpenCLIP for embedding generation and Qdrant with HNSW indexing for approximate nearest neighbor search. End-to-end retrieval latency targets are under 50 milliseconds, excluding optional LLM explanation.



---



## Table of ContentsThis is not a prototype or a demo. It is structured as a real search engine backend, optimized for speed above everything else, with clear service boundaries, precomputed embeddings, warm-loaded models, and zero disk access on the hot path.Ultra-low latency text-to-image retrieval system built on CLIP and HNSW approximate nearest neighbor search. Accepts natural language queries and returns semantically relevant images from a corpus of 1M+ images with end-to-end latency under 50ms.**Ultra-low latency text-to-image retrieval with CLIP + HNSW**



- [Architecture](#architecture)

- [Technology Stack](#technology-stack)

- [Project Structure](#project-structure)---

- [Quick Start](#quick-start)

- [ONNX Runtime Acceleration](#onnx-runtime-acceleration)

- [Kubernetes Deployment](#kubernetes-deployment)

- [Observability](#observability)## Table of Contents---A production-grade backend that takes a natural language query ("a dog playing in the park") and returns the most semantically relevant images from a corpus of 1M+ images in **under 50ms**.

- [Authentication and Rate Limiting](#authentication-and-rate-limiting)

- [CI/CD Pipeline](#cicd-pipeline)

- [API Reference](#api-reference)

- [Configuration Reference](#configuration-reference)- [Project Overview](#project-overview)

- [Performance Targets](#performance-targets)

- [Scaling Strategy](#scaling-strategy)- [Architecture](#architecture)



---- [Request Lifecycle](#request-lifecycle)## Architecture Overview---



## Architecture- [System Components](#system-components)



```- [Technology Stack](#technology-stack)

Client (POST /search)

        |- [Project Structure](#project-structure)

        v

+-------+--------+     +----------+     +----------+- [API Reference](#api-reference)```## 🏗️ Architecture

|  FastAPI        |---->|  Redis   |     |  Qdrant  |

|  API Gateway    |     |  Cache   |     |  HNSW    |- [Performance Targets](#performance-targets)

|                 |     +----------+     +----------+

|  CLIP ViT-H-14 |          ^                ^- [CLIP Model Specification](#clip-model-specification)                          +---------------------+

|  (PyTorch/ONNX) |         |                |

+-----------------+---------+----------------+- [Qdrant Configuration](#qdrant-configuration)

        |

        v (optional)- [Infrastructure](#infrastructure)                          |    Client Request    |```

  OTel Collector -> Prometheus -> Grafana

                 -> Jaeger (traces)- [Configuration Reference](#configuration-reference)

```

- [Scaling Strategy](#scaling-strategy)                          |   POST /search       |┌──────────────────────────────────────────────────────────────────┐

**Request lifecycle (hot path):**

- [Limitations and Constraints](#limitations-and-constraints)

1. Check Redis cache (0.5ms)

2. CLIP text encode via PyTorch or ONNX Runtime (3-50ms)- [License](#license)                          +----------+----------+│                         Client (curl / UI)                       │

3. Qdrant HNSW search via gRPC (2-10ms)

4. Assemble and return response (0.1ms)



All services run in-process (function calls, not HTTP/gRPC between services). This eliminates 2-5ms of serialization overhead per request.---                                     |└──────────────────────────┬───────────────────────────────────────┘



---



## Technology Stack## Project Overview                                     v                           │ POST /search



| Component | Technology | Purpose |

|-----------|-----------|---------|

| API | FastAPI 0.115 + Uvicorn | Async HTTP server |The system solves text-to-image retrieval at scale. A user submits a natural language query through the REST API. The query is encoded into a dense vector using a CLIP text encoder. That vector is searched against a prebuilt HNSW index of image embeddings stored entirely in RAM. The top-K most similar images are returned with their metadata and similarity scores.                  +------------------+------------------+                           ▼

| Embeddings | OpenCLIP ViT-H-14 (1024-dim) | Text/image encoding |

| ONNX Runtime | onnxruntime 1.19.2 | 2-3x faster CPU inference |

| Vector DB | Qdrant 1.13.2 (gRPC, HNSW, int8 quantization) | ANN search |

| Cache | Redis 7 (LRU, fail-open) | Query result cache |**Core properties:**                  |          FastAPI Gateway (:8000)     |┌──────────────────────────────────────────────────────────────────┐

| Observability | OpenTelemetry + Prometheus + Grafana + Jaeger | Traces and metrics |

| Auth | API key middleware + slowapi rate limiting | Request authentication |

| Orchestration | Docker Compose (dev) / Kubernetes + Kustomize (prod) | Container management |

| CI/CD | GitHub Actions | Lint, test, build, deploy |- Dataset scale: 1,000,000+ images (simulated or real)                  |   Input validation (Pydantic v2)     |│                     FastAPI Gateway (:8000)                       │



---- Query type: text to image semantic search



## Project Structure- Embedding model: OpenCLIP ViT-H-14 (1024-dimensional vectors)                  |   Redis cache check (~0.3ms)         |│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │



```- Index type: HNSW approximate nearest neighbor (Qdrant)

.

├── configs/- Target retrieval latency: under 50ms end-to-end (excluding LLM)                  +------------------+------------------+│  │ Input Valid. │→ │ Redis Cache  │→ │ Response Assembly       │ │

│   └── settings.py              # Centralized Pydantic settings

├── services/- All vectors reside in RAM at all times

│   ├── api_gateway/

│   │   ├── app.py               # FastAPI application + lifespan- No image embedding occurs at query time                                     |│  │ (Pydantic)  │  │ (optional)   │  │ + Latency Tracking      │ │

│   │   ├── models.py            # Request/response Pydantic models

│   │   ├── cache.py             # Redis cache layer (fail-open)- CLIP model is loaded once at startup and kept warm

│   │   ├── telemetry.py         # OpenTelemetry instrumentation

│   │   └── middleware/                              cache miss│  └─────────────┘  └──────┬───────┘  └─────────────────────────┘ │

│   │       └── __init__.py      # API key auth + rate limiting

│   ├── embedding_service/---

│   │   ├── __init__.py          # Factory: auto-selects PyTorch or ONNX

│   │   ├── embedder.py          # PyTorch CLIP text encoder                                     |└──────────────────────────┼──────────────────────────────────────┘

│   │   └── onnx_embedder.py    # ONNX Runtime CLIP text encoder

│   ├── retrieval_service/## Architecture

│   │   └── retriever.py         # Qdrant HNSW search

│   └── llm_service/                                     v                  cache miss│

│       └── llm.py               # Optional GPT-4 explanations

├── scripts/The system is composed of four in-process services behind a single FastAPI gateway, backed by Qdrant for vector storage and Redis for optional caching.

│   ├── convert_to_onnx.py       # Export CLIP to ONNX format

│   ├── benchmark_onnx_vs_pytorch.py  # ONNX vs PyTorch benchmark                  +------------------+------------------+                           ▼

│   ├── benchmark.py             # Search latency benchmark

│   └── simulate_dataset.py      # Generate test vectors```

├── indexing/

│   └── index_images.py          # Image indexing pipeline                    +---------------------------+                  |     CLIP Text Encoder (in-process)   |┌──────────────────────────────────────────────────────────────────┐

├── utils/

│   ├── logger.py                # Structured logging (structlog)                    |      Client Request        |

│   ├── metrics.py               # In-process latency percentiles

│   └── timing.py                # Precision timing context manager                    |      POST /search          |                  |   ViT-H-14 / laion2b_s32b_b79k      |│              CLIP Text Encoder (in-process, GPU)                 │

├── models/onnx/                 # ONNX model files (git-ignored)

├── k8s/                         # Kubernetes manifests                    +-------------+-------------+

│   ├── base/                    # Namespace, ConfigMap, Secrets

│   ├── api/                     # Deployment, Service, HPA, PDB                                  |                  |   1024-dim normalized vector          |│  • Model warm-loaded at startup (~4s one-time cost)              │

│   ├── qdrant/                  # StatefulSet, Service, PVC

│   ├── redis/                   # Deployment, Service                                  v

│   ├── ingress/                 # Ingress + TLS

│   ├── monitoring/              # OTel Collector, Prometheus, Grafana                    +-------------+-------------+                  |   GPU: ~3-8ms / CPU: ~20-50ms        |│  • ViT-H-14 + laion2b weights → 1024-dim vector                 │

│   └── overlays/                # dev / staging / prod Kustomize

├── .github/workflows/ci.yaml   # CI/CD pipeline                    |     FastAPI Gateway        |

├── Dockerfile                   # Multi-stage build

├── docker-compose.yml           # Full stack with monitoring profile                    |     (:8000)                |                  +------------------+------------------+│  • torch.compile() + autocast for speed                          │

├── Makefile                     # Development task runner

└── requirements.txt             # Pinned dependencies                    |                           |

```

                    |  - Input validation       |                                     |│  • ~3-8ms GPU / ~20-50ms CPU per query                           │

---

                    |    (Pydantic v2)           |

## Quick Start

                    |  - Redis cache check      |                                     v└──────────────────────────┬───────────────────────────────────────┘

### Docker Compose (recommended)

                    |    (~0.3ms on hit)         |

```bash

# Start core stack (API + Qdrant + Redis)                    |  - Response assembly      |                  +------------------+------------------+                           │ normalized float32 vector

docker compose up -d

                    |  - Latency instrumentation |

# Start with full monitoring (adds OTel, Prometheus, Grafana, Jaeger)

docker compose --profile monitoring up -d                    +-------------+-------------+                  |     Qdrant Vector DB (HNSW, RAM)     |                           ▼



# Check health                                  |

curl http://localhost:8000/health

                           cache miss                  |   gRPC interface (2x faster vs REST) |┌──────────────────────────────────────────────────────────────────┐

# Run a search

curl -X POST http://localhost:8000/search \                                  |

  -H "Content-Type: application/json" \

  -d '{"query": "a dog playing in the park", "top_k": 5}'                                  v                  |   int8 scalar quantization            |│                 Qdrant Vector DB (HNSW, RAM)                     │

```

                    +-------------+-------------+

### Local Development

                    |   CLIP Text Encoder        |                  |   ~2-10ms per search                  |│  • 1M vectors in-memory, int8 scalar quantization                │

```bash

# Install dependencies                    |   (in-process, GPU/CPU)    |

make install

                    |                           |                  +------------------+------------------+│  • HNSW index: m=16, ef_construct=200, ef_search=128             │

# Start Qdrant + Redis

docker compose up qdrant redis -d                    |  - ViT-H-14 model         |



# Generate test data                    |  - laion2b_s32b_b79k      |                                     |│  • gRPC interface (~2x faster than REST)                         │

make simulate

                    |    weights                |

# Start API with hot-reload

make serve-dev                    |  - 1024-dim normalized    |                                     v│  • ~2-10ms per search                                            │

```

                    |    float32 output         |

---

                    |  - ~3-8ms on GPU          |                  +------------------+------------------+└──────────────────────────┬───────────────────────────────────────┘

## ONNX Runtime Acceleration

                    |  - ~20-50ms on CPU        |

ONNX Runtime provides 2-3x faster CPU inference by replacing PyTorch's text encoder with an optimized ONNX graph. The system automatically selects the backend at startup.

                    +-------------+-------------+                  |       Top-K Results + Metadata       |                           │ Top-K results + metadata

### Convert the Model

                                  |

```bash

# Export CLIP text encoder to ONNX (creates models/onnx/clip_vit_h14_text_fp32.onnx)                       normalized vector                  |   Optional: async LLM explanation    |                           ▼

make onnx-convert

                                  |

# Or with FP16 quantization (half the model size)

python -m scripts.convert_to_onnx --output-dir models/onnx --validate --fp16                                  v                  +--------------------------------------+┌──────────────────────────────────────────────────────────────────┐

```

                    +-------------+-------------+

### Enable ONNX Backend

                    |   Qdrant Vector Search     |```│             (Optional) LLM Explanation Service                   │

```bash

# Via environment variable                    |   (HNSW, all in RAM)       |

export USE_ONNX=true

export ONNX_MODEL_PATH=models/onnx/clip_vit_h14_text_fp32.onnx                    |                           |│  • Async, non-blocking — never delays search results             │



# In Docker Compose, set in docker-compose.yml:                    |  - gRPC transport         |

# USE_ONNX: "true"

```                    |  - int8 scalar            |### Design Principles│  • OpenAI GPT-4 via async client                                 │



### Benchmark                    |    quantization           |



```bash                    |  - m=16,                  |│  • Disabled by default                                           │

make onnx-benchmark

```                    |    ef_construct=200,      |



The factory pattern in `services/embedding_service/__init__.py` handles backend selection:                    |    ef_search=128          |- **No image embedding at query time.** Images are pre-encoded offline. Only text is encoded on the hot path.└──────────────────────────────────────────────────────────────────┘

- If `USE_ONNX=true` and the ONNX model file exists, uses `ONNXCLIPEmbedder`

- Otherwise falls back to `CLIPEmbedder` (PyTorch) with a warning                    |  - ~2-10ms per query      |



### ONNX Tuning Options                    +-------------+-------------+- **No disk reads during retrieval.** All vectors and payloads reside in RAM.```



| Variable | Default | Description |                                  |

|----------|---------|-------------|

| `USE_ONNX` | `false` | Enable ONNX Runtime backend |                       top-K results + metadata- **CLIP model is warm-loaded.** A single forward pass runs during startup to trigger CUDA kernel compilation and memory allocation. No request ever hits a cold model.

| `ONNX_MODEL_PATH` | `models/onnx/clip_vit_h14_text_fp32.onnx` | Path to ONNX model |

| `ONNX_PROVIDERS` | `CPUExecutionProvider` | Execution providers (comma-separated) |                                  |

| `ONNX_INTRA_OP_THREADS` | `4` | Threads for intra-operator parallelism |

| `ONNX_INTER_OP_THREADS` | `2` | Threads for inter-operator parallelism |                                  v- **In-process service calls.** The embedding and retrieval services are function calls within the same process, not network hops. This eliminates ~2-5ms of serialization overhead per request.### Key Design Decisions

| `ONNX_EXECUTION_MODE` | `parallel` | ORT execution mode |

| `USE_FP16` | `false` | Use FP16 quantized model |                    +-------------+-------------+



---                    |   Response Assembly        |- **Single worker per container.** The CLIP model lives in GPU memory. Multiple workers would duplicate the model and risk OOM. Horizontal scaling is achieved through container replicas behind a load balancer.



## Kubernetes Deployment                    |                           |



The system includes complete Kubernetes manifests using Kustomize with environment overlays.                    |  - Cache result in Redis  |- **Cache is fail-open.** Redis unavailability degrades to live search with no error propagation. Search never fails because of cache issues.| Decision | Rationale |



### Structure                    |  - Optional: async LLM    |



```                    |    explanation (GPT-4)    ||----------|-----------|

k8s/

├── base/           # Shared resources (namespace, configmap, secrets)                    |  - Return JSON response   |

├── api/            # API Deployment + Service + HPA + PDB

├── qdrant/         # StatefulSet + Service (persistent storage)                    +---------------------------+---| **Single process, single worker** | CLIP model lives in GPU memory. Multiple workers = multiple model copies = OOM. Scale horizontally with containers. |

├── redis/          # Deployment + Service (ephemeral cache)

├── ingress/        # Ingress + TLS certificate```

├── monitoring/     # OTel Collector + Prometheus + Grafana

└── overlays/| **In-process service calls** | No HTTP/gRPC between embedding → retrieval. Saves ~2-5ms per request. Service boundaries are in code, not network. |

    ├── dev/        # 1 replica, no auth, no monitoring

    ├── staging/    # 2 replicas, auth enabled, 50% sampling### Design Decisions

    └── prod/       # 3+ replicas, ONNX enabled, full observability

```## System Components| **gRPC for Qdrant** | ~2x faster than REST for vector search operations. |



### Deploy| Decision | Rationale |



```bash|---|---|| **Int8 scalar quantization** | Reduces vector memory ~4x (4GB → 1GB for 1M vectors) with <1% recall loss. |

# Dev environment

kubectl apply -k k8s/overlays/dev| Single process, single worker | CLIP model resides in GPU memory. Multiple workers duplicate the model and risk out-of-memory failures. Horizontal scaling is achieved through container replicas. |



# Staging| In-process service calls | Embedding and retrieval are function calls within the same process. No HTTP or gRPC between internal services. This eliminates 2-5ms of serialization and network overhead per request. |### 1. API Gateway| **Redis cache is fail-open** | Cache miss = live search. Cache error = live search. Search NEVER fails because of cache. |

kubectl apply -k k8s/overlays/staging

| gRPC for Qdrant communication | gRPC is approximately 2x faster than REST for vector search operations. The Qdrant client uses gRPC by default. |

# Production (includes monitoring stack)

kubectl apply -k k8s/overlays/prod| int8 scalar quantization | Reduces vector memory usage by approximately 4x (from 4GB to 1GB for one million vectors) with less than 1% recall degradation. || **No image embedding at query time** | Images are pre-encoded offline. Only text is encoded on the hot path. |



# Check status| Fail-open Redis cache | Cache miss falls through to live search. Cache error falls through to live search. The search path never fails due to cache unavailability. |

kubectl -n rag-system get pods

kubectl -n rag-system get hpa| No image embedding at query time | All images are pre-encoded during the offline indexing phase. Only the text encoder runs on the hot path. |FastAPI application serving as the single entry point. Handles input validation, cache lookup, response assembly, and latency instrumentation.

```

| Warm model loading | A dummy forward pass executes during startup to pre-compile CUDA kernels and allocate memory. No request ever hits a cold model. |

### Key Design Decisions

---

- **API Deployment**: 1 worker per pod, scale horizontally via HPA (2-10 replicas)

- **Qdrant StatefulSet**: Persistent storage, stable network identity for gRPC---

- **Redis Deployment**: Ephemeral (no persistence), cache is fail-open

- **HPA**: Scales on CPU (70%) and memory (80%) utilization**Endpoints:**

- **PDB**: Minimum 1 pod available during voluntary disruptions

- **Startup probe**: 5-minute grace period for CLIP model download## Request Lifecycle



### Overlay Differences## 📁 Project Structure



| Setting | Dev | Staging | Production |A search request follows this path:

|---------|-----|---------|------------|

| API replicas | 1 | 2 | 3 || Method | Path | Description |

| HPA max | 2 | 4 | 10 |

| ONNX | off | off | on |1. **Client** sends `POST /search` with a JSON body containing the query text, top-K count, and optional filters.

| OTel | off | on (50% sampling) | on (10% sampling) |

| Auth | off | on | on ||--------|------|-------------|```

| Rate limiting | off | on (200/min) | on (100/min) |

2. **Input Validation** -- Pydantic v2 validates and parses the request body. Invalid requests are rejected with a 422 status code before any computation occurs.

---

| `POST` | `/search` | Text-to-image semantic search |rag-image-search/

## Observability

3. **Cache Lookup** -- The gateway computes a deterministic cache key from the query, top-K, and filters. If Redis contains a matching entry, the cached response is returned immediately (approximately 0.3ms).

### OpenTelemetry

| `GET` | `/health` | Liveness and readiness probe |├── configs/

The system uses OpenTelemetry for distributed tracing and metrics collection.

4. **Text Encoding** -- On cache miss, the query string is tokenized and passed through the CLIP text encoder. The output is an L2-normalized 1024-dimensional float32 vector. This step takes 3-8ms on GPU or 20-50ms on CPU.

```bash

# Enable in environment| `GET` | `/stats` | Runtime latency percentiles and collection metadata |│   └── settings.py            # Pydantic settings from .env

export OTEL_ENABLED=true

export OTEL_ENDPOINT=http://localhost:43175. **Vector Search** -- The normalized query vector is sent to Qdrant via gRPC. Qdrant performs HNSW approximate nearest neighbor search with ef=128 against the in-memory index. This returns the top-K point IDs, cosine similarity scores, and stored metadata. This step takes 2-10ms.



# Start with monitoring profile├── services/

docker compose --profile monitoring up -d

```6. **Response Assembly** -- Results are formatted into the response schema with latency instrumentation. The response is cached in Redis with a configurable TTL (default 3600 seconds).



**Traces** propagate through the full request path:**Search Request Schema:**│   ├── api_gateway/

- HTTP request span (auto-instrumented by FastAPI middleware)

- Embedding span (CLIP text encode with backend label)7. **Optional LLM Explanation** -- If requested and enabled, an async call to the OpenAI API generates a brief natural language explanation of why the results match the query. This never blocks the search response.

- Retrieval span (Qdrant search with top_k label)

- Cache spans (hit/miss counters)│   │   ├── app.py             # FastAPI application + endpoints



**Metrics** exported to Prometheus:8. **Response** -- The JSON response is returned to the client with the results, total count, and measured latency.

- `rag_search_latency` — End-to-end search latency histogram

- `rag_embedding_latency` — CLIP encoding latency by backend| Field | Type | Default | Constraints |│   │   ├── models.py          # Request/Response schemas

- `rag_retrieval_latency` — Qdrant search latency

- `rag_cache_hits_total` / `rag_cache_misses_total` — Cache effectiveness---

- `rag_search_requests_total` — Total request count

- `rag_errors_total` — Error count by type|-------|------|---------|-------------|│   │   └── cache.py           # Redis caching layer



### Grafana Dashboard## System Components



Access at `http://localhost:3000` (admin/admin). Pre-configured panels:| `query` | `string` | required | 1-512 characters |│   ├── embedding_service/



- Search request rate (req/s)### 1. API Gateway

- Cache hit rate gauge

- Error rate| `top_k` | `integer` | 10 | 1-100 |│   │   └── embedder.py        # CLIP text/image encoder (warm-loaded)

- Search latency percentiles (P50/P95/P99)

- Embedding latency by backend (PyTorch vs ONNX)The FastAPI application is the single entry point for all client traffic. It handles input validation, cache interaction, response assembly, and latency instrumentation.

- Qdrant search latency

- Memory usage per pod| `score_threshold` | `float` | 0.0 | 0.0-1.0 |│   ├── retrieval_service/



### Prometheus Alerts**Endpoints:**



Pre-configured alerts in `k8s/monitoring/prometheus.yaml`:| `filters` | `object` | null | Metadata key-value filter |│   │   └── retriever.py       # Qdrant HNSW search client



| Alert | Condition | Severity || Method | Path | Description |

|-------|-----------|----------|

| HighSearchLatency | P95 > 200ms for 5min | warning ||---|---|---|| `include_explanation` | `boolean` | false | Triggers optional LLM path |│   └── llm_service/

| HighErrorRate | > 5% errors for 5min | critical |

| PodCrashLoop | > 3 restarts in 10min | critical || POST | /search | Text-to-image semantic search |

| QdrantDown | Unreachable for 2min | critical |

| RedisCacheDown | Unreachable for 5min | warning || GET | /health | Liveness and readiness probe for load balancers |│       └── llm.py             # Optional async LLM explanations



### Jaeger Tracing| GET | /stats | Runtime latency percentiles and collection metadata |



Access at `http://localhost:16686`. View distributed traces for individual requests including span timing breakdowns.**Search Response Schema:**├── indexing/



---**Responsibilities:**



## Authentication and Rate Limiting- Validate inbound requests using Pydantic v2 schemas│   └── index_images.py        # Offline image indexing pipeline



Both are optional and disabled by default. Enable for production deployments.- Check Redis for cached responses before invoking compute



### API Key Authentication- Coordinate the embedding and retrieval services| Field | Type | Description |├── scripts/



```bash- Record latency metrics for every request

# Enable and set valid keys

export AUTH_ENABLED=true- Cache responses for repeated queries|-------|------|-------------|│   ├── simulate_dataset.py    # Generate 1M synthetic vectors

export API_KEYS="key-1-abc,key-2-def,key-3-ghi"

```- Optionally invoke the LLM service for explanations



Clients must include the `X-API-Key` header:| `query` | `string` | Echo of the input query |│   └── benchmark.py           # Latency benchmarking tool



```bash### 2. Embedding Service

curl -X POST http://localhost:8000/search \

  -H "Content-Type: application/json" \| `results` | `array` | List of `{id, score, metadata}` objects |├── infra/

  -H "X-API-Key: key-1-abc" \

  -d '{"query": "sunset over ocean", "top_k": 5}'A singleton CLIP text encoder that remains loaded in GPU or CPU memory for the entire process lifetime.

```

| `total` | `integer` | Number of results returned |│   └── docker-compose.yml     # Qdrant + Redis local stack

Exempt paths: `/health`, `/docs`, `/openapi.json`

**Characteristics:**

### Rate Limiting

- Model architecture: OpenCLIP ViT-H-14| `latency_ms` | `float` | End-to-end server-side latency |├── utils/

```bash

export RATE_LIMIT_ENABLED=true- Pretrained weights: laion2b_s32b_b79k

export RATE_LIMIT_REQUESTS=100      # Max requests per window

export RATE_LIMIT_WINDOW_SECONDS=60 # Window size- Output dimensionality: 1024| `cached` | `boolean` | Whether the response was served from cache |│   ├── logger.py              # Structured logging (structlog)

```

- Output normalization: L2-normalized to unit sphere (cosine similarity reduces to dot product)

Rate limiting is per API key (if auth enabled) or per IP address. Returns HTTP 429 when exceeded.

- Inference optimization: `torch.compile()` with `reduce-overhead` mode, `torch.amp.autocast` for mixed precision| `explanation` | `string` | LLM-generated explanation (if requested) |│   ├── timing.py              # High-precision latency measurement

---

- Thread safety: PyTorch inference under `torch.no_grad()` is safe for concurrent reads

## CI/CD Pipeline

- Warmup: a dummy forward pass runs at initialization to pre-compile CUDA kernels and pre-allocate memory│   └── metrics.py             # In-process P50/P95/P99 tracker

GitHub Actions pipeline at `.github/workflows/ci.yaml`:



```

Push to main:  Lint -> Test -> Build Docker Image -> Deploy StagingThe model is never reinitialized. It is created once during application startup and reused for every request.### 2. Embedding Service├── .env.example               # Environment variable template

Pull Request:  Lint -> Test

Manual:        Deploy to dev/staging/prod (workflow_dispatch)

```

### 3. Retrieval Service├── requirements.txt           # Pinned Python dependencies

### Pipeline Stages



1. **Lint**: ruff + mypy type checking

2. **Test**: pytest with coverage, Redis service containerA Qdrant client wrapper configured for HNSW approximate nearest neighbor search over gRPC.Singleton CLIP text encoder that remains loaded in GPU/CPU memory for the entire process lifetime.└── Makefile                   # Common development commands

3. **Build**: Multi-platform Docker image, pushed to Docker Hub

4. **Deploy**: Kustomize apply to Kubernetes cluster



### Required Secrets**Collection configuration:**```



| Secret | Description |- Vector dimensionality: 1024

|--------|-------------|

| `DOCKERHUB_USERNAME` | Docker Hub username |- Distance metric: cosine similarity**Key properties:**

| `DOCKERHUB_TOKEN` | Docker Hub access token |

| `KUBE_CONFIG` | Base64-encoded kubeconfig |- Index type: HNSW (m=16, ef_construct=200)



---- Search-time parameter: ef=128- Model: OpenCLIP ViT-H-14 with laion2b_s32b_b79k weights---



## API Reference- Storage mode: all vectors and payloads in RAM (on_disk=false)



### POST /search- Quantization: scalar int8 (quantile=0.99, always_ram=true)- Output: L2-normalized 1024-dimensional float32 vector



Text-to-image semantic search.



**Request:**The client maintains a persistent gRPC connection pool. Collection existence is verified at startup. Batch upsert is supported for the indexing pipeline.- Optimization: `torch.compile()` with `reduce-overhead` mode on GPU, `torch.amp.autocast` for mixed precision inference## 🚀 Quick Start

```json

{

  "query": "a dog playing in the park",

  "top_k": 10,### 4. LLM Service (Optional)- Thread safety: PyTorch inference under `torch.no_grad()` is safe for concurrent reads

  "score_threshold": 0.2,

  "filters": {"category": "nature"},

  "include_explanation": false

}An asynchronous, non-blocking explanation generator. When enabled, it accepts the top search results and generates a brief natural language summary of why those images match the query.- Warmup: A dummy forward pass runs at initialization to pre-compile CUDA kernels### Prerequisites

```



**Response:**

```json- Disabled by default (LLM_ENABLED=false)

{

  "query": "a dog playing in the park",- Uses the OpenAI async client

  "results": [

    {- Never blocks the retrieval path### 3. Retrieval Service- Python 3.10+

      "id": "abc123",

      "score": 0.8742,- Results are capped at the top 5 for prompt construction to limit token usage

      "metadata": {

        "filename": "dog_park.jpg",- Responses are generated with low temperature (0.3) for factual consistency- Docker & Docker Compose

        "category": "nature"

      }

    }

  ],### 5. Indexing Pipeline (Offline)Qdrant client wrapper using gRPC for vector search operations.- (Optional) NVIDIA GPU with CUDA for fast CLIP inference

  "total": 10,

  "latency_ms": 23.45,

  "cached": false

}A standalone CLI script that runs independently of the API server. It processes images in batches through the CLIP vision encoder and upserts the resulting vectors into Qdrant.

```



### GET /health

- Scans directories recursively for image files (JPEG, PNG, WebP, BMP, TIFF)**Collection configuration:**### 1. Clone and Setup

Liveness and readiness probe.

- Encodes images through CLIP's vision encoder in configurable batch sizes (default 256)

```json

{- Generates deterministic integer IDs via MD5 hash of file path for idempotent re-runs- Vector size: 1024 dimensions

  "status": "healthy",

  "clip_loaded": true,- Upserts vectors with metadata (file path, file name, dimensions, simulated S3 key)

  "qdrant_connected": true,

  "redis_connected": true,- Writes a JSON manifest for auditing and debugging- Distance metric: Cosine similarity```bash

  "device": "cpu"

}- Supports GPU acceleration with progress logging via tqdm

```

- Index type: HNSW with `m=16`, `ef_construct=200`cd rag-image-search

### GET /stats

### 6. Dataset Simulator

Runtime performance metrics with latency percentiles.

- Search parameter: `ef=128` (tunable per-query)cp .env.example .env

---

A script that generates random unit-norm vectors matching the CLIP embedding distribution for load testing and benchmarking without requiring real images or the CLIP model.

## Configuration Reference

- Storage: All vectors and payloads in RAM (`on_disk=false`)

All configuration is via environment variables. See `.env.example` for the complete list.

- Default configuration: 1,000,000 vectors in batches of 10,000

### Core Settings

- Vectors are drawn from N(0,1) and L2-normalized to the unit hypersphere- Quantization: Scalar int8 (`quantile=0.99`, `always_ram=true`), reducing memory ~4x with less than 1% recall loss# Create and activate virtual environment

| Variable | Default | Description |

|----------|---------|-------------|- Includes synthetic metadata with eight categories for filter testing

| `CLIP_MODEL_NAME` | `ViT-H-14` | OpenCLIP model |

| `CLIP_PRETRAINED` | `laion2b_s32b_b79k` | Pretrained weights |- Indexing time: approximately 5-10 minutes for one million vectorspython3 -m venv venv

| `CLIP_VECTOR_DIM` | `1024` | Embedding dimension |

| `FORCE_CPU` | `false` | Disable GPU |

| `QDRANT_HOST` | `localhost` | Qdrant hostname |

| `QDRANT_PORT` | `6333` | Qdrant REST port |### 7. Benchmark Tool**Memory estimates:**source venv/bin/activate

| `QDRANT_GRPC_PORT` | `6334` | Qdrant gRPC port |

| `REDIS_HOST` | `localhost` | Redis hostname |

| `REDIS_ENABLED` | `true` | Enable Redis cache |

| `API_PORT` | `8000` | API server port |An end-to-end latency benchmarking script that sends configurable numbers of search requests to the running API and reports percentile statistics.

| `API_WORKERS` | `1` | Uvicorn workers (keep at 1) |



### Feature Flags

- Cycles through 15 predefined natural language queries| Scale | Raw Vectors | With int8 Quantization | Total (incl. HNSW graph) |# Install dependencies

| Variable | Default | Description |

|----------|---------|-------------|- Reports P50, P90, P95, P99, min, max, mean, and standard deviation

| `USE_ONNX` | `false` | ONNX Runtime backend |

| `OTEL_ENABLED` | `false` | OpenTelemetry instrumentation |- Validates results against the 50ms P95 target|-------|-------------|------------------------|--------------------------|make install

| `AUTH_ENABLED` | `false` | API key authentication |

| `RATE_LIMIT_ENABLED` | `false` | Per-key rate limiting |- Supports configurable request count and top-K

| `LLM_ENABLED` | `false` | GPT-4 explanations |

| 1M images | ~4 GB | ~1 GB | ~2 GB |```

---

### 8. Cache Layer

## Performance Targets

| 10M images | ~40 GB | ~10 GB | ~20 GB |

| Metric | Target | Typical |

|--------|--------|---------|A Redis-backed caching layer for repeated queries.

| End-to-end latency | < 50ms | 20-35ms |

| CLIP text encode (PyTorch, CPU) | < 50ms | 20-40ms || 100M images | Requires distributed mode | ~100 GB | ~200 GB |### 2. Start Infrastructure

| CLIP text encode (ONNX, CPU) | < 20ms | 8-15ms |

| Qdrant HNSW search | < 10ms | 2-5ms |- Cache key: MD5 hash of (query + top_k + filters)

| Redis cache hit | < 1ms | 0.3-0.5ms |

| Recall@10 (HNSW ef=128) | > 95% | 97% |- Default TTL: 3600 seconds

| Throughput (single pod, CPU) | > 50 QPS | 60-80 QPS |

- Fail-open design: all cache errors are silently swallowed

---

- Connection pooling: up to 20 persistent connections### 4. LLM Service (Optional)```bash

## Scaling Strategy

- Serialization: JSON

1. **Horizontal pod scaling**: Each pod runs one CLIP model instance. HPA scales 2-10 pods based on CPU/memory

2. **Never scale workers**: Keep `API_WORKERS=1`. Multiple workers would each load a 3GB modelmake infra

3. **ONNX for CPU throughput**: Switch to ONNX Runtime for 2-3x faster encoding on CPU nodes

4. **Cache for repeated queries**: Redis absorbs popular queries at sub-millisecond latency---

5. **Qdrant quantization**: int8 scalar quantization reduces memory 4x with < 1% recall loss

6. **Precomputed embeddings**: Images are embedded offline; only text encoding happens at query timeAsync, non-blocking explanation generator. Accepts retrieved metadata and produces a brief natural language summary of why the results match the query.# Starts Qdrant (:6333/:6334) and Redis (:6379)



---## Technology Stack



## License```



MIT| Component | Technology | Version | Purpose |


|---|---|---|---|- Disabled by default (`LLM_ENABLED=false`)

| Language | Python | 3.10+ | Core application language |

| API Framework | FastAPI | 0.115.6 | HTTP server with async support and OpenAPI docs |- Uses OpenAI async client### 3. Index Data

| ASGI Server | Uvicorn | 0.34.0 | High-performance ASGI server |

| ML Framework | PyTorch | 2.1+ | Tensor operations and GPU acceleration |- Never blocks the retrieval path -- explanation is appended after results are assembled

| CLIP Model | OpenCLIP | 2.29.0 | Text and image embedding generation |

| Vector Database | Qdrant | 1.13.2 | HNSW index with gRPC and in-memory storage |- Results are capped at top 5 for prompt construction to limit token usage**Option A: Simulate 1M vectors** (no CLIP model needed, fast)

| Cache | Redis | 7.x (Alpine) | Query result caching with LRU eviction |

| Validation | Pydantic | 2.10.4 | Request/response schema validation |```bash

| Logging | structlog | 24.x | Structured JSON/console logging |

| Containerization | Docker Compose | v2 | Multi-service orchestration |### 5. Indexing Pipeline (Offline)make simulate

| Image Processing | Pillow | 10.0+ | Image loading and preprocessing |

# Generates 1M random unit-norm vectors → Qdrant

---

Batch image encoding script that runs independently of the API server.# Takes ~5-10 minutes depending on machine

## Project Structure

```

```

.- Scans a directory recursively for image files (JPEG, PNG, WebP, BMP, TIFF)

|-- configs/

|   |-- __init__.py- Encodes images through CLIP's vision encoder in configurable batch sizes**Option B: Index real images** (requires CLIP model download)

|   |-- settings.py                # Pydantic settings loaded from environment variables

|- Generates deterministic integer IDs via MD5 hash of file path (idempotent re-runs)```bash

|-- services/

|   |-- __init__.py- Upserts vectors and metadata into Qdrant# Put images in ./data/images/

|   |-- api_gateway/

|   |   |-- __init__.py- Writes a JSON manifest for auditingmake index

|   |   |-- app.py                 # FastAPI application, endpoints, lifespan hooks

|   |   |-- models.py             # Pydantic request and response schemas- Supports GPU acceleration and progress logging```

|   |   |-- cache.py              # Redis caching layer with fail-open design

|   |

|   |-- embedding_service/

|   |   |-- __init__.py### 6. Dataset Simulator### 4. Start the API Server

|   |   |-- embedder.py           # CLIP text and image encoder singleton

|   |

|   |-- retrieval_service/

|   |   |-- __init__.pyGenerates random unit-norm vectors matching the CLIP embedding distribution for load testing and benchmarking without requiring real images or the CLIP model.```bash

|   |   |-- retriever.py          # Qdrant HNSW search client with gRPC

|   |make serve

|   |-- llm_service/

|       |-- __init__.py- Default: 1,000,000 vectors in batches of 10,000# Server starts on http://localhost:8000

|       |-- llm.py                # Optional async LLM explanation generator

|- Includes synthetic metadata with categories for filter testing# CLIP model loads and warms up (~4s)

|-- indexing/

|   |-- __init__.py- Vectors are drawn from N(0,1) and L2-normalized to the unit hypersphere```

|   |-- index_images.py           # Offline batch image indexing pipeline

|

|-- scripts/

|   |-- __init__.py---### 5. Search!

|   |-- simulate_dataset.py       # Synthetic 1M vector generation

|   |-- benchmark.py              # End-to-end latency benchmark

|

|-- utils/## Performance Targets```bash

|   |-- __init__.py

|   |-- logger.py                 # Structured logging configuration (structlog)# Quick test

|   |-- timing.py                 # Nanosecond-precision timing context manager

|   |-- metrics.py                # In-process P50/P95/P99 latency tracker| Metric | Target | Typical (GPU) | Typical (CPU) |make test-search

|

|-- docker-compose.yml            # Full stack: app + qdrant + redis|--------|--------|---------------|---------------|

|-- Dockerfile                    # Multi-stage build for the API service

|-- Makefile                      # Development task runner| CLIP text encode | < 10 ms | 3-8 ms | 20-50 ms |# Or use curl directly

|-- requirements.txt              # Pinned Python dependencies

|-- .env.example                  # Environment variable reference| Qdrant HNSW search | < 10 ms | 2-5 ms | 2-5 ms |curl -X POST http://localhost:8000/search \

|-- .dockerignore                 # Docker build context exclusions

|-- .gitignore                    # Git exclusions| Redis cache hit | < 1 ms | 0.3 ms | 0.3 ms |  -H "Content-Type: application/json" \

```

| **Total end-to-end (excl. LLM)** | **< 50 ms** | **~12 ms** | **~35 ms** |  -d '{"query": "a sunset over the ocean", "top_k": 5}'

---

| P95 latency | < 50 ms | ~18 ms | ~45 ms |```

## API Reference

| P99 latency | < 100 ms | ~28 ms | ~60 ms |

### POST /search

### 6. Benchmark

Perform a text-to-image semantic search.

All latency metrics are tracked in-process via a rolling window of 10,000 observations and exposed through the `/stats` endpoint with P50, P95, and P99 percentiles.

**Request body:**

```bash

| Field | Type | Required | Default | Constraints | Description |

|---|---|---|---|---|---|---python -m scripts.benchmark --num-requests 1000 --top-k 10

| query | string | yes | -- | 1-512 characters | Natural language search query |

| top_k | integer | no | 10 | 1-100 | Number of results to return |```

| score_threshold | float | no | 0.0 | 0.0-1.0 | Minimum cosine similarity score |

| filters | object | no | null | -- | Metadata key-value filter pairs |## Infrastructure

| include_explanation | boolean | no | false | -- | Request LLM-generated explanation |

---

**Example request:**

### Docker Compose Services

```json

{## 📡 API Reference

  "query": "a dog playing in the snow",

  "top_k": 10,| Service | Image | Ports | Purpose |

  "score_threshold": 0.2,

  "filters": {"category": "animals"},|---------|-------|-------|---------|### `POST /search`

  "include_explanation": false

}| `app` | Custom (Dockerfile) | 8000 | FastAPI + CLIP inference |

```

| `qdrant` | qdrant/qdrant:v1.13.2 | 6333, 6334 | Vector DB (REST + gRPC) |Search for images by natural language query.

**Response body:**

| `redis` | redis:7-alpine | 6379 | Query result cache |

| Field | Type | Description |

|---|---|---|**Request:**

| query | string | Echo of the input query |

| results | array | List of result objects (see below) |### Dockerfile```json

| total | integer | Number of results returned |

| latency_ms | float | End-to-end server-side latency in milliseconds |{

| cached | boolean | Whether the response was served from Redis cache |

| explanation | string or null | LLM-generated explanation if requested and enabled |Multi-stage build:  "query": "a dog playing in the snow",



**Result object:**1. **deps** stage: Installs Python packages into a cached layer. Rebuilds only when `requirements.txt` changes.  "top_k": 10,



| Field | Type | Description |2. **runtime** stage: Copies installed packages and application code. Thin layer for fast iteration.  "score_threshold": 0.2,

|---|---|---|

| id | string | Unique vector point ID in Qdrant |  "filters": {"category": "animals"},

| score | float | Cosine similarity score (0.0 to 1.0) |

| metadata | object | Stored payload: file_path, file_name, s3_key, width, height, category |The container runs a single uvicorn worker. Health checks poll `/health` every 30 seconds with a 60-second startup grace period (CLIP model loading).  "include_explanation": false



**Example response:**}



```json### Resource Limits```

{

  "query": "a dog playing in the snow",

  "results": [

    {| Service | Memory Limit | Notes |**Response:**

      "id": "8372651",

      "score": 0.8234,|---------|-------------|-------|```json

      "metadata": {

        "file_path": "/data/images/img_00123.jpg",| app | 4 GB | CLIP model (~2 GB) + PyTorch overhead |{

        "file_name": "img_00123.jpg",

        "s3_key": "s3://image-bucket/images/img_00123.jpg",| qdrant | 6 GB | 1M vectors quantized + HNSW graph + payloads |  "query": "a dog playing in the snow",

        "category": "animals",

        "width": 640,| redis | 512 MB | LRU eviction, no persistence |  "results": [

        "height": 480

      }    {

    }

  ],---      "id": "8372651",

  "total": 10,

  "latency_ms": 12.45,      "score": 0.8234,

  "cached": false,

  "explanation": null## Project Structure      "metadata": {

}

```        "file_path": "/data/images/img_00123.jpg",



### GET /health```        "file_name": "img_00123.jpg",



Returns component-level health status. Used by Docker health checks and load balancers..        "s3_key": "s3://image-bucket/images/img_00123.jpg",



**Response body:**+-- configs/        "category": "animals",



| Field | Type | Description ||   +-- settings.py              # Pydantic settings from environment        "width": 640,

|---|---|---|

| status | string | "healthy" if CLIP and Qdrant are operational, "degraded" otherwise |+-- services/        "height": 480

| clip_loaded | boolean | Whether the CLIP model is loaded and ready |

| qdrant_connected | boolean | Whether the Qdrant client can reach the server ||   +-- api_gateway/      }

| redis_connected | boolean | Whether Redis is reachable |

| device | string | Inference device: "cuda" or "cpu" ||   |   +-- app.py               # FastAPI application and endpoints    }



### GET /stats|   |   +-- models.py            # Request/response Pydantic schemas  ],



Returns runtime performance metrics with latency percentiles.|   |   +-- cache.py             # Redis caching layer (fail-open)  "total": 10,



**Response body:**|   +-- embedding_service/  "latency_ms": 12.45,



| Field | Type | Description ||   |   +-- embedder.py          # CLIP text/image encoder singleton  "cached": false,

|---|---|---|

| metrics | object | Contains uptime_seconds and per-metric percentile breakdowns ||   +-- retrieval_service/  "explanation": null

| collection | object | Qdrant collection metadata: vectors_count, status, indexed |

|   |   +-- retriever.py         # Qdrant HNSW search client}

Each metric object (embedding_latency_ms, search_latency_ms, total_latency_ms) contains:

|   +-- llm_service/```

| Field | Type | Description |

|---|---|---||       +-- llm.py               # Optional async LLM explanations

| count | integer | Total observations since startup |

| window | integer | Number of observations in the rolling window (max 10,000) |+-- indexing/### `GET /health`

| p50_ms | float | 50th percentile latency |

| p95_ms | float | 95th percentile latency (null if fewer than 20 observations) ||   +-- index_images.py          # Offline batch image indexing

| p99_ms | float | 99th percentile latency (null if fewer than 100 observations) |

| mean_ms | float | Arithmetic mean |+-- scripts/Health check for load balancers and monitoring.

| min_ms | float | Minimum observed latency |

| max_ms | float | Maximum observed latency ||   +-- simulate_dataset.py      # Synthetic vector generation (1M)



---|   +-- benchmark.py             # End-to-end latency benchmark```json



## Performance Targets+-- utils/{



| Metric | Target | Typical (GPU) | Typical (CPU) ||   +-- logger.py                # Structured logging (structlog)  "status": "healthy",

|---|---|---|---|

| CLIP text encode | under 10 ms | 3-8 ms | 20-50 ms ||   +-- timing.py                # Nanosecond-precision timing  "clip_loaded": true,

| Qdrant HNSW search | under 10 ms | 2-5 ms | 2-5 ms |

| Redis cache hit | under 1 ms | 0.3 ms | 0.3 ms ||   +-- metrics.py               # In-process percentile tracker  "qdrant_connected": true,

| Total end-to-end (excluding LLM) | under 50 ms | approximately 12 ms | approximately 35 ms |

| P95 latency | under 50 ms | approximately 18 ms | approximately 45 ms |+-- docker-compose.yml           # Full stack orchestration  "redis_connected": true,

| P99 latency | under 100 ms | approximately 28 ms | approximately 60 ms |

+-- Dockerfile                   # Multi-stage app container  "device": "cuda"

All latency metrics are tracked in-process using a rolling window of 10,000 observations. Percentiles are computed on demand and exposed through the `/stats` endpoint. No external metrics infrastructure (Prometheus, Datadog) is required for basic observability.

+-- Makefile                     # Development task runner}

---

+-- requirements.txt             # Pinned Python dependencies```

## CLIP Model Specification

+-- .env.example                 # Environment variable reference

| Property | Value |

|---|---|```### `GET /stats`

| Architecture | ViT-H-14 (Vision Transformer, Huge, patch size 14) |

| Weights | laion2b_s32b_b79k (trained on 2B image-text pairs) |

| Text encoder output | 1024-dimensional float32 vector |

| Image encoder output | 1024-dimensional float32 vector |---Runtime performance metrics with latency percentiles.

| Text context length | 77 tokens |

| Model size on disk | approximately 1 GB |

| GPU VRAM requirement | approximately 2 GB |

| Normalization | L2-normalized to unit sphere |## Configuration Reference```json



The text and image encoders share the same embedding space. Cosine similarity between a text vector and an image vector measures semantic relevance. Because all vectors are L2-normalized, cosine similarity is equivalent to dot product.{



The model is loaded once during the application lifespan startup hook. A warmup forward pass runs immediately after loading to trigger CUDA kernel JIT compilation and memory pool allocation. This ensures the first real request does not pay initialization overhead.All configuration is managed through environment variables. See `.env.example` for the complete reference.  "metrics": {



---    "uptime_seconds": 3600.0,



## Qdrant Configuration### CLIP Model    "embedding_latency_ms": {



| Parameter | Value | Rationale |      "count": 10000,

|---|---|---|

| Vector size | 1024 | Must match CLIP ViT-H-14 output dimension || Variable | Default | Description |      "p50_ms": 5.2,

| Distance metric | Cosine | Standard for normalized embeddings |

| HNSW m | 16 | Graph degree: balances recall, memory, and build time ||----------|---------|-------------|      "p95_ms": 8.1,

| HNSW ef_construct | 200 | Build-time beam width: higher values improve index quality |

| HNSW ef (search) | 128 | Search-time beam width: higher values improve recall at the cost of latency || `CLIP_MODEL_NAME` | `ViT-H-14` | OpenCLIP model architecture |      "p99_ms": 12.3,

| on_disk (vectors) | false | All vectors must reside in RAM for target latency |

| on_disk (HNSW) | false | HNSW graph must reside in RAM || `CLIP_PRETRAINED` | `laion2b_s32b_b79k` | Pretrained weights identifier |      "mean_ms": 5.8

| Quantization | Scalar int8 | Reduces memory approximately 4x with less than 1% recall loss |

| Quantization quantile | 0.99 | Calibration quantile for int8 range mapping || `CLIP_VECTOR_DIM` | `1024` | Output embedding dimension (must match model) |    },

| Quantization always_ram | true | Quantized vectors are never paged to disk |

| `FORCE_CPU` | `false` | Disable GPU even if CUDA is available |    "search_latency_ms": {

**Memory estimates at scale:**

      "count": 10000,

| Dataset Size | Raw Vector Memory | With int8 Quantization | Total (vectors + HNSW graph + payloads) |

|---|---|---|---|### Qdrant      "p50_ms": 3.1,

| 1 million images | approximately 4 GB | approximately 1 GB | approximately 2 GB |

| 10 million images | approximately 40 GB | approximately 10 GB | approximately 20 GB |      "p95_ms": 7.2,

| 100 million images | approximately 400 GB | approximately 100 GB | Requires distributed deployment |

| Variable | Default | Description |      "p99_ms": 11.0,

---

|----------|---------|-------------|      "mean_ms": 3.5

## Infrastructure

| `QDRANT_HOST` | `localhost` | Qdrant server hostname |    },

### Docker Compose Services

| `QDRANT_PORT` | `6333` | REST API port |    "total_latency_ms": {

The full stack runs as three containers orchestrated by Docker Compose:

| `QDRANT_GRPC_PORT` | `6334` | gRPC port (used for search) |      "count": 10000,

| Service | Container Name | Image | Ports | Memory Limit | Purpose |

|---|---|---|---|---|---|| `QDRANT_COLLECTION` | `image_vectors` | Collection name |      "p50_ms": 9.5,

| app | rag-api | Custom (Dockerfile) | 8000 | 4 GB | FastAPI server with CLIP model |

| qdrant | rag-qdrant | qdrant/qdrant:v1.13.2 | 6333, 6334 | 6 GB | Vector database (REST + gRPC) || `QDRANT_HNSW_M` | `16` | HNSW graph degree (higher = better recall, more RAM) |      "p95_ms": 18.3,

| redis | rag-redis | redis:7-alpine | 6379 | 512 MB | Query result cache |

| `QDRANT_HNSW_EF_CONSTRUCT` | `200` | HNSW build-time beam width |      "p99_ms": 28.1,

### Dockerfile

| `QDRANT_HNSW_EF` | `128` | HNSW search-time beam width |      "mean_ms": 10.2

The application image uses a multi-stage build:

    }

1. **deps stage** -- Installs system dependencies (build-essential, libgl1, libglib2.0) and Python packages from requirements.txt into a cached layer. This layer only rebuilds when dependencies change.

### Redis  },

2. **runtime stage** -- Copies installed packages and application code. Minimal system libraries for runtime. The final image does not contain build tools.

  "collection": {

The container runs a single uvicorn worker with access logging disabled (the application does its own structured logging). The health check polls `/health` every 30 seconds with a 60-second startup grace period to allow for CLIP model loading.

| Variable | Default | Description |    "collection": "image_vectors",

### Service Dependencies

|----------|---------|-------------|    "vectors_count": 1000000,

The `app` service depends on `qdrant` and `redis` with health check conditions. The application will not start until both backend services report healthy. Qdrant health is verified via its `/healthz` endpoint. Redis health is verified via `redis-cli ping`.

| `REDIS_HOST` | `localhost` | Redis server hostname |    "status": "GREEN",

### Redis Configuration

| `REDIS_PORT` | `6379` | Redis port |    "indexed": true

Redis is configured as an ephemeral LRU cache:

- Maximum memory: 512 MB| `REDIS_ENABLED` | `true` | Enable/disable caching |  }

- Eviction policy: allkeys-lru

- Persistence: disabled (no RDB snapshots, no AOF)| `REDIS_CACHE_TTL` | `3600` | Cache entry TTL in seconds |}

- Purpose: cache repeated search queries, not durable storage

```

---

### API Server

## Configuration Reference

---

All configuration is managed through environment variables. The `.env.example` file contains the complete reference with defaults.

| Variable | Default | Description |

### CLIP Model Settings

|----------|---------|-------------|## ⚡ Performance Targets

| Variable | Default | Description |

|---|---|---|| `API_HOST` | `0.0.0.0` | Bind address |

| CLIP_MODEL_NAME | ViT-H-14 | OpenCLIP model architecture identifier |

| CLIP_PRETRAINED | laion2b_s32b_b79k | Pretrained weights identifier || `API_PORT` | `8000` | Bind port || Metric | Target | Typical (GPU) | Typical (CPU) |

| CLIP_VECTOR_DIM | 1024 | Output embedding dimension (must match model) |

| FORCE_CPU | false | Force CPU inference even if CUDA is available || `API_WORKERS` | `1` | Uvicorn worker count (keep at 1) ||--------|--------|---------------|---------------|



### Qdrant Settings| CLIP text encode | < 10ms | 3-8ms | 20-50ms |



| Variable | Default | Description |---| Qdrant HNSW search | < 10ms | 2-5ms | 2-5ms |

|---|---|---|

| QDRANT_HOST | localhost | Qdrant server hostname || Redis cache hit | < 1ms | 0.3ms | 0.3ms |

| QDRANT_PORT | 6333 | Qdrant REST API port |

| QDRANT_GRPC_PORT | 6334 | Qdrant gRPC port (used for search operations) |## Scaling Considerations| **Total E2E (excl. LLM)** | **< 50ms** | **~12ms** | **~35ms** |

| QDRANT_COLLECTION | image_vectors | Name of the vector collection |

| QDRANT_HNSW_M | 16 | HNSW graph degree parameter || P95 latency | < 50ms | ~18ms | ~45ms |

| QDRANT_HNSW_EF_CONSTRUCT | 200 | HNSW index build-time beam width |

| QDRANT_HNSW_EF | 128 | HNSW search-time beam width |### Vertical Scaling (up to 10M images)| P99 latency | < 100ms | ~28ms | ~60ms |



### Redis Settings



| Variable | Default | Description |- Increase machine RAM. Qdrant handles 10M vectors on a single node with 32 GB RAM.---

|---|---|---|

| REDIS_HOST | localhost | Redis server hostname |- Increase `QDRANT_HNSW_M` to 32 for better recall at higher cardinality.

| REDIS_PORT | 6379 | Redis server port |

| REDIS_DB | 0 | Redis database number |- Use dedicated GPU with 8+ GB VRAM for faster CLIP inference.## 🧠 CLIP Model Details

| REDIS_ENABLED | true | Enable or disable the cache layer entirely |

| REDIS_CACHE_TTL | 3600 | Cache entry time-to-live in seconds |



### API Server Settings### Horizontal Scaling (10M-100M images)| Property | Value |



| Variable | Default | Description ||----------|-------|

|---|---|---|

| API_HOST | 0.0.0.0 | Server bind address |- Deploy Qdrant in cluster mode with sharding across nodes.| Model | ViT-H-14 (OpenCLIP) |

| API_PORT | 8000 | Server bind port |

| API_WORKERS | 1 | Uvicorn worker count (must remain 1; see Architecture) |- Add read replicas for search throughput.| Pretrained weights | laion2b_s32b_b79k |



### Indexing Settings- Run multiple API containers behind a load balancer, each with its own GPU.| Vector dimension | 1024 |



| Variable | Default | Description |- Use Redis Cluster for distributed caching.| Text context length | 77 tokens |

|---|---|---|

| IMAGE_DIR | ./data/images | Source directory for the image indexing pipeline || Model size | ~1GB |

| INDEX_BATCH_SIZE | 256 | Number of images per GPU encoding batch |

| INDEX_NUM_WORKERS | 4 | DataLoader worker count for parallel image loading |### Beyond 100M images| GPU VRAM required | ~2GB |



### LLM Settings (Optional)



| Variable | Default | Description |- Switch from scalar quantization to product quantization (PQ) for further memory reduction.The model is loaded once at startup and kept warm in memory. Every search

|---|---|---|

| LLM_ENABLED | false | Enable the LLM explanation service |- Consider IVFPQ indexing for billion-scale datasets (trades latency for memory).query only runs the text encoder (~3-8ms on GPU). Image encoding happens

| LLM_MODEL | gpt-4 | OpenAI model identifier |

| OPENAI_API_KEY | (empty) | OpenAI API key |- Shard the CLIP inference layer across multiple GPUs with model parallelism.offline during indexing.



### Search Defaults- Pre-warm cache with known high-traffic queries.



| Variable | Default | Description |---

|---|---|---|

| SEARCH_TOP_K | 10 | Default number of results per search query |---

| SEARCH_SCORE_THRESHOLD | 0.2 | Default minimum similarity score |

## 🔧 Configuration

---

## Technology Stack

## Scaling Strategy

All configuration is via environment variables. See `.env.example` for the complete list.

### Single Node -- Up to 10 Million Images

| Component | Technology | Version |

- Increase machine RAM to 32 GB or more. Qdrant handles 10 million vectors on a single node.

- Increase QDRANT_HNSW_M to 32 for better recall at higher cardinality.|-----------|-----------|---------|Key tuning parameters:

- Use a dedicated GPU with 8+ GB VRAM for faster CLIP text encoding.

- The API server remains a single worker per container. Deploy multiple containers behind a reverse proxy for throughput.| Language | Python | 3.10+ |



### Distributed -- 10 Million to 100 Million Images| API Framework | FastAPI | 0.115.6 || Variable | Default | Effect |



- Deploy Qdrant in cluster mode with sharding across multiple nodes.| ASGI Server | Uvicorn | 0.34.0 ||----------|---------|--------|

- Add Qdrant read replicas to increase search throughput.

- Run multiple API containers, each with its own GPU, behind NGINX or Envoy.| ML Framework | PyTorch | 2.1+ || `QDRANT_HNSW_M` | 16 | Higher = better recall, more RAM |

- Replace single-node Redis with Redis Cluster for distributed caching.

- Pre-warm cache with known high-traffic queries during deployment.| CLIP Implementation | OpenCLIP | 2.29.0 || `QDRANT_HNSW_EF_CONSTRUCT` | 200 | Higher = better index quality, slower build |



### Beyond 100 Million Images| Vector Database | Qdrant | 1.13.2 || `QDRANT_HNSW_EF` | 128 | Higher = better search recall, slower search |



- Switch from scalar int8 quantization to product quantization (PQ) for further memory reduction.| Cache | Redis | 7.x || `CLIP_VECTOR_DIM` | 1024 | Must match the model |

- Consider IVF-PQ indexing for billion-scale datasets, trading some latency for significantly lower memory usage.

- Shard the CLIP inference layer across multiple GPUs if text encoding becomes a bottleneck.| Serialization | Pydantic | 2.10.4 || `SEARCH_TOP_K` | 10 | Default results per query |

- Introduce a query routing layer that directs queries to the appropriate shard based on embedding space partitioning.

| Logging | structlog | 24.x || `REDIS_CACHE_TTL` | 3600 | Cache expiry in seconds |

---

| Containerization | Docker Compose | v2 |

## Limitations and Constraints

---

1. **Single-node deployment baseline.** The default configuration targets a single machine. The architecture supports horizontal scaling, but distributed deployment requires manual configuration of Qdrant cluster mode and a load balancer.

---

2. **CLIP is the only embedding model.** The embedding service is hardcoded to OpenCLIP ViT-H-14. Swapping to SigLIP, EVA-CLIP, or another model requires modifying the embedder and adjusting the vector dimension configuration. No model-agnostic abstraction layer exists.

## 📈 Scaling to 10M+ Images

3. **Images are not served by this system.** The API returns file paths and S3 keys. Actual image delivery is the responsibility of a CDN, object storage service, or static file server.

## Constraints and Assumptions

4. **Metadata storage is limited.** Image metadata is stored in Qdrant payloads, which is sufficient for lightweight data (path, dimensions, category). Complex metadata queries or joins require a relational database sidecar such as PostgreSQL.

### Current limits (single node)

5. **No authentication or authorization.** The API is completely open. API key validation, OAuth, or mTLS must be added before any non-local deployment.

1. **Single-node deployment baseline.** The architecture supports horizontal scaling but the default configuration targets a single machine.

6. **No rate limiting.** The API does not enforce request rate limits. In production, this should be handled by a reverse proxy or API gateway (NGINX, Kong, Envoy).

2. **CLIP is the sole embedding model.** The embedding service interface is simple enough to swap for SigLIP or EVA-CLIP, but no abstraction layer exists for model-agnostic operation.- **1M vectors**: ~2GB Qdrant RAM (with int8 quantization)

7. **LangChain and LlamaIndex are explicitly excluded.** All orchestration uses direct function calls. This is a deliberate decision to avoid abstraction overhead and maintain full control over the hot path.

3. **Images are stored externally.** The system returns file paths and S3 keys. Image serving is the responsibility of a CDN or object storage layer.- **10M vectors**: ~20GB Qdrant RAM

8. **Single worker constraint.** The CLIP model occupies GPU memory. Running multiple workers within the same container would duplicate the model and risk OOM. Scaling must be done through additional containers, not additional workers.

4. **Metadata is stored in Qdrant payloads.** Sufficient for lightweight metadata (path, dimensions, category). For rich metadata with complex queries, a relational sidecar (PostgreSQL) should be introduced.- **100M vectors**: Requires Qdrant distributed mode

9. **HNSW recall is approximate.** HNSW search is not exact nearest neighbor search. With the default parameters (m=16, ef=128), recall at top-10 exceeds 95% on typical distributions, but some relevant results may be missed compared to brute-force search.

5. **No authentication or authorization.** API security (API keys, OAuth, mTLS) must be added before any non-local deployment.

10. **Simulated dataset limitations.** The synthetic vector generator produces random unit-norm vectors that match the statistical distribution of CLIP embeddings but do not carry semantic meaning. Benchmark results with simulated data validate latency and throughput but not retrieval quality.

6. **LangChain and LlamaIndex are explicitly excluded.** All orchestration is direct function calls to avoid unnecessary abstraction overhead.### Scaling strategy

---



## License

---1. **Vertical (10M)**: Bigger machine, more RAM. Qdrant handles 10M vectors on a single node with 32GB RAM.

MIT



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
