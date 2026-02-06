# Multi-Modal RAG Image Search Engine

A production-grade, ultra-low latency text-to-image retrieval system. Accepts natural language queries like "a dog playing in the park" and returns the most semantically relevant images from a large corpus. Built on OpenCLIP for embedding generation and Qdrant with HNSW indexing for approximate nearest neighbor search. End-to-end retrieval latency is under 100ms on CPU and under 50ms on GPU.

---

## Table of Contents

- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [ONNX Runtime Acceleration](#onnx-runtime-acceleration)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Observability](#observability)
- [Authentication and Rate Limiting](#authentication-and-rate-limiting)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Performance Targets](#performance-targets)
- [Scaling Strategy](#scaling-strategy)

---

## Architecture

```
Client (POST /search)
        |
        v
+-------------------+     +----------+     +-----------+
|  FastAPI           |---->|  Redis   |     |  Qdrant   |
|  API Gateway       |     |  Cache   |     |  HNSW     |
|                    |     +----------+     +-----------+
|  CLIP ViT-B-32     |          ^                ^
|  (PyTorch / ONNX)  |         |                |
+-------------------+---------+----------------+
        |
        v (optional)
  OTel Collector -> Prometheus -> Grafana
                 -> Jaeger (traces)
```

**Request lifecycle (hot path):**

1. Check Redis cache (~0.3ms on hit)
2. CLIP text encode via PyTorch or ONNX Runtime (~80ms CPU / ~3-8ms GPU)
3. Qdrant HNSW search via gRPC (~2-10ms)
4. Assemble and return response (~0.1ms)

All services run in-process (function calls, not HTTP/gRPC between services). This eliminates 2-5ms of serialization overhead per request.

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | FastAPI 0.115 + Uvicorn | Async HTTP server |
| Embeddings | OpenCLIP ViT-B-32 (512-dim) | Text and image encoding |
| ONNX Runtime | onnxruntime 1.19.2 | 2-3x faster CPU inference |
| Vector DB | Qdrant 1.13+ (gRPC, HNSW, int8 quantization) | ANN search |
| Cache | Redis 7+ (LRU, fail-open) | Query result cache |
| Observability | OpenTelemetry + Prometheus + Grafana + Jaeger | Traces and metrics |
| Auth | API key middleware + slowapi rate limiting | Request authentication |
| Orchestration | Docker Compose (dev) / Kubernetes + Kustomize (prod) | Container management |
| CI/CD | GitHub Actions | Lint, test, build, deploy |
| Frontend | React 18 + Vite + Tailwind CSS | Chat-based search UI |

---

## Project Structure

```
.
├── configs/
│   └── settings.py              # Centralized Pydantic settings from env vars
├── services/
│   ├── api_gateway/
│   │   ├── app.py               # FastAPI application + lifespan + endpoints
│   │   ├── models.py            # Request/response Pydantic models
│   │   ├── cache.py             # Redis cache layer (fail-open)
│   │   ├── telemetry.py         # OpenTelemetry instrumentation
│   │   └── middleware/
│   │       └── __init__.py      # API key auth + rate limiting
│   ├── embedding_service/
│   │   ├── __init__.py          # Factory: auto-selects PyTorch or ONNX
│   │   ├── embedder.py          # PyTorch CLIP text/image encoder
│   │   └── onnx_embedder.py     # ONNX Runtime CLIP text encoder
│   ├── retrieval_service/
│   │   └── retriever.py         # Qdrant HNSW search + upsert
│   └── llm_service/
│       └── llm.py               # Optional GPT-4 explanations
├── indexing/
│   └── index_images.py          # Offline batch image indexing pipeline
├── scripts/
│   ├── convert_to_onnx.py       # Export CLIP text encoder to ONNX
│   ├── benchmark_onnx_vs_pytorch.py  # ONNX vs PyTorch benchmark
│   ├── benchmark.py             # Search latency benchmark
│   └── simulate_dataset.py      # Generate test vectors
├── frontend/
│   ├── src/                     # React app (chat UI, image search)
│   └── package.json             # Node.js dependencies
├── utils/
│   ├── logger.py                # Structured logging (structlog)
│   ├── metrics.py               # In-process latency percentiles
│   └── timing.py                # Precision timing context manager
├── models/onnx/                 # ONNX model files (git-ignored)
├── k8s/                         # Kubernetes manifests (Kustomize)
│   ├── base/                    # Namespace, ConfigMap, Secrets
│   ├── api/                     # Deployment, Service, HPA, PDB
│   ├── qdrant/                  # StatefulSet, Service, PVC
│   ├── redis/                   # Deployment, Service
│   ├── ingress/                 # Ingress + TLS certificate
│   ├── monitoring/              # OTel Collector, Prometheus, Grafana
│   └── overlays/                # dev / staging / prod Kustomize overlays
├── Dockerfile                   # Multi-stage production build
├── docker-compose.yml           # Full stack with monitoring profile
├── Makefile                     # Development task runner
└── requirements.txt             # Pinned Python dependencies
```

---

## Quick Start

### Docker Compose

```bash
# Start core stack (API + Qdrant + Redis)
docker compose up -d

# Start with full monitoring (adds OTel, Prometheus, Grafana, Jaeger)
docker compose --profile monitoring up -d

# Check health
curl http://localhost:8000/health

# Run a search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "chocolate cake", "top_k": 5}'
```

### Native (no Docker)

```bash
# Install Python dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Qdrant (binary)
./bin/qdrant --config-path ./qdrant_config.yaml &

# Start Redis
brew services start redis   # macOS
# or: redis-server &

# Index images
python -m indexing.index_images --image-dir ./data/food-101/images --batch-size 64

# Start the API server
python -m services.api_gateway.app
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## ONNX Runtime Acceleration

ONNX Runtime provides 2-3x faster CPU inference by replacing PyTorch's text encoder with an optimized ONNX graph. The system automatically selects the backend at startup.

### Convert the Model

```bash
# Export CLIP text encoder to ONNX
make onnx-convert

# Or with FP16 quantization (half the model size)
python -m scripts.convert_to_onnx --output-dir models/onnx --validate --fp16
```

### Enable ONNX Backend

```bash
export USE_ONNX=true
export ONNX_MODEL_PATH=models/onnx/clip_vit_h14_text_fp32.onnx
```

### Benchmark

```bash
make onnx-benchmark
```

### ONNX Tuning Options

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_ONNX` | `false` | Enable ONNX Runtime backend |
| `ONNX_MODEL_PATH` | `models/onnx/clip_vit_h14_text_fp32.onnx` | Path to ONNX model |
| `ONNX_PROVIDERS` | `CPUExecutionProvider` | Execution providers (comma-separated) |
| `ONNX_INTRA_OP_THREADS` | `4` | Threads for intra-operator parallelism |
| `ONNX_INTER_OP_THREADS` | `2` | Threads for inter-operator parallelism |
| `ONNX_EXECUTION_MODE` | `parallel` | ORT execution mode |
| `USE_FP16` | `false` | Use FP16 quantized model |

---

## Kubernetes Deployment

Complete Kubernetes manifests using Kustomize with environment overlays.

### Structure

```
k8s/
├── base/           # Shared resources (namespace, configmap, secrets)
├── api/            # API Deployment + Service + HPA + PDB
├── qdrant/         # StatefulSet + Service (persistent storage)
├── redis/          # Deployment + Service (ephemeral cache)
├── ingress/        # Ingress + TLS certificate
├── monitoring/     # OTel Collector + Prometheus + Grafana
└── overlays/
    ├── dev/        # 1 replica, no auth, no monitoring
    ├── staging/    # 2 replicas, auth enabled, 50% sampling
    └── prod/       # 3+ replicas, ONNX enabled, full observability
```

### Deploy

```bash
# Dev environment
kubectl apply -k k8s/overlays/dev

# Production (includes monitoring stack)
kubectl apply -k k8s/overlays/prod
```

### Overlay Differences

| Setting | Dev | Staging | Production |
|---------|-----|---------|------------|
| API replicas | 1 | 2 | 3 |
| HPA max | 2 | 4 | 10 |
| ONNX | off | off | on |
| OTel | off | on (50%) | on (10%) |
| Auth | off | on | on |
| Rate limiting | off | 200/min | 100/min |

---

## Observability

### OpenTelemetry

```bash
export OTEL_ENABLED=true
export OTEL_ENDPOINT=http://localhost:4317

docker compose --profile monitoring up -d
```

**Traces** propagate through the full request path:

- HTTP request span (auto-instrumented by FastAPI middleware)
- Embedding span (CLIP text encode with backend label)
- Retrieval span (Qdrant search with top_k label)
- Cache spans (hit/miss counters)

**Metrics** exported to Prometheus:

- `rag_search_latency` — End-to-end search latency histogram
- `rag_embedding_latency` — CLIP encoding latency by backend
- `rag_retrieval_latency` — Qdrant search latency
- `rag_cache_hits_total` / `rag_cache_misses_total` — Cache effectiveness
- `rag_search_requests_total` — Total request count

### Monitoring Stack

| Service | Port | Purpose |
|---------|------|---------|
| Prometheus | 9090 | Metrics storage |
| Grafana | 3000 | Dashboards |
| OTel Collector | 4317 | Trace/metric ingestion |

---

## Authentication and Rate Limiting

```bash
export AUTH_ENABLED=true
export API_KEYS=key1,key2,key3
export RATE_LIMIT_ENABLED=true
export RATE_LIMIT_REQUESTS=100
export RATE_LIMIT_WINDOW_SECONDS=60
```

When enabled, all requests require an `X-API-Key` header. Rate limiting is per-key using a sliding window.

---

## API Reference

### `POST /search`

Text-to-image semantic search.

```json
{
  "query": "chocolate cake with frosting",
  "top_k": 5,
  "score_threshold": 0.2
}
```

**Response:**

```json
{
  "results": [
    {
      "id": 123456,
      "score": 0.87,
      "metadata": {
        "file_path": "data/food-101/images/chocolate_cake/104030.jpg",
        "file_name": "104030.jpg",
        "width": 512,
        "height": 512
      }
    }
  ],
  "total": 5,
  "latency_ms": 86.2,
  "cached": false
}
```

### `POST /search/image`

Image-to-image search. Upload an image to find visually similar images.

```bash
curl -X POST http://localhost:8000/search/image \
  -F "file=@photo.jpg" \
  -F "top_k=5"
```

### `POST /upload`

Upload and index a new image.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@new_image.jpg" \
  -F "category=desserts"
```

### `GET /health`

Liveness and readiness probe. Returns status of all subsystems.

### `GET /stats`

Runtime latency percentiles (p50/p95/p99) and collection metadata.

---

## Configuration Reference

All settings are configured via environment variables (or `.env` file).

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIP_MODEL_NAME` | `ViT-B-32` | OpenCLIP model architecture |
| `CLIP_PRETRAINED` | `laion2b_s34b_b79k` | Pretrained weights |
| `CLIP_VECTOR_DIM` | `512` | Embedding dimensionality |
| `FORCE_CPU` | `false` | Force CPU inference (disable GPU/MPS) |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant REST port |
| `QDRANT_GRPC_PORT` | `6334` | Qdrant gRPC port |
| `QDRANT_COLLECTION` | `image_vectors` | Collection name |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `REDIS_ENABLED` | `true` | Enable Redis caching |
| `API_PORT` | `8000` | API server port |
| `LLM_ENABLED` | `false` | Enable GPT-4 explanations |
| `OTEL_ENABLED` | `false` | Enable OpenTelemetry |
| `AUTH_ENABLED` | `false` | Enable API key auth |
| `RATE_LIMIT_ENABLED` | `false` | Enable rate limiting |

---

## Performance Targets

| Metric | Target | Measured (CPU) |
|--------|--------|----------------|
| End-to-end search (cold) | < 100ms | ~86ms |
| End-to-end search (cached) | < 1ms | ~0.1ms |
| CLIP text encode | < 50ms (GPU) | ~83ms (CPU) |
| Qdrant HNSW search | < 10ms | ~1.5ms |
| Indexing throughput | — | ~2.8 img/s (CPU) |

---

## Scaling Strategy

| Axis | Mechanism |
|------|-----------|
| Compute | HPA: 2-10 API pod replicas based on CPU/memory |
| Vectors | Qdrant sharding for collections beyond 10M vectors |
| Cache | Redis cluster for distributed caching |
| Throughput | ONNX Runtime for 2-3x faster CPU inference |
| Availability | PDB ensures minimum 1 pod during disruptions |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Single worker per container** | CLIP model lives in memory. Multiple workers duplicate the model. Scale horizontally with replicas. |
| **In-process service calls** | No HTTP/gRPC between embedding and retrieval. Saves ~2-5ms per request. |
| **gRPC for Qdrant** | ~2x faster than REST for vector search operations. |
| **Int8 scalar quantization** | Reduces vector memory ~4x with <1% recall loss. |
| **Fail-open Redis cache** | Cache miss or error falls through to live search. Search never fails due to cache. |
| **No image embedding at query time** | Images are pre-encoded offline. Only text is encoded on the hot path. |
| **Warm model loading** | A dummy forward pass runs at startup to pre-compile kernels. No request hits a cold model. |

---

## License

MIT
