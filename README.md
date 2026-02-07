# Multi-Modal RAG Engine

A production-grade Retrieval-Augmented Generation system that operates across text, image, and web modalities. The engine indexes documents (PDFs, images, web pages), encodes them into a shared vector space, and retrieves the most relevant content for natural language queries. LLM-powered chat provides grounded answers with source attribution.

Version 2.0 introduces unified cross-modal embeddings, a Vision-Language Model for image understanding, a knowledge graph for multi-hop reasoning, semantic caching, cross-encoder reranking, and web content grounding.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Keys and Configuration](#api-keys-and-configuration)
- [API Reference](#api-reference)
- [ONNX Runtime Acceleration](#onnx-runtime-acceleration)
- [Observability (OpenTelemetry)](#observability-opentelemetry)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Documentation](#documentation)
- [Performance Targets](#performance-targets)
- [License](#license)

---

## Architecture Overview

```
                         +-------------------+
                         |   React Frontend  |
                         | (Vite + Tailwind) |
                         +--------+----------+
                                  |
                                  v
+---------------------------------------------------------------+
|                     FastAPI API Gateway                        |
|  /search  /chat  /upload  /web  /graph  /health  /stats       |
+-------+--------+--------+--------+--------+--------+---------+
        |        |        |        |        |        |
   +----v---+ +--v----+ +-v------+ +--v---+ +--v--+ +---v---+
   |Semantic| |Modality| |Unified | | VLM  | |Graph| |Semantic|
   | Cache  | |Router  | |Embedder| |SmolVLM| |  KG | | Cache |
   | L1/L2  | |        | |Jina v2 | |      | |     | | L3    |
   +--------+ +--------+ +--------+ +------+ +-----+ +-------+
        |                     |                  |
        v                     v                  v
   +--------+         +-------------+     +-----------+
   | Redis  |         |   Qdrant    |     | NetworkX  |
   | L2     |         | HNSW + gRPC |     | DiGraph   |
   +--------+         +------+------+     +-----------+
                             |
                      +------v------+
                      |  Reranker   |
                      | CrossEncoder|
                      +-------------+
                             |
                      +------v------+
                      | LLM Service |
                      | Cerebras    |
                      +-------------+
```

**V2 search pipeline:**

1. Check semantic cache: L1 in-process LRU, L2 Redis, L3 Qdrant ANN (cosine > 0.93)
2. Route query through modality router (text/image/hybrid detection)
3. Encode query with unified embedder (Jina-CLIP v2, 768-dim)
4. Hybrid search with knowledge graph expansion and RRF fusion
5. Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
6. Store result in cache (fire-and-forget)

**V2 chat pipeline:**

1. Semantic cache check for query
2. Modality routing and unified retrieval with graph expansion
3. Cross-encoder reranking
4. LLM response cache check (7-day TTL)
5. Stream LLM answer via SSE (Cerebras primary, Groq fallback)

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | FastAPI 0.115 + Uvicorn | Async HTTP server, SSE streaming |
| Unified Embeddings | Jina-CLIP v2 (768-dim) | Cross-modal text + image encoding |
| Legacy Embeddings | OpenCLIP ViT-B-32 / ViT-H-14 | Backward-compatible CLIP encoding |
| Text Embeddings | sentence-transformers (MiniLM-L6-v2) | PDF text chunk encoding |
| ONNX Runtime | onnxruntime 1.19 | 2-3x faster CPU inference |
| VLM | SmolVLM-500M-Instruct | Local image captioning |
| VLM Fallback | GPT-4o-mini | Cloud VLM when local confidence is low |
| Vector DB | Qdrant 1.13+ (gRPC, HNSW, int8 quantization) | ANN search |
| Cache L1 | In-process OrderedDict LRU | Sub-millisecond cache |
| Cache L2 | Redis 7+ (LRU, fail-open) | Distributed cache |
| Cache L3 | Qdrant ANN semantic cache | Fuzzy query matching |
| Knowledge Graph | NetworkX DiGraph + JSON persistence | Multi-hop reasoning |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precision reranking |
| LLM (Primary) | Cerebras (llama-3.3-70b) | Chat response generation |
| LLM (Fallback) | Groq (llama-3.3-70b) | Failover LLM |
| Semantic Chunking | Embedding similarity boundaries | Topic-aware text splitting |
| Web Scraping | Jina Reader / Firecrawl / httpx | Web content ingestion |
| Observability | OpenTelemetry + Prometheus + Grafana + Jaeger | Traces, metrics, dashboards |
| Auth | API key middleware + slowapi rate limiting | Request authentication |
| Orchestration | Docker Compose (dev) / Kubernetes + Kustomize (prod) | Container management |
| Frontend | React 18 + Vite + Tailwind CSS | Chat-based search UI |

---

## Project Structure

```
.
+-- configs/
|   +-- settings.py                     # Centralized Pydantic settings (60+ env vars)
+-- services/
|   +-- api_gateway/
|   |   +-- app.py                      # FastAPI app, lifespan, router mounting
|   |   +-- models.py                   # Pydantic request/response models
|   |   +-- cache.py                    # Redis cache layer (fail-open)
|   |   +-- telemetry.py                # OpenTelemetry instrumentation
|   |   +-- middleware/                 # API key auth + rate limiting
|   |   +-- endpoints/
|   |       +-- search.py               # /search, /search/image
|   |       +-- chat.py                 # /chat (SSE streaming RAG)
|   |       +-- upload.py               # /upload, /upload/pdf, /pdfs
|   |       +-- web.py                  # /web/index, /web/search-grounding
|   |       +-- graph.py                # /graph/stats, /graph/related, /graph/expand
|   |       +-- health.py               # /health, /stats
|   +-- embedding_service/
|   |   +-- embedder.py                 # PyTorch CLIP text/image encoder
|   |   +-- onnx_embedder.py            # ONNX Runtime CLIP text encoder
|   |   +-- text_embedder.py            # sentence-transformers for text
|   |   +-- unified_embedder.py         # Jina-CLIP v2 cross-modal embedder
|   +-- retrieval_service/
|   |   +-- retriever.py                # Legacy Qdrant search + upsert
|   |   +-- hybrid_retriever.py         # Unified collection, RRF fusion, graph expansion
|   |   +-- reranker.py                 # Cross-encoder precision reranking
|   +-- cache_service/
|   |   +-- semantic_cache.py           # L1/L2/L3 multi-tier semantic cache
|   |   +-- deduplication.py            # SHA256 + cosine semantic deduplication
|   +-- document_service/
|   |   +-- semantic_chunker.py         # Topic-boundary semantic chunking
|   +-- graph_service/
|   |   +-- entity_extractor.py         # LLM-based entity/relation extraction
|   |   +-- knowledge_graph.py          # NetworkX graph with BFS traversal
|   +-- vlm_service/
|   |   +-- local_vlm.py               # SmolVLM-500M + GPT-4o-mini fallback
|   +-- routing_service/
|   |   +-- modality_router.py          # Heuristic query routing with LRU cache
|   +-- web_service/
|   |   +-- web_scraper.py              # Jina/Firecrawl/httpx + YouTube + GitHub
|   +-- llm_service/
|   |   +-- llm.py                      # Cerebras + Groq with auto-failover
|   +-- pdf_service/
|       +-- parser.py                   # PyMuPDF PDF parsing
+-- indexing/
|   +-- index_images.py                 # Offline batch image indexing
+-- scripts/
|   +-- convert_to_onnx.py              # Export CLIP to ONNX
|   +-- benchmark_onnx_vs_pytorch.py    # ONNX vs PyTorch benchmark
|   +-- benchmark.py                    # Search latency benchmark
|   +-- simulate_dataset.py             # Generate test vectors
|   +-- migrate_to_unified.py           # Migrate legacy collections to V2
+-- frontend/                           # React app (chat UI, image search)
+-- docs/                               # Technical documentation
+-- k8s/                                # Kubernetes manifests (Kustomize)
+-- Dockerfile                          # Multi-stage production build
+-- docker-compose.yml                  # Full stack with monitoring profile
+-- Makefile                            # Development task runner
+-- requirements.txt                    # Pinned Python dependencies
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for Qdrant and Redis)
- Node.js 18+ (for frontend)

### 1. Clone and Install

```bash
git clone https://github.com/MeghVyas3132/Multi-Modal-RAG-engine.git
cd Multi-Modal-RAG-engine

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set CEREBRAS_API_KEY at minimum
```

### 3. Start Infrastructure

```bash
# Docker Compose (recommended)
docker compose up -d

# Or native
./bin/qdrant --config-path ./qdrant_config.yaml &
redis-server &
```

### 4. Start the API Server

```bash
make serve
# Server: http://localhost:8000
# Docs:   http://localhost:8000/docs
```

### 5. Start the Frontend

```bash
cd frontend && npm install && npm run dev
# Opens at http://localhost:5173
```

### 6. Start Observability Stack (Optional)

```bash
docker compose --profile monitoring up -d
# Grafana:    http://localhost:3000  (admin/admin)
# Prometheus: http://localhost:9090
# Jaeger:     http://localhost:16686
```

---

## API Keys and Configuration

### Required

| Key | Provider | Purpose |
|-----|----------|---------|
| `CEREBRAS_API_KEY` | [Cerebras](https://cloud.cerebras.ai/) | Primary LLM for /chat |

### Recommended

| Key | Provider | Purpose |
|-----|----------|---------|
| `GROQ_API_KEY` | [Groq](https://console.groq.com/) | Fallback LLM |

### Optional

| Key | Provider | Purpose |
|-----|----------|---------|
| `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com/) | GPT-4o-mini VLM fallback |
| `JINA_API_KEY` | [Jina AI](https://jina.ai/) | Web scraping (free tier works without key) |
| `FIRECRAWL_API_KEY` | [Firecrawl](https://firecrawl.dev/) | JS-rendered web scraping |

Search, upload, and indexing work without any API keys. Only the chat feature requires `CEREBRAS_API_KEY`.

See [docs/configuration.md](docs/configuration.md) for the complete list of 60+ environment variables.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/search` | Text-to-image/text semantic search |
| POST | `/search/image` | Image-to-image similarity search |
| POST | `/chat` | RAG chat with SSE streaming |
| POST | `/upload` | Upload and index an image |
| POST | `/upload/pdf` | Upload and index a PDF |
| GET | `/pdfs` | List uploaded PDFs |
| POST | `/web/index` | Scrape and index a URL |
| POST | `/web/search-grounding` | Web-augmented search fallback |
| GET | `/graph/stats` | Graph node/edge statistics |
| GET | `/graph/related` | Find related entities |
| GET | `/graph/expand` | Expand query with graph entities |
| POST | `/graph/save` | Persist graph to disk |
| GET | `/health` | Liveness and readiness probe |
| GET | `/stats` | Runtime latency percentiles |

Full documentation with request/response schemas: [docs/api-reference.md](docs/api-reference.md)

---

## ONNX Runtime Acceleration

ONNX Runtime provides 2-3x faster CPU inference. See [docs/onnx.md](docs/onnx.md) for details.

```bash
make onnx-convert          # Export CLIP text encoder to ONNX
export USE_ONNX=true       # Enable in .env
make onnx-benchmark        # Verify speedup
```

---

## Observability (OpenTelemetry)

Full distributed tracing, metrics, and dashboards. See [docs/observability.md](docs/observability.md).

```bash
export OTEL_ENABLED=true
docker compose --profile monitoring up -d
```

| Service | Port | Purpose |
|---------|------|---------|
| Grafana | 3000 | Dashboards |
| Prometheus | 9090 | Metrics |
| Jaeger | 16686 | Traces |

---

## Kubernetes Deployment

See [docs/deployment.md](docs/deployment.md) for complete instructions.

```bash
kubectl apply -k k8s/overlays/dev       # Development
kubectl apply -k k8s/overlays/staging   # Staging
kubectl apply -k k8s/overlays/prod      # Production
```

---

## Documentation

Detailed technical documentation is in the [docs/](docs/) directory:

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design, data flow, and design decisions |
| [Services](docs/services.md) | Deep dive into each service module |
| [API Reference](docs/api-reference.md) | Complete endpoint documentation with examples |
| [Configuration](docs/configuration.md) | All environment variables and their defaults |
| [Deployment](docs/deployment.md) | Docker, Kubernetes, and production setup |
| [Migration](docs/migration.md) | Migrating from V1 to V2 |
| [Observability](docs/observability.md) | OpenTelemetry, Prometheus, Grafana setup |
| [ONNX Runtime](docs/onnx.md) | Model export, benchmarking, and tuning |
| [Frontend](docs/frontend.md) | React app setup and API integration |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |

---

## Performance Targets

| Metric | Target | Measured (CPU) |
|--------|--------|----------------|
| End-to-end search (cold) | < 100ms | ~86ms |
| End-to-end search (cached L1) | < 1ms | ~0.1ms |
| Unified embedding (Jina-CLIP v2) | < 50ms | ~30ms |
| Legacy CLIP encode | < 50ms (GPU) | ~83ms (CPU) |
| Qdrant HNSW search | < 10ms | ~1.5ms |
| Cross-encoder reranking (50 candidates) | < 150ms | ~100ms |
| Chat SSE first-token | < 500ms | ~300ms |

---

## License

MIT
