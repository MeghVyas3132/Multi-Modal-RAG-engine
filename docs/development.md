# Development Guide

This document covers local development workflows, testing, code conventions, and contribution guidelines.

---

## Environment Setup

### System Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required for `asyncio.TaskGroup` and typing features |
| pip | 23+ | For dependency resolution |
| Docker | 24+ | For Qdrant and Redis |
| Docker Compose | v2 | Compose plugin (not standalone) |
| Node.js | 18+ | Frontend only |
| Git | 2.40+ | For development workflow |

### Initial Setup

```bash
# Clone
git clone https://github.com/MeghVyas3132/Multi-Modal-RAG-engine.git
cd Multi-Modal-RAG-engine

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### API Keys

At minimum, set the following in `.env`:

```
CEREBRAS_API_KEY=your_key_here
```

Optional keys for full functionality:

| Key | Source | Used For |
|-----|--------|----------|
| CEREBRAS_API_KEY | cerebras.ai | Primary LLM for chat |
| GROQ_API_KEY | console.groq.com | LLM fallback |
| OPENAI_API_KEY | platform.openai.com | VLM fallback (GPT-4o-mini) |
| JINA_API_KEY | jina.ai | Web search grounding |
| FIRECRAWL_API_KEY | firecrawl.dev | Web scraping |

---

## Project Structure

### Backend

```
services/           # Core business logic (one folder per domain)
  api_gateway/      # FastAPI app, endpoints, middleware
  embedding_service/    # Text and image embedding
  retrieval_service/    # Vector search, hybrid retrieval, reranking
  llm_service/      # LLM abstraction (Cerebras, Groq)
  vlm_service/      # Vision-Language Model
  pdf_service/      # PDF parsing
  document_service/ # Semantic text chunking
  routing_service/  # Query modality router
  web_service/      # Web scraping
  graph_service/    # Knowledge graph, entity extraction
  cache_service/    # Semantic cache, deduplication
configs/            # Settings (Pydantic BaseSettings)
utils/              # Logging, metrics, timing decorators
indexing/           # Batch indexing scripts
scripts/            # Utilities (ONNX export, benchmarks, migration)
```

### Frontend

```
frontend/
  src/
    App.jsx         # Router and layout
    main.jsx        # Entry point
    components/     # Reusable UI components
    pages/          # Route pages
    services/       # API client (api.js)
    hooks/          # Custom React hooks
    data/           # Static data
```

### Infrastructure

```
k8s/                # Kubernetes manifests (Kustomize)
docker-compose.yml  # Local development stack
Dockerfile          # Multi-stage production image
Makefile            # Development shortcuts
```

---

## Makefile Commands

The Makefile provides shortcuts for common tasks:

| Command | Description |
|---------|-------------|
| `make serve` | Start API server on port 8000 |
| `make serve-dev` | Start with auto-reload |
| `make index-images` | Run image indexing pipeline |
| `make index-pdf` | Run PDF indexing pipeline |
| `make onnx-convert` | Export CLIP model to ONNX format |
| `make benchmark` | Run performance benchmarks |
| `make benchmark-onnx` | Compare ONNX vs PyTorch performance |
| `make docker-build` | Build Docker image |
| `make docker-up` | Start Docker Compose stack |
| `make docker-down` | Stop Docker Compose stack |
| `make migrate` | Run V1 to V2 migration |
| `make clean` | Remove cache files and artifacts |

---

## Code Conventions

### Architecture Pattern

The codebase follows a service-oriented architecture with singletons:

```python
# Each service initializes lazily via get_instance()
class EmbedderService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

All services are imported and used through their module-level `get_instance()` or factory functions.

### Async vs Sync

- FastAPI endpoints are `async def`
- CPU-bound work (embedding, model inference) runs in `asyncio.get_event_loop().run_in_executor(None, sync_function)`
- Async functions (web scraping, entity extraction) must be `await`ed in async context
- Never call an `async def` function from inside a sync function passed to `run_in_executor`

### Import Style

Each service package exposes its public API through `__init__.py`:

```python
# services/embedding_service/__init__.py
from .unified_embedder import UnifiedEmbedder

__all__ = ["UnifiedEmbedder"]
```

### Configuration

All settings are centralized in `configs/settings.py` using Pydantic `BaseSettings` with `.env` file support:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    CLIP_MODEL_NAME: str = "ViT-B-32"
```

Access via `settings` singleton:

```python
from configs.settings import settings

model_name = settings.CLIP_MODEL_NAME
```

### Logging

Use the structured logger from `utils/logger.py`:

```python
from utils.logger import logger

logger.info("Processing query", extra={"query": query, "modality": modality})
```

### Error Handling

Endpoints use FastAPI `HTTPException` with descriptive messages:

```python
from fastapi import HTTPException

raise HTTPException(status_code=500, detail="Embedding generation failed")
```

Services catch and log exceptions, then re-raise or return graceful defaults.

---

## Adding a New Service

1. Create a new directory under `services/`:

```
services/my_service/
  __init__.py
  my_service.py
```

2. Implement the service class with a singleton pattern:

```python
class MyService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process(self, data):
        pass
```

3. Export from `__init__.py`:

```python
from .my_service import MyService
__all__ = ["MyService"]
```

4. Add configuration to `configs/settings.py` if needed.

5. Create an endpoint in `services/api_gateway/endpoints/` if the service is user-facing.

6. Register the endpoint router in `services/api_gateway/app.py`.

---

## Adding a New Endpoint

1. Create a new file in `services/api_gateway/endpoints/`:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/my-resource", tags=["my-resource"])

@router.post("/action")
async def my_action(request: MyRequest):
    pass
```

2. Register in `services/api_gateway/app.py`:

```python
from services.api_gateway.endpoints import my_endpoint

app.include_router(my_endpoint.router)
```

3. Add the Pydantic request/response models to `services/api_gateway/models.py`.

---

## Indexing Data

### Images

Place images in `data/pdf_images/` organized by folder:

```
data/pdf_images/
  my_topic/
    image1.jpg
    image2.png
```

Run indexing:

```bash
make index-images
```

### PDFs

Place PDF files in `data/pdfs/`:

```
data/pdfs/
  document.pdf
```

Run indexing:

```bash
make index-pdf
```

PDFs can also be uploaded through the API or frontend.

---

## ONNX Model Export

For production performance, export the CLIP model to ONNX format:

```bash
# Export the model
make onnx-convert

# Benchmark to verify speedup
make benchmark-onnx
```

The exported model is saved to `models/onnx/`. Enable it with:

```
USE_ONNX=true
ONNX_MODEL_PATH=models/onnx
```

Expected improvement: 30-40% faster inference on CPU.

---

## Frontend Development

### Setup

```bash
cd frontend
npm install
npm run dev
```

The development server runs on `http://localhost:5173` with hot module replacement.

### Build

```bash
npm run build
```

Output goes to `frontend/dist/`.

### API Configuration

The frontend API client is in `frontend/src/services/api.js`. By default it connects to `http://localhost:8000`.

---

## Troubleshooting

### Models fail to download

If you see download errors on startup, check:
- Internet connectivity
- HuggingFace Hub availability
- Disk space (models require approximately 3 GB total)
- Set `HF_HOME` or `TRANSFORMERS_CACHE` to a writable directory

### Qdrant connection refused

```bash
# Check if Qdrant is running
docker compose ps
docker compose logs qdrant

# Restart
docker compose restart qdrant
```

### Redis connection refused

Redis is fail-open. The system works without it, but with higher latency. To start Redis:

```bash
docker compose restart redis
```

### Out of memory

The API server loads multiple models. If memory is constrained:
- Disable VLM: `ENABLE_VLM=false`
- Use ONNX embedder: `USE_ONNX=true` (lower memory footprint)
- Reduce Qdrant HNSW `m` parameter
- Use a smaller CLIP model variant

### Import errors

Ensure you are running from the project root directory. The application expects imports relative to the project root:

```bash
# Correct
cd /path/to/Multi-Modal-RAG-engine
python -m uvicorn services.api_gateway.app:app

# Incorrect
cd services/api_gateway
python app.py
```
