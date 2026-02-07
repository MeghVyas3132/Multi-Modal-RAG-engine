# Configuration Reference

All settings are managed through environment variables or a `.env` file. The `configs/settings.py` module validates all values at startup using Pydantic.

---

## Loading Order

1. Environment variables (highest priority)
2. `.env` file in the project root
3. Default values in `configs/settings.py`

---

## CLIP Model (Legacy)

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIP_MODEL_NAME` | `ViT-B-32` | OpenCLIP model architecture |
| `CLIP_PRETRAINED` | `laion2b_s34b_b79k` | Pretrained weights identifier |
| `CLIP_VECTOR_DIM` | `512` | Embedding dimensionality (must match model) |
| `FORCE_CPU` | `false` | Force CPU inference even if GPU/MPS is available |

Common model configurations:

| Model | Pretrained | Dim | RAM | Speed (CPU) |
|-------|-----------|-----|-----|-------------|
| ViT-B-32 | laion2b_s34b_b79k | 512 | ~600 MB | ~83ms |
| ViT-L-14 | laion2b_s32b_b82k | 768 | ~1.2 GB | ~200ms |
| ViT-H-14 | laion2b_s32b_b79k | 1024 | ~2.5 GB | ~400ms |

---

## Unified Embeddings (V2)

| Variable | Default | Description |
|----------|---------|-------------|
| `UNIFIED_ENABLED` | `true` | Use unified embedder for cross-modal search |
| `UNIFIED_MODEL_NAME` | `jinaai/jina-clip-v2` | Unified text+image model |
| `UNIFIED_VECTOR_DIM` | `768` | Unified embedding dimension |
| `UNIFIED_COLLECTION` | `unified_vectors` | Single Qdrant collection for all modalities |
| `UNIFIED_BATCH_SIZE` | `32` | Batch size for unified encoding |

---

## Qdrant

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant REST API port |
| `QDRANT_GRPC_PORT` | `6334` | Qdrant gRPC port (used for search) |
| `QDRANT_COLLECTION` | `image_vectors` | Legacy image collection name |
| `QDRANT_HNSW_M` | `16` | HNSW graph degree (higher = better recall, more RAM) |
| `QDRANT_HNSW_EF_CONSTRUCT` | `200` | HNSW build-time search width |
| `QDRANT_HNSW_EF` | `128` | HNSW search-time search width |

---

## Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `REDIS_ENABLED` | `true` | Enable Redis caching (fail-open if disabled or unreachable) |

---

## API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `API_WORKERS` | `1` | Uvicorn worker count (keep at 1, see architecture docs) |

---

## LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENABLED` | `true` | Enable LLM for chat responses |
| `LLM_MODEL` | `llama-3.3-70b` | LLM model name |
| `CEREBRAS_API_KEY` | `""` | Cerebras Cloud API key (primary LLM) |
| `CEREBRAS_BASE_URL` | `https://api.cerebras.ai/v1` | Cerebras API endpoint |
| `GROQ_API_KEY` | `""` | Groq Cloud API key (fallback LLM) |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq API endpoint |
| `OPENAI_API_KEY` | `""` | OpenAI API key (VLM fallback) |

---

## VLM (Vision-Language Model)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_ENABLED` | `true` | Enable local VLM for image captioning |
| `VLM_MODEL_NAME` | `HuggingFaceTB/SmolVLM-500M-Instruct` | Local VLM model |
| `VLM_CAPTION_BATCH_SIZE` | `8` | Batch size for VLM captioning |
| `VLM_CONFIDENCE_THRESHOLD` | `0.85` | Below this confidence, use GPT-4o-mini fallback |
| `VLM_CACHE_DIR` | `./data/vlm_cache` | Caption cache directory |
| `VLM_MAX_NEW_TOKENS` | `256` | Max tokens for VLM generation |

---

## Modality Router

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_ENABLED` | `true` | Enable modality-aware query routing |
| `ROUTER_MODE` | `heuristic` | Router mode: `heuristic`, `llm`, or `trained` |
| `ROUTER_CACHE_SIZE` | `1000` | LRU cache size for router decisions |
| `ROUTER_HYBRID_THRESHOLD` | `0.7` | Below this confidence, use hybrid search |

---

## Web Scraping

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_SCRAPING_ENABLED` | `true` | Enable web content ingestion |
| `JINA_API_KEY` | `""` | Jina Reader API key (free tier works without) |
| `FIRECRAWL_API_KEY` | `""` | Firecrawl API key for JS-rendered pages |
| `WEB_GROUNDING_THRESHOLD` | `0.65` | Below this retrieval score, search the web |
| `WEB_CACHE_DIR` | `./data/web_cache` | Web content cache directory |
| `WEB_MAX_PAGES_PER_CRAWL` | `50` | Max pages per crawl job |

---

## Knowledge Graph

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPH_ENABLED` | `true` | Enable knowledge graph construction |
| `GRAPH_PERSIST_PATH` | `./data/knowledge_graph.json` | Graph persistence file path |
| `GRAPH_MAX_HOPS` | `2` | Max hops for BFS graph traversal |
| `GRAPH_ENTITY_BATCH_SIZE` | `16` | Chunks per LLM entity extraction call |

---

## Semantic Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_CACHE_ENABLED` | `true` | Enable multi-tier semantic caching |
| `SEMANTIC_CACHE_L1_SIZE` | `500` | In-process LRU cache max entries |
| `SEMANTIC_CACHE_L2_TTL` | `86400` | Redis cache TTL in seconds (24 hours) |
| `SEMANTIC_CACHE_SIMILARITY` | `0.93` | Cosine threshold for L3 semantic cache hits |
| `LLM_CACHE_TTL` | `604800` | LLM response cache TTL in seconds (7 days) |
| `DEDUP_THRESHOLD` | `0.98` | Cosine threshold for chunk deduplication |

---

## Semantic Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNKING_STRATEGY` | `semantic` | Chunking mode: `fixed`, `semantic`, or `hierarchical` |
| `SEMANTIC_CHUNK_MIN_SIZE` | `128` | Minimum characters per semantic chunk |
| `SEMANTIC_CHUNK_MAX_SIZE` | `1024` | Maximum characters per semantic chunk |
| `SEMANTIC_CHUNK_THRESHOLD` | `0.75` | Cosine similarity threshold for topic boundary |

---

## Hybrid Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `HYBRID_SEARCH_ENABLED` | `true` | Enable hybrid dense+sparse retrieval |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model name |
| `RERANKER_TOP_N` | `10` | Results to return after reranking |
| `RETRIEVAL_TOP_K_INITIAL` | `50` | Initial retrieval candidates before reranking |

---

## Text Embeddings (Legacy)

| Variable | Default | Description |
|----------|---------|-------------|
| `TEXT_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TEXT_VECTOR_DIM` | `384` | Text embedding dimension |
| `PDF_TEXT_COLLECTION` | `pdf_text_vectors` | Qdrant collection for PDF text chunks |

---

## PDF Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `PDF_CHUNK_SIZE` | `512` | Max characters per text chunk (fixed mode) |
| `PDF_CHUNK_OVERLAP` | `64` | Overlap characters between chunks |
| `PDF_UPLOAD_DIR` | `./data/pdfs` | Directory for uploaded PDF files |
| `PARSER_BACKEND` | `pymupdf` | PDF parser: `pymupdf` or `mineru` |

---

## Search Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_TOP_K` | `10` | Default number of search results |
| `SEARCH_SCORE_THRESHOLD` | `0.2` | Default minimum similarity score |

---

## ONNX Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_ONNX` | `false` | Enable ONNX Runtime backend |
| `ONNX_MODEL_PATH` | `models/onnx/clip_vit_h14_text_fp32.onnx` | Path to ONNX model file |
| `ONNX_PROVIDERS` | `CPUExecutionProvider` | Execution providers (comma-separated) |
| `ONNX_INTRA_OP_THREADS` | `4` | Threads for intra-operator parallelism |
| `ONNX_INTER_OP_THREADS` | `2` | Threads for inter-operator parallelism |
| `ONNX_EXECUTION_MODE` | `parallel` | ORT execution mode: `parallel` or `sequential` |
| `USE_FP16` | `false` | Use FP16 quantized ONNX model |

---

## OpenTelemetry

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `true` | Enable OpenTelemetry tracing and metrics |
| `OTEL_ENDPOINT` | `http://localhost:4317` | OTLP gRPC endpoint |
| `OTEL_SERVICE_NAME` | `multimodal-rag-engine` | Service name in traces |
| `OTEL_SAMPLE_RATE` | `1.0` | Trace sampling rate (0.0 to 1.0) |

---

## Authentication and Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_ENABLED` | `false` | Enable API key authentication |
| `API_KEYS` | `""` | Comma-separated valid API keys |
| `RATE_LIMIT_ENABLED` | `false` | Enable per-key rate limiting |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window per key |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window in seconds |

---

## Indexing

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_DIR` | `./data/images` | Default image directory for batch indexing |
| `INDEX_BATCH_SIZE` | `256` | Batch size for offline indexing |
| `INDEX_NUM_WORKERS` | `4` | Worker threads for indexing |

---

## Multi-lingual

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTILINGUAL_ENABLED` | `true` | Enable language detection |
| `DEFAULT_LANGUAGE` | `en` | Default language for monolingual fallback |
