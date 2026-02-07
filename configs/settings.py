"""
Centralized configuration — loaded once at process startup.

Why a single settings module?
  - Every service reads the same env vars.
  - Pydantic validates types at import time so we fail fast on bad config.
  - No scattered os.getenv() calls across the codebase.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Immutable, validated application settings from environment."""

    # ── CLIP (legacy, kept for backward compat) ─────────────
    clip_model_name: str = Field(default="ViT-B-32")
    clip_pretrained: str = Field(default="laion2b_s34b_b79k")
    clip_vector_dim: int = Field(default=512)
    force_cpu: bool = Field(default=False)

    # ── Unified Embeddings (V2 — single space for text+image)
    unified_enabled: bool = Field(default=True, description="Use unified embedder for cross-modal search")
    unified_model_name: str = Field(default="jinaai/jina-clip-v2", description="Unified text+image model")
    unified_vector_dim: int = Field(default=768, description="Unified embedding dimension")
    unified_collection: str = Field(default="unified_vectors", description="Single Qdrant collection for all modalities")
    unified_batch_size: int = Field(default=32, description="Batch size for unified encoding")

    # ── Qdrant ──────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_grpc_port: int = Field(default=6334)
    qdrant_collection: str = Field(default="image_vectors")
    qdrant_hnsw_m: int = Field(default=16)
    qdrant_hnsw_ef_construct: int = Field(default=200)
    qdrant_hnsw_ef: int = Field(default=128)

    # ── Redis ───────────────────────────────────────────────
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_cache_ttl: int = Field(default=3600)
    redis_enabled: bool = Field(default=True)

    # ── API ─────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)

    # ── Indexing ────────────────────────────────────────────
    image_dir: str = Field(default="./data/images")
    index_batch_size: int = Field(default=256)
    index_num_workers: int = Field(default=4)

    # ── LLM ─────────────────────────────────────────────────
    llm_enabled: bool = Field(default=True)
    llm_model: str = Field(default="llama-3.3-70b")

    # ── Cerebras (primary LLM provider) ─────────────────────
    cerebras_api_key: str = Field(default="", description="Cerebras Cloud API key")
    cerebras_base_url: str = Field(default="https://api.cerebras.ai/v1")

    # ── Groq (fallback LLM provider) ────────────────────────
    groq_api_key: str = Field(default="", description="Groq Cloud API key")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")

    # ── OpenAI (GPT-4o-mini for VLM fallback + router) ─────
    openai_api_key: str = Field(default="", description="OpenAI API key for GPT-4o-mini VLM fallback")

    # ── Text Embeddings (legacy, kept for backward compat) ──
    text_model_name: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model for text")
    text_vector_dim: int = Field(default=384, description="Text embedding dimension")
    pdf_text_collection: str = Field(default="pdf_text_vectors", description="Qdrant collection for PDF text chunks")

    # ── PDF Processing ──────────────────────────────────────
    pdf_chunk_size: int = Field(default=512, description="Max chars per text chunk")
    pdf_chunk_overlap: int = Field(default=64, description="Overlap chars between chunks")
    pdf_upload_dir: str = Field(default="./data/pdfs", description="Directory for uploaded PDFs")
    parser_backend: str = Field(default="pymupdf", description="PDF parser: pymupdf or mineru")

    # ── VLM (Vision-Language Model) ─────────────────────────
    vlm_enabled: bool = Field(default=True, description="Enable local VLM for image understanding")
    vlm_model_name: str = Field(default="HuggingFaceTB/SmolVLM-500M-Instruct", description="Local VLM model")
    vlm_caption_batch_size: int = Field(default=8, description="Batch size for VLM captioning")
    vlm_confidence_threshold: float = Field(default=0.85, description="Below this, use GPT-4o fallback")
    vlm_cache_dir: str = Field(default="./data/vlm_cache", description="Caption cache directory")
    vlm_max_new_tokens: int = Field(default=256, description="Max tokens for VLM generation")

    # ── Modality Router ─────────────────────────────────────
    router_enabled: bool = Field(default=True, description="Enable modality-aware query routing")
    router_mode: str = Field(default="heuristic", description="Router mode: heuristic, llm, or trained")
    router_cache_size: int = Field(default=1000, description="LRU cache size for router decisions")
    router_hybrid_threshold: float = Field(default=0.7, description="Below this confidence, use hybrid search")

    # ── Web Scraping ────────────────────────────────────────
    web_scraping_enabled: bool = Field(default=True, description="Enable web content ingestion")
    jina_api_key: str = Field(default="", description="Jina Reader API key (optional, free tier)")
    firecrawl_api_key: str = Field(default="", description="Firecrawl API key for JS rendering")
    web_grounding_threshold: float = Field(default=0.65, description="Below this retrieval score, search web")
    web_cache_dir: str = Field(default="./data/web_cache", description="Web content cache directory")
    web_max_pages_per_crawl: int = Field(default=50, description="Max pages per crawl job")

    # ── Knowledge Graph ─────────────────────────────────────
    graph_enabled: bool = Field(default=True, description="Enable knowledge graph construction")
    graph_persist_path: str = Field(default="./data/knowledge_graph.json", description="Graph persistence file")
    graph_max_hops: int = Field(default=2, description="Max hops for graph traversal")
    graph_entity_batch_size: int = Field(default=16, description="Chunks per LLM entity extraction call")

    # ── Semantic Cache ──────────────────────────────────────
    semantic_cache_enabled: bool = Field(default=True, description="Enable multi-tier semantic caching")
    semantic_cache_l1_size: int = Field(default=500, description="In-process LRU cache size")
    semantic_cache_l2_ttl: int = Field(default=86400, description="Redis semantic cache TTL (24h)")
    semantic_cache_similarity: float = Field(default=0.93, description="Cosine threshold for semantic cache hits")
    llm_cache_ttl: int = Field(default=604800, description="LLM response cache TTL (7 days)")
    dedup_threshold: float = Field(default=0.98, description="Cosine threshold for chunk deduplication")

    # ── Semantic Chunking ───────────────────────────────────
    chunking_strategy: str = Field(default="semantic", description="Chunking: fixed, semantic, or hierarchical")
    semantic_chunk_min_size: int = Field(default=128, description="Min chars per semantic chunk")
    semantic_chunk_max_size: int = Field(default=1024, description="Max chars per semantic chunk")
    semantic_chunk_threshold: float = Field(default=0.75, description="Similarity threshold for topic boundary")

    # ── Multi-lingual ───────────────────────────────────────
    multilingual_enabled: bool = Field(default=True, description="Enable language detection and multilingual support")
    default_language: str = Field(default="en", description="Default language for monolingual fallback")

    # ── Hybrid Retrieval ────────────────────────────────────
    hybrid_search_enabled: bool = Field(default=True, description="Enable hybrid dense+sparse retrieval")
    reranker_enabled: bool = Field(default=True, description="Enable cross-encoder reranking")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Reranker model")
    reranker_top_n: int = Field(default=10, description="How many results to return after reranking")
    retrieval_top_k_initial: int = Field(default=50, description="Initial retrieval candidates before reranking")

    # ── Search defaults ─────────────────────────────────────
    search_top_k: int = Field(default=10)
    search_score_threshold: float = Field(default=0.2)

    # ── ONNX Runtime ───────────────────────────────────────
    use_onnx: bool = Field(default=False, description="Use ONNX Runtime instead of PyTorch for inference")
    onnx_model_path: str = Field(default="models/onnx/clip_vit_h14_text_fp32.onnx")
    onnx_providers: str = Field(default="CPUExecutionProvider", description="Comma-separated ONNX execution providers")
    onnx_intra_op_threads: int = Field(default=4, description="Threads for intra-operator parallelism")
    onnx_inter_op_threads: int = Field(default=2, description="Threads for inter-operator parallelism")
    onnx_execution_mode: str = Field(default="parallel", description="ORT execution mode: parallel or sequential")
    use_fp16: bool = Field(default=False, description="Use FP16 quantized ONNX model")

    # ── Observability ──────────────────────────────────────
    otel_enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing and metrics")
    otel_endpoint: str = Field(default="http://localhost:4317", description="OTLP gRPC endpoint")
    otel_service_name: str = Field(default="multimodal-rag-engine")
    otel_sample_rate: float = Field(default=1.0, description="Trace sampling rate (0.0 to 1.0)")

    # ── Auth / Rate Limiting ────────────────────────────────
    auth_enabled: bool = Field(default=False, description="Enable API key authentication")
    api_keys: str = Field(default="", description="Comma-separated valid API keys")
    rate_limit_enabled: bool = Field(default=False, description="Enable per-key rate limiting")
    rate_limit_requests: int = Field(default=100, description="Max requests per window per key")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window in seconds")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton accessor — parsed once and cached for the process lifetime.
    Import this wherever you need config:
        from configs.settings import get_settings
        cfg = get_settings()
    """
    return Settings()
