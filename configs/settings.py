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

    # ── CLIP ────────────────────────────────────────────────
    clip_model_name: str = Field(default="ViT-B-32")
    clip_pretrained: str = Field(default="laion2b_s34b_b79k")
    clip_vector_dim: int = Field(default=512)
    force_cpu: bool = Field(default=False)

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

    # ── OpenAI (legacy, unused) ─────────────────────────────
    openai_api_key: str = Field(default="")

    # ── Text Embeddings (for PDF RAG) ───────────────────────
    text_model_name: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model for text")
    text_vector_dim: int = Field(default=384, description="Text embedding dimension")
    pdf_text_collection: str = Field(default="pdf_text_vectors", description="Qdrant collection for PDF text chunks")

    # ── PDF Processing ──────────────────────────────────────
    pdf_chunk_size: int = Field(default=512, description="Max chars per text chunk")
    pdf_chunk_overlap: int = Field(default=64, description="Overlap chars between chunks")
    pdf_upload_dir: str = Field(default="./data/pdfs", description="Directory for uploaded PDFs")

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
    otel_service_name: str = Field(default="clip-image-search")
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
