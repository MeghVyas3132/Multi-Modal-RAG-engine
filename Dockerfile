# ================================================================
# Dockerfile -- Multi-Modal RAG Image Search API
#
# Multi-stage build:
#   Stage 1: Install Python dependencies (cached layer)
#   Stage 2: Copy application code (thin layer, fast rebuilds)
#
# The image runs a single uvicorn worker. CLIP model is loaded
# into memory at startup. Scale horizontally with replicas,
# not vertically with multiple workers.
#
# ONNX Runtime is included for optional accelerated inference.
# Set USE_ONNX=true and mount the ONNX model to enable.
# ================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 1: Dependencies ──────────────────────────────────
FROM base AS deps

# System dependencies required by PyTorch, Pillow, gRPC, and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Application ───────────────────────────────────
FROM base AS runtime

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# System libs needed at runtime (Pillow, OpenCV, gRPC)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY configs/ ./configs/
COPY services/ ./services/
COPY indexing/ ./indexing/
COPY scripts/ ./scripts/
COPY utils/ ./utils/

# Create models directory for ONNX models (mount at runtime)
RUN mkdir -p models/onnx

# Default port
EXPOSE 8000

# Health check against the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker -- CLIP model lives in this process.
# Scaling is done with container replicas, not workers.
CMD ["uvicorn", "services.api_gateway.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
