# Deployment Guide

This document covers deploying the Multi-Modal RAG Engine in development, staging, and production environments.

---

## Local Development

### Prerequisites

- Python 3.11+
- Docker and Docker Compose v2
- Node.js 18+ (frontend only)
- 4 GB RAM minimum (8 GB recommended for unified embedder)

### Setup

```bash
# Clone the repository
git clone https://github.com/MeghVyas3132/Multi-Modal-RAG-engine.git
cd Multi-Modal-RAG-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env and set API keys
```

### Start Infrastructure

```bash
# Start Qdrant + Redis
docker compose up -d

# Verify
curl http://localhost:6333/healthz   # Qdrant
redis-cli ping                       # Redis
```

### Start the API

```bash
# Production mode
make serve

# Development mode (auto-reload)
make serve-dev
```

### Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Docker Compose

The `docker-compose.yml` defines the full stack:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| app | Built from Dockerfile | 8000 | API server |
| qdrant | qdrant/qdrant:v1.13.2 | 6333, 6334 | Vector database |
| redis | redis:7-alpine | 6379 | Cache |
| otel-collector | otel/opentelemetry-collector-contrib | 4317 | Telemetry (monitoring profile) |
| prometheus | prom/prometheus | 9090 | Metrics (monitoring profile) |
| grafana | grafana/grafana | 3000 | Dashboards (monitoring profile) |
| jaeger | jaegertracing/all-in-one | 16686 | Traces (monitoring profile) |

### Core Stack

```bash
docker compose up -d
```

### With Monitoring

```bash
docker compose --profile monitoring up -d
```

### Resource Limits

| Service | Memory Limit | Notes |
|---------|-------------|-------|
| app | 6 GB | CLIP + Jina-CLIP + VLM + cross-encoder |
| qdrant | 6 GB | Vectors + HNSW graph in RAM |
| redis | 512 MB | LRU eviction when full |

### Volumes

| Volume | Purpose |
|--------|---------|
| `qdrant_data` | Qdrant persistent storage |
| `huggingface_cache` | Model downloads cache |
| `prometheus_data` | Prometheus TSDB |
| `grafana_data` | Grafana dashboards and config |

---

## Dockerfile

Multi-stage build for minimal image size:

**Stage 1 (deps)**: Installs Python packages from `requirements.txt`. This layer is cached unless `requirements.txt` changes.

**Stage 2 (runtime)**: Copies installed packages and application code. Only this layer rebuilds when code changes.

Key details:
- Base image: `python:3.11-slim`
- Health check: `curl -f http://localhost:8000/health`
- Start period: 300s (5 minutes for model download on first run)
- Single worker: `uvicorn --workers 1`

---

## Kubernetes

Complete Kubernetes manifests using Kustomize with three environment overlays.

### Directory Structure

```
k8s/
+-- base/               # Shared resources
|   +-- namespace.yaml
|   +-- configmap.yaml
|   +-- secrets.yaml
|   +-- kustomization.yaml
+-- api/                # API Deployment
|   +-- deployment.yaml
|   +-- service.yaml
|   +-- hpa.yaml        # Horizontal Pod Autoscaler
|   +-- pdb.yaml        # Pod Disruption Budget
|   +-- hf-cache-pvc.yaml
+-- qdrant/             # Qdrant StatefulSet
|   +-- statefulset.yaml
|   +-- service.yaml
|   +-- pvc.yaml
+-- redis/              # Redis Deployment
|   +-- deployment.yaml
|   +-- service.yaml
+-- ingress/            # Ingress + TLS
|   +-- ingress.yaml
|   +-- certificate.yaml
+-- monitoring/         # Observability stack
|   +-- otel-collector.yaml
|   +-- otel-collector-config.yaml
|   +-- prometheus.yaml
|   +-- prometheus-config.yaml
|   +-- grafana.yaml
+-- overlays/
    +-- dev/
    +-- staging/
    +-- prod/
```

### Deploy

```bash
# Development
kubectl apply -k k8s/overlays/dev

# Staging
kubectl apply -k k8s/overlays/staging

# Production
kubectl apply -k k8s/overlays/prod

# Delete
kubectl delete -k k8s/overlays/dev
```

### Overlay Comparison

| Setting | Dev | Staging | Production |
|---------|-----|---------|------------|
| Replicas | 1 | 2 | 3 |
| HPA max | 2 | 4 | 10 |
| ONNX | off | off | on |
| OTel | off | on (50%) | on (10%) |
| Auth | off | on | on |
| Rate limit | off | 200/min | 100/min |
| Memory request | 2 Gi | 4 Gi | 4 Gi |
| Memory limit | 6 Gi | 8 Gi | 8 Gi |

### Horizontal Pod Autoscaler

The HPA scales API pods based on:
- CPU utilization target: 70%
- Memory utilization target: 80%

### Pod Disruption Budget

The PDB ensures at least 1 pod remains available during voluntary disruptions (node drain, rolling update).

---

## Production Checklist

Before deploying to production, verify:

### Security

- [ ] `AUTH_ENABLED=true` with strong API keys
- [ ] `RATE_LIMIT_ENABLED=true` with appropriate limits
- [ ] TLS certificate configured in ingress
- [ ] API keys stored in Kubernetes secrets, not configmaps
- [ ] CORS origins restricted to your frontend domain

### Performance

- [ ] ONNX Runtime enabled (`USE_ONNX=true`) with exported model
- [ ] Qdrant int8 scalar quantization enabled for large collections
- [ ] Redis `maxmemory` set appropriately
- [ ] HPA configured with reasonable min/max replicas

### Observability

- [ ] `OTEL_ENABLED=true` with reduced sample rate (0.1 for production)
- [ ] Prometheus retention configured (default 7 days)
- [ ] Grafana dashboards imported
- [ ] Alerting rules configured for error rate and latency SLOs

### Data

- [ ] Qdrant persistent volume with appropriate storage class
- [ ] Knowledge graph persistence path on a durable volume
- [ ] PDF upload directory on persistent storage
- [ ] HuggingFace model cache on a shared PVC

### Resilience

- [ ] PDB configured (minAvailable: 1)
- [ ] Liveness and readiness probes configured
- [ ] Redis is fail-open (search works without it)
- [ ] LLM fallback from Cerebras to Groq is tested
