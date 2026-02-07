# Monitoring and Observability

This document covers the observability stack, metrics, tracing, and dashboarding for the Multi-Modal RAG Engine.

---

## Stack Overview

| Component | Purpose | Port |
|-----------|---------|------|
| OpenTelemetry Collector | Receives and routes telemetry | 4317 (gRPC) |
| Prometheus | Metrics storage and querying | 9090 |
| Grafana | Dashboards and alerting | 3000 |
| Jaeger | Distributed tracing | 16686 |

---

## Enabling Observability

### Environment Variables

```env
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_SERVICE_NAME=multimodal-rag-engine
```

### Docker Compose

Start the full monitoring stack:

```bash
docker compose --profile monitoring up -d
```

This starts the otel-collector, Prometheus, Grafana, and Jaeger alongside the core services.

---

## Instrumentation

### Telemetry Module

The telemetry system is implemented in `services/api_gateway/telemetry.py`:

- **Tracer**: Creates spans for request lifecycle tracking
- **Meter**: Exposes counters, histograms, and gauges
- **Resource**: Tags all telemetry with `service.name=multimodal-rag-engine` and `service.version=2.0.0`
- **Exporter**: OTLP gRPC to the collector

When `OTEL_ENABLED=false`, the module returns no-op providers that produce zero overhead.

### Automatic Instrumentation

The following are instrumented automatically via OpenTelemetry contrib packages:

| Library | Package |
|---------|---------|
| FastAPI | `opentelemetry-instrumentation-fastapi` |
| HTTP client | `opentelemetry-instrumentation-httpx` |
| Logging | `opentelemetry-instrumentation-logging` |

### Custom Metrics

The application exposes the following custom metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `search_requests_total` | Counter | Total search requests by modality |
| `search_latency_seconds` | Histogram | Search request duration |
| `embedding_latency_seconds` | Histogram | Embedding generation duration |
| `llm_requests_total` | Counter | LLM API calls by provider |
| `llm_latency_seconds` | Histogram | LLM response time |
| `cache_hits_total` | Counter | Cache hit count |
| `cache_misses_total` | Counter | Cache miss count |
| `upload_requests_total` | Counter | Document upload count |
| `reranker_latency_seconds` | Histogram | Reranking duration |

### Custom Spans

Key operations create child spans:

- `embed_text` / `embed_image` -- embedding generation
- `qdrant_search` -- vector similarity search
- `rerank` -- cross-encoder reranking
- `llm_generate` -- LLM text generation
- `vlm_caption` -- VLM image captioning
- `entity_extraction` -- knowledge graph entity extraction
- `cache_lookup` -- semantic cache check
- `web_scrape` -- web page scraping

---

## Timing Decorators

The `utils/timing.py` module provides decorators for performance measurement:

```python
from utils.timing import timed

@timed
def expensive_operation():
    pass
```

This logs the execution time and, if OTel is enabled, records it as a span.

---

## Prometheus Metrics

### Endpoint

The application exposes a Prometheus metrics endpoint at `/metrics` (when OTel is enabled).

### Prometheus Configuration

The Prometheus config in `k8s/monitoring/prometheus-config.yaml` (and `docker-compose.yml` volume mount) scrapes:

- The API application on port 8000
- The OTel collector on port 8888
- Qdrant on port 6333

### Useful PromQL Queries

Request rate (5-minute window):

```promql
rate(search_requests_total[5m])
```

P95 search latency:

```promql
histogram_quantile(0.95, rate(search_latency_seconds_bucket[5m]))
```

Cache hit ratio:

```promql
rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))
```

Error rate:

```promql
rate(http_server_request_duration_seconds_count{http_status_code=~"5.."}[5m])
```

---

## Grafana

### Access

Default URL: `http://localhost:3000`

Default credentials:
- Username: `admin`
- Password: `admin`

### Data Sources

Add Prometheus as a data source:
- URL: `http://prometheus:9090`
- Access: Server (default)

Add Jaeger as a data source:
- URL: `http://jaeger:16686`
- Access: Server (default)

### Recommended Dashboards

Create dashboards for the following views:

**Overview Dashboard**:
- Request rate (search, chat, upload, web-index)
- Error rate by endpoint
- P50, P90, P95 latency by endpoint
- Active connections

**Search Performance**:
- Search latency breakdown (embedding, vector search, reranking, LLM)
- Cache hit ratio over time
- Results per query distribution
- Modality breakdown (text vs image vs hybrid)

**Infrastructure**:
- CPU and memory usage per pod
- Qdrant collection sizes and query latency
- Redis hit rate and memory usage
- HPA replica count over time

---

## Jaeger Tracing

### Access

Default URL: `http://localhost:16686`

### Using Traces

1. Select service `multimodal-rag-engine` from the dropdown
2. Choose an operation (e.g., `POST /search`)
3. View the trace waterfall showing all spans

### Trace Context

Traces flow through:

```
HTTP Request
  +-- middleware (auth, rate-limit, timing)
  +-- endpoint handler
      +-- cache_lookup
      +-- embed_text / embed_image
      +-- qdrant_search
      +-- rerank
      +-- llm_generate (for chat)
      +-- cache_store
```

Each span includes attributes such as query text, result count, latency, and model names.

---

## Alerting

### Recommended Alert Rules

Configure in Grafana or Prometheus Alertmanager:

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Error Rate | 5xx rate > 1% for 5 minutes | Critical |
| High Latency | P95 search latency > 5s for 5 minutes | Warning |
| Qdrant Down | Qdrant health check fails for 2 minutes | Critical |
| High Memory | Container memory > 90% limit for 10 minutes | Warning |
| Cache Degraded | Redis connection errors > 0 for 5 minutes | Warning |
| LLM Errors | LLM error rate > 5% for 5 minutes | Warning |

---

## Production Tuning

### Sample Rate

In production, reduce the trace sample rate to control cost and volume:

```env
# Sample 10% of requests
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=0.1
```

### Metrics Retention

Configure Prometheus retention in the deployment:

```yaml
args:
  - "--storage.tsdb.retention.time=7d"
  - "--storage.tsdb.retention.size=5GB"
```

### Collector Pipeline

The OTel Collector config (`k8s/monitoring/otel-collector-config.yaml`) defines:

- **Receivers**: OTLP gRPC on port 4317
- **Processors**: Batch processor (reduces export overhead)
- **Exporters**: Prometheus (metrics), Jaeger (traces)
