"""
OpenTelemetry instrumentation -- traces, metrics, and log correlation.

Architecture decisions:
  1. TracerProvider with OTLP gRPC exporter for distributed traces.
     Spans propagate through embedding, retrieval, and cache layers.
  2. MeterProvider with OTLP exporter + Prometheus-compatible histograms.
     Grafana reads these via the OTel Collector's Prometheus exporter.
  3. Head-based sampling at configurable rate (default 100% for dev,
     lower in production to control trace volume).
  4. FastAPI auto-instrumentation captures HTTP spans (method, path,
     status, latency) with zero application code changes.
  5. This module is imported conditionally (cfg.otel_enabled) so
     there is zero overhead when telemetry is disabled.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from opentelemetry import trace, metrics as otel_metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from configs.settings import get_settings
from utils.logger import get_logger

_log = get_logger(__name__)

# Module-level references set once by setup_telemetry()
_tracer: Optional[trace.Tracer] = None
_meter: Optional[otel_metrics.Meter] = None

# Pre-created instruments (populated after setup)
_search_latency_histogram = None
_embedding_latency_histogram = None
_retrieval_latency_histogram = None
_cache_hit_counter = None
_cache_miss_counter = None
_search_request_counter = None
_error_counter = None


def setup_telemetry(app: FastAPI) -> None:
    """
    Initialize OpenTelemetry SDK and instrument the FastAPI app.
    Called once at startup from the lifespan hook.
    """
    global _tracer, _meter
    global _search_latency_histogram, _embedding_latency_histogram
    global _retrieval_latency_histogram, _cache_hit_counter, _cache_miss_counter
    global _search_request_counter, _error_counter

    cfg = get_settings()

    resource = Resource.create({
        SERVICE_NAME: cfg.otel_service_name,
        SERVICE_VERSION: "1.0.0",
        "deployment.environment": "production",
    })

    # ── Tracing ─────────────────────────────────────────────
    sampler = TraceIdRatioBased(cfg.otel_sample_rate)
    tracer_provider = TracerProvider(resource=resource, sampler=sampler)

    span_exporter = OTLPSpanExporter(endpoint=cfg.otel_endpoint, insecure=True)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            span_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
        )
    )
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer("clip-image-search", "1.0.0")

    # ── Metrics ─────────────────────────────────────────────
    metric_exporter = OTLPMetricExporter(endpoint=cfg.otel_endpoint, insecure=True)
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter, export_interval_millis=15000
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    otel_metrics.set_meter_provider(meter_provider)
    _meter = otel_metrics.get_meter("clip-image-search", "1.0.0")

    # ── Create instruments ──────────────────────────────────
    _search_latency_histogram = _meter.create_histogram(
        name="search.latency",
        description="End-to-end search latency in milliseconds",
        unit="ms",
    )
    _embedding_latency_histogram = _meter.create_histogram(
        name="embedding.latency",
        description="CLIP text encoding latency in milliseconds",
        unit="ms",
    )
    _retrieval_latency_histogram = _meter.create_histogram(
        name="retrieval.latency",
        description="Qdrant vector search latency in milliseconds",
        unit="ms",
    )
    _cache_hit_counter = _meter.create_counter(
        name="cache.hits",
        description="Number of Redis cache hits",
    )
    _cache_miss_counter = _meter.create_counter(
        name="cache.misses",
        description="Number of Redis cache misses",
    )
    _search_request_counter = _meter.create_counter(
        name="search.requests",
        description="Total search requests",
    )
    _error_counter = _meter.create_counter(
        name="errors.total",
        description="Total error count by type",
    )

    # ── Instrument FastAPI ──────────────────────────────────
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health,docs,openapi.json",
    )

    _log.info(
        "otel_initialized",
        endpoint=cfg.otel_endpoint,
        sample_rate=cfg.otel_sample_rate,
        service=cfg.otel_service_name,
    )


# ── Convenience accessors ───────────────────────────────────

def get_tracer() -> Optional[trace.Tracer]:
    """Get the global OTel tracer (None if not initialized)."""
    return _tracer


def get_meter() -> Optional[otel_metrics.Meter]:
    """Get the global OTel meter (None if not initialized)."""
    return _meter


def record_search_latency(ms: float, query: str = "") -> None:
    """Record end-to-end search latency."""
    if _search_latency_histogram:
        _search_latency_histogram.record(ms, {"query_length": len(query)})


def record_embedding_latency(ms: float, backend: str = "pytorch") -> None:
    """Record CLIP embedding latency."""
    if _embedding_latency_histogram:
        _embedding_latency_histogram.record(ms, {"backend": backend})


def record_retrieval_latency(ms: float, top_k: int = 10) -> None:
    """Record Qdrant search latency."""
    if _retrieval_latency_histogram:
        _retrieval_latency_histogram.record(ms, {"top_k": top_k})


def record_cache_hit() -> None:
    """Increment cache hit counter."""
    if _cache_hit_counter:
        _cache_hit_counter.add(1)


def record_cache_miss() -> None:
    """Increment cache miss counter."""
    if _cache_miss_counter:
        _cache_miss_counter.add(1)


def record_search_request(cached: bool = False) -> None:
    """Increment total search request counter."""
    if _search_request_counter:
        _search_request_counter.add(1, {"cached": str(cached)})


def record_error(error_type: str) -> None:
    """Increment error counter by type."""
    if _error_counter:
        _error_counter.add(1, {"error_type": error_type})


def create_span(name: str, attributes: Optional[dict] = None):
    """
    Create a new OTel span. Returns a context manager.

    Usage:
        with create_span("clip.encode_text", {"query": text}) as span:
            result = encode(text)
            span.set_attribute("vector_dim", 1024)
    """
    if _tracer:
        return _tracer.start_as_current_span(name, attributes=attributes or {})

    # Return a no-op context manager when OTel is disabled
    from contextlib import nullcontext
    return nullcontext()
