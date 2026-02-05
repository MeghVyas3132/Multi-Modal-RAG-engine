"""
Structured logging — consistent, machine-parseable, fast.

Why structlog?
  - JSON output for production log aggregation (ELK, Datadog, etc.)
  - Human-readable console output for local dev.
  - Zero-cost when log level is above threshold (bound loggers).
  - Thread-safe out of the box.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """
    Call once at process startup. Configures both stdlib logging
    and structlog in one shot.
    """
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named, bound logger. Use this everywhere."""
    return structlog.get_logger(name)
