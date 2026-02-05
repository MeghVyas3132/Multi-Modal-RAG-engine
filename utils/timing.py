"""
Precision timing utilities for latency instrumentation.

Design decision: We use time.perf_counter_ns() (monotonic, nanosecond)
instead of time.time() because we need sub-millisecond accuracy and
monotonicity guarantees. Wall-clock time can jump on NTP sync.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from utils.logger import get_logger

_log = get_logger(__name__)


@contextmanager
def timed(label: str) -> Generator[dict, None, None]:
    """
    Context manager that measures elapsed time in milliseconds.

    Usage:
        with timed("clip_encode") as t:
            vector = model.encode(text)
        print(t["ms"])  # e.g. 4.32

    The dict is populated *after* the block finishes, so you can
    read t["ms"] or t["ns"] after the `with` block.
    """
    result: dict = {}
    start = time.perf_counter_ns()
    try:
        yield result
    finally:
        elapsed_ns = time.perf_counter_ns() - start
        result["ns"] = elapsed_ns
        result["ms"] = elapsed_ns / 1_000_000
        _log.info(f"{label}", latency_ms=round(result["ms"], 3))
