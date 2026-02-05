"""
In-process metrics collector for latency percentiles.

Why not Prometheus?
  - We're running local-first. No scrape target needed yet.
  - This gives us P50/P95/P99 in /stats without any infra.
  - When we scale, we swap this for a Prometheus histogram — same interface.

Thread-safety: Uses a deque with maxlen (atomic appends in CPython)
plus a lock for percentile computation.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from typing import Dict, Any


# Keep the last 10,000 observations per metric. Enough for percentiles,
# bounded memory regardless of request volume.
_WINDOW = 10_000


class MetricsCollector:
    """Process-global, thread-safe latency tracker."""

    def __init__(self) -> None:
        self._data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_WINDOW))
        self._counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._start_time = time.monotonic()

    def record(self, name: str, value_ms: float) -> None:
        """Record a latency observation in milliseconds."""
        self._data[name].append(value_ms)
        self._counts[name] += 1

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable dict of all metrics with percentiles.
        Called by the /stats endpoint.
        """
        with self._lock:
            result: Dict[str, Any] = {
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            }
            for name, values in self._data.items():
                vals = sorted(values)
                n = len(vals)
                if n == 0:
                    continue
                result[name] = {
                    "count": self._counts[name],
                    "window": n,
                    "p50_ms": round(vals[n // 2], 3),
                    "p95_ms": round(vals[int(n * 0.95)], 3) if n >= 20 else None,
                    "p99_ms": round(vals[int(n * 0.99)], 3) if n >= 100 else None,
                    "mean_ms": round(statistics.mean(vals), 3),
                    "min_ms": round(vals[0], 3),
                    "max_ms": round(vals[-1], 3),
                }
            return result


# Singleton — import this wherever you need metrics.
metrics = MetricsCollector()
