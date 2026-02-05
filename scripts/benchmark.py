"""
Latency benchmark — validate the <50ms target.

Sends N search requests to the running API and reports P50/P95/P99.
Use this after indexing to verify end-to-end performance.

Usage:
    python -m scripts.benchmark --num-requests 1000 --concurrency 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Sample queries that exercise different CLIP text encoder paths
_QUERIES = [
    "a dog playing in the snow",
    "modern skyscraper at sunset",
    "close up of a red flower",
    "people walking in a busy city",
    "a plate of sushi",
    "vintage car on a highway",
    "abstract painting with bright colors",
    "soccer player scoring a goal",
    "mountain landscape with lake",
    "cat sleeping on a keyboard",
    "underwater coral reef with fish",
    "astronaut floating in space",
    "steam train crossing a bridge",
    "child reading a book in a library",
    "neon signs in Tokyo at night",
]


def run_benchmark(
    base_url: str = "http://localhost:8000",
    num_requests: int = 100,
    top_k: int = 10,
) -> None:
    """Run sequential search requests and measure latency."""
    import httpx

    client = httpx.Client(base_url=base_url, timeout=30.0)

    # Warm up — first request may have cold-cache overhead
    _warm = client.post("/search", json={"query": "warmup", "top_k": 1})
    print(f"Warmup status: {_warm.status_code}")

    latencies = []
    errors = 0

    print(f"\nRunning {num_requests} search requests...")
    for i in range(num_requests):
        query = _QUERIES[i % len(_QUERIES)]
        payload = {"query": query, "top_k": top_k}

        start = time.perf_counter_ns()
        try:
            resp = client.post("/search", json=payload)
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

            if resp.status_code == 200:
                body = resp.json()
                server_ms = body.get("latency_ms", 0)
                latencies.append(elapsed_ms)
            else:
                errors += 1
                print(f"  FAIL Request {i}: status {resp.status_code}")
        except Exception as e:
            errors += 1
            print(f"  FAIL Request {i}: {e}")

    client.close()

    if not latencies:
        print("No successful requests!")
        return

    latencies.sort()
    n = len(latencies)

    print(f"\n{'='*50}")
    print(f" Benchmark Results ({n} requests)")
    print(f"{'='*50}")
    print(f"  Successful: {n}")
    print(f"  Errors:     {errors}")
    print(f"  Top-K:      {top_k}")
    print(f"")
    print(f"  Min:        {latencies[0]:.2f} ms")
    print(f"  P50:        {latencies[n // 2]:.2f} ms")
    print(f"  P90:        {latencies[int(n * 0.90)]:.2f} ms")
    print(f"  P95:        {latencies[int(n * 0.95)]:.2f} ms")
    print(f"  P99:        {latencies[int(n * 0.99)]:.2f} ms")
    print(f"  Max:        {latencies[-1]:.2f} ms")
    print(f"  Mean:       {statistics.mean(latencies):.2f} ms")
    print(f"  Stdev:      {statistics.stdev(latencies):.2f} ms" if n > 1 else "")
    print(f"{'='*50}")

    target = 50.0
    p95 = latencies[int(n * 0.95)]
    if p95 <= target:
        print(f"  PASS P95 ({p95:.1f}ms) is UNDER target ({target}ms)")
    else:
        print(f"  WARN P95 ({p95:.1f}ms) is OVER target ({target}ms)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark search latency")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--num-requests", "-n", type=int, default=100)
    parser.add_argument("--top-k", "-k", type=int, default=10)
    args = parser.parse_args()
    run_benchmark(args.url, args.num_requests, args.top_k)


if __name__ == "__main__":
    main()
