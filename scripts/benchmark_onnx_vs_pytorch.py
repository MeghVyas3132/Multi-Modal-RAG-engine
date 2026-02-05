"""
Benchmark: ONNX Runtime vs PyTorch inference latency.

Runs side-by-side comparison of CLIP text encoding using both
backends, measuring P50/P95/P99 latency and memory usage.

Usage:
  python -m scripts.benchmark_onnx_vs_pytorch
  python -m scripts.benchmark_onnx_vs_pytorch --iterations 1000 --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np


# ── Test queries ────────────────────────────────────────────
TEST_QUERIES = [
    "a cat sitting on a windowsill",
    "sunset over the ocean with dramatic clouds",
    "a person riding a bicycle through a city",
    "abstract painting with bright colors and geometric shapes",
    "a red sports car on a mountain highway",
    "snow-covered mountain peaks at sunrise",
    "a golden retriever playing fetch in a park",
    "city skyline at night with reflections on water",
    "fresh fruit and vegetables on a wooden table",
    "astronaut floating in space above Earth",
    "a cozy coffee shop with bookshelves",
    "tropical beach with palm trees and clear water",
    "vintage camera on a leather surface",
    "cherry blossoms in a Japanese garden",
    "thunderstorm over a wheat field",
]


def percentile(arr: List[float], pct: float) -> float:
    """Calculate the pct-th percentile of a sorted list."""
    idx = int(len(arr) * pct / 100)
    idx = min(idx, len(arr) - 1)
    return arr[idx]


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def benchmark_pytorch(
    model_name: str,
    pretrained: str,
    queries: List[str],
    iterations: int,
) -> Tuple[List[float], float]:
    """
    Benchmark PyTorch CLIP text encoding.

    Returns:
        (latency_ms_list, memory_mb)
    """
    import torch
    import open_clip

    print("  Loading PyTorch model...")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device="cpu"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    mem_before = get_process_memory_mb()

    # Warmup
    for _ in range(5):
        tokens = tokenizer(["warmup"]).to("cpu")
        with torch.no_grad():
            out = model.encode_text(tokens)
            out = out / out.norm(dim=-1, keepdim=True)

    # Benchmark
    latencies: List[float] = []
    for i in range(iterations):
        query = queries[i % len(queries)]
        tokens = tokenizer([query]).to("cpu")

        start = time.perf_counter_ns()
        with torch.no_grad():
            out = model.encode_text(tokens)
            out = out / out.norm(dim=-1, keepdim=True)
            _ = out.squeeze(0).cpu().numpy()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        latencies.append(elapsed_ms)

    mem_after = get_process_memory_mb()
    model_mem = mem_after - mem_before

    # Release model memory
    del model
    gc.collect()

    return latencies, model_mem


def benchmark_onnx(
    onnx_path: str,
    model_name: str,
    queries: List[str],
    iterations: int,
    intra_threads: int = 4,
    inter_threads: int = 2,
) -> Tuple[List[float], float]:
    """
    Benchmark ONNX Runtime CLIP text encoding.

    Returns:
        (latency_ms_list, memory_mb)
    """
    import onnxruntime as ort
    import open_clip

    print("  Loading ONNX model...")
    tokenizer = open_clip.get_tokenizer(model_name)

    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = intra_threads
    session_opts.inter_op_num_threads = inter_threads
    session_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    mem_before = get_process_memory_mb()

    session = ort.InferenceSession(
        onnx_path,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Warmup
    for _ in range(5):
        tokens = tokenizer(["warmup"]).numpy().astype(np.int64)
        session.run([output_name], {input_name: tokens})

    # Benchmark
    latencies: List[float] = []
    for i in range(iterations):
        query = queries[i % len(queries)]
        tokens = tokenizer([query]).numpy().astype(np.int64)

        start = time.perf_counter_ns()
        outputs = session.run([output_name], {input_name: tokens})
        _ = outputs[0].squeeze(0).astype(np.float32)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        latencies.append(elapsed_ms)

    mem_after = get_process_memory_mb()
    model_mem = mem_after - mem_before

    # Release session
    del session
    gc.collect()

    return latencies, model_mem


def print_results(
    pt_latencies: List[float],
    ort_latencies: List[float],
    pt_mem: float,
    ort_mem: float,
) -> None:
    """Print formatted comparison table."""
    pt_sorted = sorted(pt_latencies)
    ort_sorted = sorted(ort_latencies)

    print(f"\n{'='*70}")
    print(f"  ONNX vs PyTorch Benchmark Results ({len(pt_latencies)} iterations)")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<20} {'PyTorch (ms)':>15} {'ONNX (ms)':>15} {'Speedup':>10}")
    print(f"  {'-'*60}")

    metrics_list = [
        ("P50", 50),
        ("P90", 90),
        ("P95", 95),
        ("P99", 99),
    ]

    for name, pct in metrics_list:
        pt_val = round(percentile(pt_sorted, pct), 2)
        ort_val = round(percentile(ort_sorted, pct), 2)
        speedup = pt_val / max(ort_val, 0.01)
        print(f"  {name:<20} {pt_val:>15.2f} {ort_val:>15.2f} {speedup:>9.1f}x")

    pt_mean = round(sum(pt_latencies) / len(pt_latencies), 2)
    ort_mean = round(sum(ort_latencies) / len(ort_latencies), 2)
    print(f"  {'Mean':<20} {pt_mean:>15.2f} {ort_mean:>15.2f} {pt_mean/max(ort_mean, 0.01):>9.1f}x")
    print(f"  {'Min':<20} {round(pt_sorted[0], 2):>15.2f} {round(ort_sorted[0], 2):>15.2f}")
    print(f"  {'Max':<20} {round(pt_sorted[-1], 2):>15.2f} {round(ort_sorted[-1], 2):>15.2f}")

    if pt_mem > 0 and ort_mem > 0:
        print(f"\n  {'Memory (MB)':<20} {pt_mem:>15.1f} {ort_mem:>15.1f} {pt_mem/max(ort_mem, 1):>9.1f}x")

    print(f"\n{'='*70}")


def save_csv(
    output_path: str,
    pt_latencies: List[float],
    ort_latencies: List[float],
) -> None:
    """Save raw latencies to CSV for further analysis."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "pytorch_ms", "onnx_ms"])
        for i, (pt, ort_val) in enumerate(zip(pt_latencies, ort_latencies)):
            writer.writerow([i, round(pt, 4), round(ort_val, 4)])
    print(f"  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX vs PyTorch CLIP inference")
    parser.add_argument("--model", default="ViT-H-14", help="OpenCLIP model name")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k", help="Pretrained weights")
    parser.add_argument("--onnx-path", default="models/onnx/clip_vit_h14_text_fp32.onnx", help="ONNX model path")
    parser.add_argument("--iterations", type=int, default=200, help="Number of benchmark iterations")
    parser.add_argument("--output", default=None, help="CSV output path (optional)")
    parser.add_argument("--intra-threads", type=int, default=4, help="ONNX intra-op threads")
    parser.add_argument("--inter-threads", type=int, default=2, help="ONNX inter-op threads")
    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        print(f"ERROR: ONNX model not found at {args.onnx_path}")
        print(f"Run 'python -m scripts.convert_to_onnx' first.")
        sys.exit(1)

    queries = TEST_QUERIES

    print("\n[1/2] Benchmarking PyTorch backend...")
    pt_latencies, pt_mem = benchmark_pytorch(
        args.model, args.pretrained, queries, args.iterations
    )

    print("\n[2/2] Benchmarking ONNX Runtime backend...")
    ort_latencies, ort_mem = benchmark_onnx(
        args.onnx_path, args.model, queries, args.iterations,
        intra_threads=args.intra_threads,
        inter_threads=args.inter_threads,
    )

    print_results(pt_latencies, ort_latencies, pt_mem, ort_mem)

    if args.output:
        save_csv(args.output, pt_latencies, ort_latencies)


if __name__ == "__main__":
    main()
