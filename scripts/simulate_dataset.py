"""
Simulate 1M image vectors and upsert them into Qdrant.

Why simulate?
  - You probably don't have 1M images lying around.
  - This lets us validate HNSW search latency, memory usage, and
    system behavior at scale WITHOUT the CLIP model or real images.
  - The vectors are random unit-norm float32 — identical distribution
    to real CLIP embeddings for ANN benchmarking purposes.

Usage:
    python -m scripts.simulate_dataset --num-vectors 1000000 --batch-size 10000

On a machine with 8GB RAM + Qdrant in Docker:
  - 1M * 1024 dim * 4 bytes = ~4GB raw vectors
  - With int8 quantization = ~1GB in Qdrant
  - Total Qdrant memory: ~2GB (vectors + HNSW graph + payloads)
  - Indexing time: ~5-10 minutes for 1M vectors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import get_settings
from services.retrieval_service.retriever import VectorRetriever
from utils.logger import setup_logging, get_logger

_log = get_logger(__name__)


def _generate_batch(
    batch_size: int,
    vector_dim: int,
    start_id: int,
) -> tuple:
    """
    Generate a batch of random unit-norm vectors with fake metadata.
    
    The vectors are drawn from N(0,1) and L2-normalized, which matches
    the distribution of CLIP embeddings on the unit hypersphere.
    """
    vectors = np.random.randn(batch_size, vector_dim).astype(np.float32)
    # L2 normalize to unit sphere (same as CLIP output)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    ids = list(range(start_id, start_id + batch_size))

    # Simulated metadata — mirrors what real indexing would produce
    categories = ["nature", "architecture", "people", "animals", "food", "vehicles", "art", "sports"]
    payloads = [
        {
            "file_path": f"/data/images/img_{id_:08d}.jpg",
            "file_name": f"img_{id_:08d}.jpg",
            "s3_key": f"s3://image-bucket/images/img_{id_:08d}.jpg",
            "category": categories[id_ % len(categories)],
            "width": 640,
            "height": 480,
            "file_size_bytes": 150_000,
            "simulated": True,
        }
        for id_ in ids
    ]

    return ids, vectors, payloads


def simulate(num_vectors: int = 1_000_000, batch_size: int = 10_000) -> None:
    """
    Generate and upsert `num_vectors` random vectors into Qdrant.
    """
    setup_logging(level="INFO")
    cfg = get_settings()

    _log.info(
        "simulation_started",
        num_vectors=num_vectors,
        batch_size=batch_size,
        vector_dim=cfg.clip_vector_dim,
    )

    start = time.monotonic()

    # Connect to Qdrant and ensure collection
    retriever = VectorRetriever()
    retriever.ensure_collection()

    total_upserted = 0

    for batch_start in tqdm(
        range(0, num_vectors, batch_size),
        desc="Simulating vectors",
        unit="batch",
    ):
        current_batch_size = min(batch_size, num_vectors - batch_start)
        ids, vectors, payloads = _generate_batch(
            current_batch_size, cfg.clip_vector_dim, batch_start
        )

        retriever.upsert_batch(ids, vectors, payloads)
        total_upserted += current_batch_size

        if total_upserted % 100_000 == 0:
            _log.info("simulation_progress", upserted=total_upserted, target=num_vectors)

    elapsed = time.monotonic() - start

    _log.info(
        "simulation_complete",
        total_upserted=total_upserted,
        elapsed_seconds=round(elapsed, 1),
        vectors_per_second=round(total_upserted / elapsed, 0) if elapsed > 0 else 0,
    )

    # Verify
    info = retriever.collection_info()
    _log.info("collection_status", **info)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate 1M image vectors in Qdrant")
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=1_000_000,
        help="Number of vectors to generate (default: 1M)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Batch size for Qdrant upsert (default: 10K)",
    )
    args = parser.parse_args()
    simulate(args.num_vectors, args.batch_size)


if __name__ == "__main__":
    main()
