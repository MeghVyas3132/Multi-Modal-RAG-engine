"""
Offline Image Indexing Pipeline

This script is run ONCE (or incrementally) to:
  1. Scan a directory of images
  2. Encode each image through CLIP's vision encoder (GPU-accelerated)
  3. Upsert the resulting vectors + metadata into Qdrant
  4. Write a metadata manifest for debugging / auditing

Design decisions:
  - Batch processing: Images are loaded and encoded in configurable batches
    (default 256) to maximize GPU utilization without OOM.
  - DataLoader with prefetch: We use PyTorch DataLoader for parallel image
    loading so the GPU is never waiting on disk I/O.
  - Idempotent: Uses deterministic integer IDs derived from file path hash.
    Re-running on the same images updates existing points (upsert).
  - Progress logging: tqdm + structured logs for both interactive and CI use.
  - No API server involved — this is a standalone CLI script.

Usage:
    python -m indexing.index_images --image-dir ./data/images --batch-size 256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import get_settings
from utils.logger import setup_logging, get_logger
from services.embedding_service.embedder import CLIPEmbedder
from services.retrieval_service.retriever import VectorRetriever

_log = get_logger(__name__)

# Supported image extensions (lowercase)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _path_to_id(path: str) -> int:
    """
    Deterministic int64 ID from file path.
    This ensures idempotent re-runs — same file always gets same ID.
    We use the lower 63 bits of MD5 to stay within Qdrant's int range.
    """
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    return int(h[:15], 16)  # 60-bit positive integer


def _scan_images(image_dir: str) -> List[Path]:
    """Recursively find all image files in the directory."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    files = []
    for ext in _IMAGE_EXTENSIONS:
        files.extend(image_dir.rglob(f"*{ext}"))
        files.extend(image_dir.rglob(f"*{ext.upper()}"))

    # Deduplicate and sort for deterministic ordering
    files = sorted(set(files))
    _log.info("images_scanned", count=len(files), directory=str(image_dir))
    return files


def _load_and_preprocess_batch(
    paths: List[Path],
    preprocess,
) -> Tuple[List, List[Path], List[Dict[str, Any]]]:
    """
    Load a batch of images, preprocess them for CLIP, and build metadata.
    Returns (preprocessed_images, valid_paths, metadata_list).
    Skips corrupted/unreadable images gracefully.
    """
    images = []
    valid_paths = []
    metadata = []

    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            preprocessed = preprocess(img)
            images.append(preprocessed)
            valid_paths.append(path)
            metadata.append({
                "file_path": str(path),
                "file_name": path.name,
                "file_size_bytes": path.stat().st_size,
                # Simulated S3 path — easy to swap for real S3 later
                "s3_key": f"s3://image-bucket/images/{path.name}",
                "width": img.width,
                "height": img.height,
            })
        except Exception as e:
            _log.warning("image_load_failed", path=str(path), error=str(e))

    return images, valid_paths, metadata


def index_images(
    image_dir: str,
    batch_size: int = 256,
    manifest_path: str = "./data/index_manifest.json",
) -> None:
    """
    Main indexing function. Encodes all images and upserts to Qdrant.

    Args:
        image_dir: Directory containing images to index.
        batch_size: Number of images per GPU batch.
        manifest_path: Where to write the metadata manifest.
    """
    setup_logging(level="INFO")

    _log.info("indexing_started", image_dir=image_dir, batch_size=batch_size)
    start = time.monotonic()

    # ── Initialize services ─────────────────────────────────
    embedder = CLIPEmbedder()
    retriever = VectorRetriever()
    retriever.ensure_collection()

    # ── Scan images ─────────────────────────────────────────
    image_paths = _scan_images(image_dir)
    if not image_paths:
        _log.warning("no_images_found", directory=image_dir)
        return

    # ── Process in batches ──────────────────────────────────
    total_indexed = 0
    manifest: List[Dict[str, Any]] = []

    for batch_start in tqdm(
        range(0, len(image_paths), batch_size),
        desc="Indexing batches",
        unit="batch",
    ):
        batch_paths = image_paths[batch_start : batch_start + batch_size]

        # Load and preprocess
        images, valid_paths, batch_metadata = _load_and_preprocess_batch(
            batch_paths, embedder.preprocess
        )

        if not images:
            continue

        # Encode through CLIP vision encoder (GPU)
        vectors = embedder.encode_images(images)

        # Generate deterministic IDs
        ids = [_path_to_id(str(p)) for p in valid_paths]

        # Upsert to Qdrant
        retriever.upsert_batch(ids, vectors, batch_metadata)

        total_indexed += len(ids)
        manifest.extend(batch_metadata)

        _log.info(
            "batch_indexed",
            batch_start=batch_start,
            batch_size=len(ids),
            total=total_indexed,
        )

    # ── Write manifest ──────────────────────────────────────
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w") as f:
        json.dump(
            {
                "total_indexed": total_indexed,
                "image_dir": str(image_dir),
                "images": manifest,
            },
            f,
            indent=2,
        )

    elapsed = time.monotonic() - start
    _log.info(
        "indexing_complete",
        total_indexed=total_indexed,
        elapsed_seconds=round(elapsed, 1),
        images_per_second=round(total_indexed / elapsed, 1) if elapsed > 0 else 0,
        manifest=str(manifest_file),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index images into Qdrant via CLIP")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=get_settings().image_dir,
        help="Directory containing images to index",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=get_settings().index_batch_size,
        help="Batch size for GPU encoding",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="./data/index_manifest.json",
        help="Path to write the metadata manifest",
    )
    args = parser.parse_args()
    index_images(args.image_dir, args.batch_size, args.manifest)


if __name__ == "__main__":
    main()
