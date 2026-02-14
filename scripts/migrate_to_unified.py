"""
Migration Script — re-index existing data from legacy collections to unified collection.

Usage:
    python -m scripts.migrate_to_unified [--dry-run] [--batch-size 32]

What it does:
  1. Reads all points from image_vectors and pdf_text_vectors collections.
  2. Re-embeds text/images through the unified embedder (Jina-CLIP v2).
  3. Adds VLM captions for images (if VLM enabled).
  4. Extracts entities for the knowledge graph.
  5. Upserts into the unified_vectors collection with enriched metadata.
  6. Does NOT delete legacy collections (kept for backward compat).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import get_settings
from utils.logger import setup_logging, get_logger

setup_logging(level="INFO")
_log = get_logger(__name__)


def migrate_text_chunks(dry_run: bool = False, batch_size: int = 32) -> int:
    """Re-embed and migrate text chunks to unified collection."""
    cfg = get_settings()

    from qdrant_client import QdrantClient
    from services.embedding_service.unified_embedder import create_unified_embedder
    from services.retrieval_service.hybrid_retriever import get_hybrid_retriever

    client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
    embedder = create_unified_embedder()
    hybrid = get_hybrid_retriever()

    # Check if source collection exists
    collections = [c.name for c in client.get_collections().collections]
    if cfg.pdf_text_collection not in collections:
        _log.info("no_text_collection", collection=cfg.pdf_text_collection)
        return 0

    info = client.get_collection(cfg.pdf_text_collection)
    total = info.points_count or 0
    _log.info("text_migration_start", total=total)

    if total == 0:
        return 0

    migrated = 0
    offset = None

    while True:
        # Scroll through existing points
        points, offset = client.scroll(
            collection_name=cfg.pdf_text_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # We'll re-embed
        )

        if not points:
            break

        texts = []
        point_ids = []
        payloads = []

        for point in points:
            payload = point.payload or {}
            text = payload.get("text", "")
            if not text:
                continue

            texts.append(text)
            # Create new deterministic ID
            id_str = f"unified:{payload.get('source_pdf', 'unknown')}:chunk:{payload.get('chunk_index', 0)}"
            point_id = int(hashlib.md5(id_str.encode()).hexdigest()[:15], 16)
            point_ids.append(point_id)

            # Enrich metadata
            enriched = dict(payload)
            enriched["modality"] = "text"
            enriched["migrated_from"] = cfg.pdf_text_collection
            payloads.append(enriched)

        if texts and not dry_run:
            # Re-embed with unified embedder
            vectors = embedder.encode_text_batch(texts)
            hybrid.upsert_unified_batch(point_ids, vectors, payloads)
            migrated += len(texts)
            _log.info("text_batch_migrated", count=len(texts), total=migrated)
        elif texts:
            migrated += len(texts)
            _log.info("text_batch_dry_run", count=len(texts), total=migrated)

        if offset is None:
            break

    return migrated


def migrate_images(dry_run: bool = False, batch_size: int = 16) -> int:
    """Re-embed and migrate images to unified collection."""
    cfg = get_settings()

    from PIL import Image
    from qdrant_client import QdrantClient
    from services.embedding_service.unified_embedder import create_unified_embedder
    from services.retrieval_service.hybrid_retriever import get_hybrid_retriever

    client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
    embedder = create_unified_embedder()
    hybrid = get_hybrid_retriever()

    collections = [c.name for c in client.get_collections().collections]
    if cfg.qdrant_collection not in collections:
        _log.info("no_image_collection", collection=cfg.qdrant_collection)
        return 0

    info = client.get_collection(cfg.qdrant_collection)
    total = info.points_count or 0
    _log.info("image_migration_start", total=total)

    if total == 0:
        return 0

    migrated = 0
    offset = None
    data_dir = Path(__file__).parent.parent / "data"

    while True:
        points, offset = client.scroll(
            collection_name=cfg.qdrant_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            file_path = payload.get("file_path", "")
            if not file_path:
                continue

            # Try to load the image
            full_path = Path(__file__).parent.parent / file_path
            if not full_path.exists():
                _log.debug("image_not_found", path=str(full_path))
                continue

            try:
                img = Image.open(full_path).convert("RGB")
            except Exception:
                continue

            if dry_run:
                migrated += 1
                continue

            # Re-embed with unified embedder
            vec = embedder.encode_image(img)

            # VLM caption
            caption = ""
            if cfg.vlm_enabled:
                try:
                    from services.vlm_service.local_vlm import get_vlm
                    vlm = get_vlm()
                    caption, conf = vlm.caption_image(img)
                    payload["caption"] = caption
                    payload["caption_confidence"] = conf
                except Exception:
                    pass

            # Create new ID
            id_str = f"unified:img:{file_path}"
            point_id = int(hashlib.md5(id_str.encode()).hexdigest()[:15], 16)

            # Enrich metadata
            payload["modality"] = "image"
            payload["migrated_from"] = cfg.qdrant_collection

            hybrid.upsert_unified_batch(
                [point_id], np.array([vec]), [payload]
            )
            migrated += 1

        _log.info("image_batch_migrated", total=migrated)

        if offset is None:
            break

    return migrated


def build_knowledge_graph(dry_run: bool = False) -> int:
    """Extract entities from all text chunks and build the graph."""
    cfg = get_settings()
    if not cfg.graph_enabled:
        _log.info("graph_disabled")
        return 0

    from qdrant_client import QdrantClient

    client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)

    # Read from unified collection (or text collection if no migration yet)
    source_collection = cfg.unified_collection
    collections = [c.name for c in client.get_collections().collections]
    if source_collection not in collections:
        source_collection = cfg.pdf_text_collection

    if source_collection not in collections:
        _log.info("no_source_collection")
        return 0

    from services.graph_service.entity_extractor import extract_entities_batch
    from services.graph_service.knowledge_graph import get_knowledge_graph

    graph = get_knowledge_graph()
    processed = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=source_collection,
            limit=16,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        texts = []
        sources = []
        for p in points:
            text = (p.payload or {}).get("text", "")
            source = (p.payload or {}).get("source_pdf", (p.payload or {}).get("source_url", ""))
            if text:
                texts.append(text)
                sources.append(source)

        if texts and not dry_run:
            # extract_entities_batch is async — run it properly
            import asyncio
            chunk_dicts = [
                {"text": t, "source": s, "chunk_id": f"migrate:{i}"}
                for i, (t, s) in enumerate(zip(texts, sources))
            ]
            entities_list = asyncio.run(extract_entities_batch(chunk_dicts))
            # Returns List[Tuple[List[str], List[Dict]]]
            for (entities, relationships), source in zip(entities_list, sources):
                if entities:
                    graph.add_entities(entities, source=source)
                if relationships:
                    graph.add_relationships(relationships)
            processed += len(texts)
            _log.info("entities_extracted", batch=len(texts), total=processed)
        elif texts:
            processed += len(texts)

        if offset is None:
            break

    if not dry_run:
        graph.save()
        _log.info("graph_saved", **graph.stats())

    return processed


def main():
    parser = argparse.ArgumentParser(description="Migrate to unified V2 collection")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--skip-text", action="store_true", help="Skip text migration")
    parser.add_argument("--skip-images", action="store_true", help="Skip image migration")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph building")
    args = parser.parse_args()

    start = time.time()
    _log.info("migration_start", dry_run=args.dry_run)

    text_count = 0
    image_count = 0
    entity_count = 0

    if not args.skip_text:
        text_count = migrate_text_chunks(args.dry_run, args.batch_size)

    if not args.skip_images:
        image_count = migrate_images(args.dry_run, args.batch_size // 2)

    if not args.skip_graph:
        entity_count = build_knowledge_graph(args.dry_run)

    elapsed = time.time() - start
    _log.info(
        "migration_complete",
        text_chunks=text_count,
        images=image_count,
        entities=entity_count,
        elapsed_s=round(elapsed, 1),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
