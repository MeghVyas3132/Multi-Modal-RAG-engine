"""
Upload endpoints — PDF, image, and web URL ingestion.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Set

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed

from services.api_gateway.models import PDFUploadResponse

_log = get_logger(__name__)
router = APIRouter(tags=["upload"])
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="upload")

_data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"

# Hold references to background tasks so they are not garbage-collected
_background_tasks: Set[asyncio.Task] = set()

# Max upload sizes
_MAX_IMAGE_SIZE = 20 * 1024 * 1024   # 20 MB
_MAX_PDF_SIZE = 100 * 1024 * 1024    # 100 MB


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent path traversal and
    other filesystem attacks.  Strips directory components and
    replaces dangerous characters.
    """
    # Take only the basename — strips ../../ and absolute paths
    filename = os.path.basename(filename)
    # Remove null bytes
    filename = filename.replace("\x00", "")
    # Whitelist: keep only alnum, dots, hyphens, underscores, spaces
    filename = re.sub(r"[^\w\s\-.]", "_", filename)
    # Collapse multiple dots/underscores
    filename = re.sub(r"\.{2,}", ".", filename)
    filename = filename.strip(". ")
    if not filename:
        filename = "upload"
    return filename


@router.post("/upload")
async def upload_and_index(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
):
    """Upload a single image, encode, and index."""
    from PIL import Image
    import io

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if len(contents) > _MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit")
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    safe_filename = _sanitize_filename(file.filename or "image.png")

    upload_dir = _data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / safe_filename
    save_path.write_bytes(contents)

    cfg = get_settings()
    loop = asyncio.get_event_loop()

    def _encode_and_upsert():
        path_str = str(save_path)
        point_id = int(hashlib.md5(path_str.encode()).hexdigest()[:15], 16)
        metadata = {
            "file_path": str(save_path.relative_to(_data_dir.parent)),
            "file_name": safe_filename,
            "file_size_bytes": len(contents),
            "width": img.width,
            "height": img.height,
            "modality": "image",
        }
        if category:
            metadata["category"] = category

        if cfg.unified_enabled:
            from services.embedding_service.unified_embedder import get_unified_embedder
            from services.retrieval_service.hybrid_retriever import get_hybrid_retriever
            import numpy as np

            embedder = get_unified_embedder()
            vec = embedder.encode_image(img)

            # VLM caption
            if cfg.vlm_enabled:
                try:
                    from services.vlm_service.local_vlm import get_vlm
                    vlm = get_vlm()
                    caption, confidence = vlm.caption_image(img)
                    metadata["caption"] = caption
                    metadata["caption_confidence"] = confidence
                except Exception as e:
                    _log.debug("vlm_caption_failed", error=str(e))

            hybrid = get_hybrid_retriever()
            hybrid.upsert_unified_batch(
                [point_id], np.array([vec]), [metadata]
            )
        else:
            from services.api_gateway.app import get_active_embedder
            from services.retrieval_service.retriever import get_retriever

            embedder = get_active_embedder()
            preprocessed = embedder.preprocess(img)
            vec = embedder.encode_images([preprocessed])
            retriever = get_retriever()
            retriever.upsert_batch([point_id], vec, [metadata])

        return point_id, metadata

    with timed("upload_index") as t:
        point_id, metadata = await loop.run_in_executor(_executor, _encode_and_upsert)

    return {
        "status": "indexed",
        "id": str(point_id),
        "metadata": metadata,
        "latency_ms": round(t["ms"], 2),
    }


@router.post("/upload/pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract text + images, embed and index.
    V2: uses semantic chunking, unified embeddings, VLM captioning, 
    entity extraction, and deduplication.
    """
    from services.pdf_service.parser import parse_pdf

    total_start = time.perf_counter_ns()
    cfg = get_settings()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    safe_filename = _sanitize_filename(file.filename)
    if not safe_filename.lower().endswith(".pdf"):
        safe_filename += ".pdf"

    contents = await file.read()
    if len(contents) > _MAX_PDF_SIZE:
        raise HTTPException(status_code=413, detail="PDF exceeds 100 MB limit")

    pdf_dir = Path(cfg.pdf_upload_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    save_path = pdf_dir / safe_filename
    save_path.write_bytes(contents)

    loop = asyncio.get_event_loop()

    def _process():
        result = parse_pdf(save_path)
        chunks_indexed = 0
        images_indexed = 0

        # ── Text chunking & indexing ────────────────────────
        if result.chunks:
            # Use semantic chunker for V2
            if cfg.chunking_strategy != "fixed":
                try:
                    from services.document_service.semantic_chunker import auto_chunk
                    # Get embedder for semantic similarity detection
                    embedder = None
                    if cfg.unified_enabled:
                        try:
                            from services.embedding_service.unified_embedder import get_unified_embedder
                            embedder = get_unified_embedder()
                        except Exception:
                            pass

                    full_text = "\n".join(c.text for c in result.chunks)
                    doc_chunks = auto_chunk(
                        full_text,
                        page_num=1,
                        source=safe_filename,
                        modality="text",
                        embedder=embedder,
                    )
                    texts = [dc.content for dc in doc_chunks]
                except Exception as e:
                    _log.warning("semantic_chunker_fallback", error=str(e))
                    texts = [c.text for c in result.chunks]
            else:
                texts = [c.text for c in result.chunks]

            # Deduplication
            if cfg.unified_enabled:
                from services.embedding_service.unified_embedder import get_unified_embedder
                embedder = get_unified_embedder()
                vectors = embedder.encode_text_batch(texts)

                # Dedup check
                try:
                    from services.cache_service.deduplication import DeduplicationService
                    dedup = DeduplicationService()
                    unique_indices = dedup.filter_duplicates(
                        texts, list(vectors), cfg.unified_collection
                    )
                    texts = [texts[i] for i in unique_indices]
                    vectors = vectors[unique_indices]
                except Exception:
                    pass

                ids = []
                payloads = []
                for i, text in enumerate(texts):
                    chunk_id_str = f"{safe_filename}:chunk:{i}"
                    point_id = int(
                        hashlib.md5(chunk_id_str.encode()).hexdigest()[:15], 16
                    )
                    ids.append(point_id)
                    payloads.append({
                        "text": text,
                        "source_pdf": safe_filename,
                        "chunk_index": i,
                        "modality": "text",
                        "type": "text_chunk",
                    })

                from services.retrieval_service.hybrid_retriever import get_hybrid_retriever
                hybrid = get_hybrid_retriever()
                import numpy as np
                hybrid.upsert_unified_batch(ids, vectors, payloads)
                chunks_indexed = len(ids)
            else:
                from services.embedding_service.text_embedder import create_text_embedder
                text_embedder = create_text_embedder()
                vectors = text_embedder.encode_batch(texts)

                ids = []
                payloads = []
                for i, text in enumerate(texts):
                    chunk_id_str = f"{safe_filename}:chunk:{i}"
                    point_id = int(
                        hashlib.md5(chunk_id_str.encode()).hexdigest()[:15], 16
                    )
                    ids.append(point_id)
                    payloads.append({
                        "text": text,
                        "source_pdf": safe_filename,
                        "chunk_index": i,
                        "type": "text_chunk",
                    })

                from services.retrieval_service.retriever import get_retriever
                retriever = get_retriever()
                retriever.upsert_text_batch(ids, vectors, payloads)
                chunks_indexed = len(ids)

            # ── Entity extraction deferred to async context ───
            # _extract_entities_background is async, so we collect texts
            # and call it after run_in_executor returns.

        # ── Image indexing ──────────────────────────────────
        if result.images:
            images_indexed = _index_pdf_images(
                result.images, safe_filename, cfg
            )

        return result, chunks_indexed, images_indexed, texts if result.chunks else []

    with timed("pdf_upload") as t:
        result, chunks_indexed, images_indexed, extracted_texts = await loop.run_in_executor(
            _executor, _process
        )

    # Fire-and-forget entity extraction (properly async, with task ref)
    if cfg.graph_enabled and extracted_texts:
        task = asyncio.create_task(
            _extract_entities_background(extracted_texts, safe_filename)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    total_ms = (time.perf_counter_ns() - total_start) / 1_000_000

    return PDFUploadResponse(
        status="indexed",
        filename=safe_filename,
        total_pages=result.total_pages,
        chunks_indexed=chunks_indexed,
        images_indexed=images_indexed,
        latency_ms=round(total_ms, 2),
        metadata=result.metadata,
    )


def _index_pdf_images(images, filename: str, cfg):
    """Embed + upsert PDF images into the index."""
    from PIL import Image as PILImage
    from services.pdf_service.parser import save_extracted_images
    import numpy as np

    saved_paths = save_extracted_images(
        images, _data_dir / "pdf_images", filename
    )
    indexed = 0

    for img_data, img_path in zip(images, saved_paths):
        try:
            pil_img = PILImage.open(img_path).convert("RGB")
            img_id_str = f"{filename}:img:{img_data.page_num}:{img_data.image_index}"
            point_id = int(
                hashlib.md5(img_id_str.encode()).hexdigest()[:15], 16
            )
            metadata = {
                "file_path": str(img_path.relative_to(_data_dir.parent)),
                "file_name": img_path.name,
                "source_pdf": filename,
                "page_num": img_data.page_num,
                "modality": "image",
                "type": "pdf_image",
            }

            if cfg.unified_enabled:
                from services.embedding_service.unified_embedder import get_unified_embedder
                embedder = get_unified_embedder()
                vec = embedder.encode_image(pil_img)

                if cfg.vlm_enabled:
                    try:
                        from services.vlm_service.local_vlm import get_vlm
                        vlm = get_vlm()
                        caption, conf = vlm.caption_image(pil_img)
                        metadata["caption"] = caption
                        metadata["caption_confidence"] = conf
                    except Exception:
                        pass

                from services.retrieval_service.hybrid_retriever import get_hybrid_retriever
                hybrid = get_hybrid_retriever()
                hybrid.upsert_unified_batch(
                    [point_id], np.array([vec]), [metadata]
                )
            else:
                from services.api_gateway.app import get_active_embedder
                from services.retrieval_service.retriever import get_retriever

                embedder = get_active_embedder()
                preprocessed = embedder.preprocess(pil_img)
                vec = embedder.encode_images([preprocessed])
                get_retriever().upsert_batch([point_id], vec, [metadata])

            indexed += 1
        except Exception as e:
            _log.warning("pdf_image_index_failed", error=str(e))

    return indexed


async def _extract_entities_background(texts, source):
    """Extract entities from chunks and add to knowledge graph."""
    try:
        from services.graph_service.entity_extractor import extract_entities_batch
        from services.graph_service.knowledge_graph import get_knowledge_graph

        # extract_entities_batch expects List[Dict] with 'text', 'source', 'chunk_id'
        chunk_dicts = [
            {"text": t, "source": source, "chunk_id": f"{source}:chunk:{i}"}
            for i, t in enumerate(texts)
        ]
        entities_list = await extract_entities_batch(chunk_dicts)
        graph = get_knowledge_graph()

        # extract_entities_batch returns List[Tuple[List[str], List[Dict]]]
        for entities, relationships in entities_list:
            if entities:
                graph.add_entities(entities, source=source)
            if relationships:
                graph.add_relationships(relationships)

        graph.save()
    except Exception as e:
        _log.debug("entity_extraction_failed", error=str(e))


@router.get("/pdfs")
async def list_pdfs():
    """List all uploaded PDFs."""
    cfg = get_settings()
    pdf_dir = Path(cfg.pdf_upload_dir)
    if not pdf_dir.exists():
        return {"pdfs": []}

    return {
        "pdfs": [
            {"filename": f.name, "size_bytes": f.stat().st_size}
            for f in sorted(pdf_dir.glob("*.pdf"))
        ]
    }
