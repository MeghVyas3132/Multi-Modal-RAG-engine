"""
PDF Parser Service — extract text and images from uploaded PDFs.

Architecture decisions:
  1. PyMuPDF (fitz) is used for extraction — it's 10x faster than
     pdfplumber and handles scanned PDFs better.
  2. Text is chunked with configurable size + overlap to ensure
     semantic coherence across chunk boundaries.
  3. Images are extracted at original resolution and saved as PNG
     for downstream CLIP encoding.
  4. Each chunk carries metadata (page_num, chunk_index, source_pdf)
     so the LLM can cite specific pages in its answer.
  5. Processing is synchronous — offloaded to ThreadPoolExecutor
     by the API layer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed

_log = get_logger(__name__)


@dataclass
class TextChunk:
    """A single chunk of text from a PDF page."""
    text: str
    page_num: int
    chunk_index: int
    source_pdf: str
    char_start: int = 0
    char_end: int = 0


@dataclass
class ExtractedImage:
    """An image extracted from a PDF page."""
    image_bytes: bytes
    page_num: int
    image_index: int
    source_pdf: str
    width: int = 0
    height: int = 0
    ext: str = "png"


@dataclass
class PDFParseResult:
    """Complete extraction result from a PDF."""
    filename: str
    total_pages: int
    chunks: List[TextChunk] = field(default_factory=list)
    images: List[ExtractedImage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()


def _chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    page_num: int,
    source_pdf: str,
) -> List[TextChunk]:
    """
    Split text into overlapping chunks of approximately chunk_size chars.
    Tries to break at sentence boundaries when possible.
    """
    if not text or len(text) < 20:
        return []

    chunks: List[TextChunk] = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at sentence boundary (. ! ?) if not at end
        if end < len(text):
            # Look backwards from end for a sentence break
            for boundary in range(end, max(start + chunk_size // 2, start), -1):
                if text[boundary - 1] in '.!?\n':
                    end = boundary
                    break

        chunk_text = text[start:end].strip()

        if len(chunk_text) >= 20:  # Skip very short chunks
            chunks.append(TextChunk(
                text=chunk_text,
                page_num=page_num,
                chunk_index=chunk_idx,
                source_pdf=source_pdf,
                char_start=start,
                char_end=end,
            ))
            chunk_idx += 1

        # Advance with overlap
        start = end - overlap if end < len(text) else len(text)

    return chunks


def parse_pdf(
    pdf_path: str | Path,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    extract_images: bool = True,
) -> PDFParseResult:
    """
    Extract text chunks and images from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Max characters per chunk (default from settings).
        chunk_overlap: Overlap between chunks (default from settings).
        extract_images: Whether to extract embedded images.

    Returns:
        PDFParseResult with chunks, images, and metadata.
    """
    cfg = get_settings()
    chunk_size = chunk_size or cfg.pdf_chunk_size
    chunk_overlap = chunk_overlap or cfg.pdf_chunk_overlap
    pdf_path = Path(pdf_path)

    _log.info("pdf_parse_begin", file=pdf_path.name)

    with timed("pdf_parse") as t:
        doc = fitz.open(str(pdf_path))

        result = PDFParseResult(
            filename=pdf_path.name,
            total_pages=len(doc),
            metadata={
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "page_count": len(doc),
                "file_size_bytes": pdf_path.stat().st_size,
            },
        )

        all_chunks: List[TextChunk] = []
        all_images: List[ExtractedImage] = []

        for page_num, page in enumerate(doc):
            # ── Extract text ────────────────────────────────
            raw_text = page.get_text("text")
            cleaned = _clean_text(raw_text)

            if cleaned:
                page_chunks = _chunk_text(
                    text=cleaned,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap,
                    page_num=page_num + 1,  # 1-indexed
                    source_pdf=pdf_path.name,
                )
                all_chunks.extend(page_chunks)

            # ── Extract images ──────────────────────────────
            if extract_images:
                image_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        # Convert CMYK to RGB
                        if pix.n > 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        # Skip tiny images (icons, bullets)
                        if pix.width < 50 or pix.height < 50:
                            continue

                        all_images.append(ExtractedImage(
                            image_bytes=pix.tobytes("png"),
                            page_num=page_num + 1,
                            image_index=img_idx,
                            source_pdf=pdf_path.name,
                            width=pix.width,
                            height=pix.height,
                        ))
                    except Exception as e:
                        _log.warning(
                            "pdf_image_extract_failed",
                            page=page_num + 1,
                            xref=xref,
                            error=str(e),
                        )

        doc.close()

        result.chunks = all_chunks
        result.images = all_images

        # Re-index chunk_index globally (not per-page)
        for i, chunk in enumerate(result.chunks):
            chunk.chunk_index = i

    _log.info(
        "pdf_parse_complete",
        file=pdf_path.name,
        pages=result.total_pages,
        chunks=len(result.chunks),
        images=len(result.images),
        parse_ms=round(t["ms"], 2),
    )

    return result


def save_extracted_images(
    images: List[ExtractedImage],
    output_dir: str | Path,
    pdf_name: str,
) -> List[Path]:
    """
    Save extracted images to disk for CLIP indexing.

    Returns:
        List of saved file paths.
    """
    output_dir = Path(output_dir) / pdf_name.replace(".pdf", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for img in images:
        filename = f"page{img.page_num}_img{img.image_index}.{img.ext}"
        path = output_dir / filename
        path.write_bytes(img.image_bytes)
        saved_paths.append(path)

    _log.info("pdf_images_saved", count=len(saved_paths), dir=str(output_dir))
    return saved_paths
