# API Reference

Complete endpoint documentation for the Multi-Modal RAG Engine V2.

Base URL: `http://localhost:8000`

Interactive documentation: `http://localhost:8000/docs` (Swagger UI)

---

## Search

### POST /search

Text-to-image/text semantic search with optional hybrid retrieval, graph expansion, and reranking.

**Request Body:**

```json
{
  "query": "chocolate cake with frosting",
  "top_k": 10,
  "score_threshold": 0.2,
  "filters": null,
  "include_explanation": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| query | string | required | Natural language search query |
| top_k | integer | 10 | Maximum number of results to return |
| score_threshold | float | 0.2 | Minimum similarity score |
| filters | object | null | Optional metadata filters |
| include_explanation | boolean | false | Include score explanation |

**Response:**

```json
{
  "query": "chocolate cake with frosting",
  "results": [
    {
      "id": 123456,
      "score": 0.87,
      "metadata": {
        "file_path": "data/images/chocolate_cake/104030.jpg",
        "file_name": "104030.jpg",
        "width": 512,
        "height": 512,
        "modality": "image"
      }
    }
  ],
  "total": 5,
  "latency_ms": 86.2,
  "cached": false
}
```

### POST /search/image

Image-to-image similarity search. Upload an image to find visually similar images.

**Request:** `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| file | file | required | Image file (JPEG, PNG, WebP) |
| top_k | integer | 10 | Maximum results |
| score_threshold | float | 0.0 | Minimum similarity |

**Response:** Same format as `/search`.

```bash
curl -X POST http://localhost:8000/search/image \
  -F "file=@photo.jpg" \
  -F "top_k=5"
```

---

## Chat

### POST /chat

RAG chat with Server-Sent Events (SSE) streaming. Retrieves relevant context from the index and streams an LLM-generated answer.

**Request Body:**

```json
{
  "query": "What is the process of photosynthesis?",
  "top_k": 5,
  "include_images": true,
  "score_threshold": 0.2,
  "filters": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| query | string | required | User question |
| top_k | integer | 5 | Number of retrieval results for context |
| include_images | boolean | true | Include image results in context |
| score_threshold | float | 0.2 | Minimum retrieval score |
| filters | object | null | Optional metadata filters |

**Response:** SSE stream with three event types:

```
event: retrieval
data: {"results": [...], "total": 5, "latency_ms": 45.2}

event: token
data: {"token": "Photo"}

event: token
data: {"token": "synthesis"}

event: token
data: {"token": " is"}

event: done
data: {"total_tokens": 256, "latency_ms": 1200}
```

Error event:

```
event: error
data: {"error": "LLM service unavailable"}
```

---

## Upload

### POST /upload

Upload and index a single image.

**Request:** `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| file | file | required | Image file |
| category | string | null | Optional category tag |

**Response:**

```json
{
  "status": "indexed",
  "point_id": 123456789,
  "metadata": {
    "file_path": "data/uploads/photo.jpg",
    "file_name": "photo.jpg",
    "width": 1024,
    "height": 768,
    "modality": "image"
  }
}
```

### POST /upload/pdf

Upload and index a PDF document. Extracts text chunks and images, embeds them, and stores in the vector index.

**Request:** `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| file | file | required | PDF file |

**Response:**

```json
{
  "status": "indexed",
  "filename": "document.pdf",
  "total_pages": 42,
  "chunks_indexed": 156,
  "images_indexed": 8,
  "latency_ms": 3200.5,
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "page_count": 42
  }
}
```

### GET /pdfs

List all uploaded PDF files.

**Response:**

```json
{
  "pdfs": [
    {"filename": "document.pdf", "size_bytes": 2456789},
    {"filename": "textbook.pdf", "size_bytes": 15234567}
  ]
}
```

---

## Web

### POST /web/index

Scrape a URL, chunk the content, embed, and index into the vector store. Supports web pages, YouTube videos (transcript), and GitHub repositories.

**Request Body:**

```json
{
  "url": "https://en.wikipedia.org/wiki/Photosynthesis",
  "recursive": false,
  "max_pages": 1
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| url | string | required | URL to scrape and index |
| recursive | boolean | false | Crawl linked pages |
| max_pages | integer | 1 | Maximum pages to crawl |

**Response:**

```json
{
  "status": "indexed",
  "url": "https://en.wikipedia.org/wiki/Photosynthesis",
  "title": "Photosynthesis - Wikipedia",
  "source_type": "web",
  "chunks_indexed": 23,
  "latency_ms": 4500.2
}
```

### POST /web/search-grounding

If local retrieval quality is below the threshold, falls back to web search for grounding.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| query | string | required | Search query |
| threshold | float | 0.65 | Quality threshold (below this triggers web fallback) |

**Response (local results sufficient):**

```json
{
  "grounded": true,
  "source": "local",
  "results": [...]
}
```

**Response (web fallback):**

```json
{
  "grounded": true,
  "source": "web",
  "content": "Extracted web content...",
  "title": "Page Title"
}
```

---

## Knowledge Graph

### GET /graph/stats

Return knowledge graph statistics.

**Response:**

```json
{
  "enabled": true,
  "nodes": 1245,
  "edges": 3456,
  "entity_types": {
    "concept": 456,
    "person": 234,
    "organization": 123
  }
}
```

### GET /graph/related

Find entities related to a given entity in the graph.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| entity | string | required | Entity name to look up |
| max_hops | integer | 2 | Maximum graph traversal depth |
| max_results | integer | 20 | Maximum related entities |

**Response:**

```json
{
  "entity": "photosynthesis",
  "related": [
    {"entity": "chloroplast", "relation": "occurs_in", "hops": 1},
    {"entity": "ATP", "relation": "produces", "hops": 1},
    {"entity": "cellular respiration", "relation": "related_to", "hops": 2}
  ]
}
```

### GET /graph/expand

Expand a query with graph entities. Useful for debugging what the graph-augmented retriever will do.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| query | string | required | Query to expand |

**Response:**

```json
{
  "query": "How does photosynthesis work?",
  "matched_entities": ["photosynthesis"],
  "expansion_terms": ["chloroplast", "light reactions", "Calvin cycle", "ATP"]
}
```

### POST /graph/save

Persist the knowledge graph to disk.

**Response:**

```json
{"status": "saved"}
```

---

## System

### GET /health

Liveness and readiness probe. Returns the status of all subsystems.

**Response:**

```json
{
  "status": "healthy",
  "clip_model_loaded": true,
  "text_model_loaded": true,
  "qdrant_connected": true,
  "redis_connected": true,
  "device": "cpu",
  "collection_count": 25000,
  "unified_embedder_loaded": true,
  "vlm_loaded": true,
  "graph_loaded": true
}
```

### GET /stats

Runtime performance statistics.

**Response:**

```json
{
  "total_requests": 1500,
  "uptime_seconds": 3600,
  "latency_percentiles": {
    "p50_ms": 45.2,
    "p95_ms": 120.5,
    "p99_ms": 250.1
  },
  "embedding_latency": {
    "p50_ms": 12.3,
    "p95_ms": 35.6
  },
  "collection_info": {
    "image_vectors": {"count": 15000},
    "pdf_text_vectors": {"count": 8000},
    "unified_vectors": {"count": 23000}
  },
  "cache_stats": {
    "l1_size": 245,
    "l2_connected": true,
    "l3_vectors": 1200
  },
  "graph_stats": {
    "nodes": 1245,
    "edges": 3456
  }
}
```
