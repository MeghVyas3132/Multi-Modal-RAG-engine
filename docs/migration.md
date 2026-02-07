# Migration Guide: V1 to V2

This document covers migrating from the V1 architecture (separate CLIP collections) to the V2 architecture (unified Jina-CLIP embeddings, knowledge graph, semantic cache, and VLM).

---

## Overview of Changes

### V1 Architecture

- **Embedding**: OpenCLIP ViT-B-32 (512-dimensional vectors)
- **Collections**: `image_vectors` and `pdf_text_vectors` (separate)
- **Search**: Single-modal (text-to-image OR text-to-text, never cross-modal)
- **LLM**: Single provider
- **Caching**: None
- **Knowledge Graph**: None
- **VLM**: None

### V2 Architecture

- **Embedding**: Jina-CLIP v2 (768-dimensional vectors) with unified cross-modal space
- **Collections**: `unified_vectors` (single collection for all modalities), legacy collections retained
- **Search**: Cross-modal hybrid retrieval with reranking
- **LLM**: Cerebras primary, Groq fallback
- **Caching**: Redis L2 + semantic vector cache with deduplication
- **Knowledge Graph**: NetworkX-based entity graph with LLM extraction
- **VLM**: SmolVLM-500M-Instruct with GPT-4o-mini fallback

---

## Migration Script

The migration script re-embeds all existing data through the V2 pipeline and populates the unified collection.

### Location

```
scripts/migrate_to_unified.py
```

### What It Does

1. **Text Migration**: Scrolls through all points in `pdf_text_vectors`, re-embeds each text chunk using Jina-CLIP v2, and upserts into `unified_vectors` with enriched metadata including `modality: text` and `migrated_from` fields.

2. **Image Migration**: Scrolls through all points in `image_vectors`, loads the original image file from disk, re-embeds using Jina-CLIP v2, optionally generates a VLM caption, and upserts into `unified_vectors` with `modality: image`.

3. **Knowledge Graph Building**: Reads all text chunks from the unified collection (or text collection if migration has not run yet), extracts entities and relationships using the LLM-based entity extractor, builds a NetworkX graph, and saves to disk.

### What It Does Not Do

- It does NOT delete legacy collections. They are preserved for backward compatibility.
- It does NOT modify existing points in legacy collections.
- It does NOT re-process data that has already been migrated (idempotent via deterministic IDs).

---

## Running the Migration

### Prerequisites

Before running the migration:

1. Ensure Qdrant is running and accessible.
2. Ensure the `.env` file has valid API keys (at minimum `CEREBRAS_API_KEY` if graph building is enabled).
3. Ensure original image files are still available at their original paths (for image re-embedding).

### Dry Run

Preview what will be migrated without making changes:

```bash
python -m scripts.migrate_to_unified --dry-run
```

Output shows the number of text chunks and images that would be processed.

### Full Migration

```bash
# Default (batch size 32 for text, 16 for images)
python -m scripts.migrate_to_unified

# Custom batch size
python -m scripts.migrate_to_unified --batch-size 64

# Skip specific steps
python -m scripts.migrate_to_unified --skip-images
python -m scripts.migrate_to_unified --skip-graph
python -m scripts.migrate_to_unified --skip-text

# Using Makefile
make migrate
```

### Selective Migration

You can skip individual phases:

| Flag | Effect |
|------|--------|
| `--skip-text` | Skip text chunk re-embedding |
| `--skip-images` | Skip image re-embedding |
| `--skip-graph` | Skip knowledge graph construction |
| `--dry-run` | Preview only, no writes |
| `--batch-size N` | Set text batch size (image batch is half) |

---

## Migration Process Detail

### Text Chunks

For each batch of text chunks from `pdf_text_vectors`:

1. Read point payload (text, source_pdf, chunk_index, page_number, etc.)
2. Generate a deterministic point ID: `md5("unified:{source_pdf}:chunk:{chunk_index}")[:15]`
3. Re-embed text using `UnifiedEmbedder.encode_text_batch()`
4. Add metadata: `modality=text`, `migrated_from=pdf_text_vectors`
5. Upsert into `unified_vectors` via `HybridRetriever.upsert_unified_batch()`

### Images

For each image point in `image_vectors`:

1. Read point payload (file_path, topic, etc.)
2. Load the original image file from disk
3. Re-embed using `UnifiedEmbedder.encode_image()`
4. If VLM is enabled, generate a caption and store it in metadata
5. Generate deterministic ID: `md5("unified:img:{file_path}")[:15]`
6. Add metadata: `modality=image`, `migrated_from=image_vectors`
7. Upsert into `unified_vectors`

### Knowledge Graph

After data migration:

1. Scroll through all text chunks in the unified collection
2. Extract entities and relationships using `extract_entities_batch()` (LLM-based)
3. Add to NetworkX graph via `KnowledgeGraph.add_entities()` and `add_relationships()`
4. Save graph to disk as JSON

---

## Post-Migration Verification

After migration completes, verify the results:

### Check Collection Counts

```bash
# Using the API health endpoint
curl http://localhost:8000/health | python -m json.tool

# Or check Qdrant directly
curl http://localhost:6333/collections/unified_vectors | python -m json.tool
```

The `unified_vectors` collection should contain the sum of text chunks and images from legacy collections.

### Check Knowledge Graph

```bash
curl http://localhost:8000/graph/stats | python -m json.tool
```

Should show non-zero entity and relationship counts.

### Test Cross-Modal Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your test query", "top_k": 5}'
```

Results should include both text chunks and images ranked together.

### Validate Legacy Collections

Legacy collections remain untouched:

```bash
curl http://localhost:6333/collections/image_vectors | python -m json.tool
curl http://localhost:6333/collections/pdf_text_vectors | python -m json.tool
```

---

## Rollback

If migration needs to be reverted:

1. Delete the unified collection:

```bash
curl -X DELETE http://localhost:6333/collections/unified_vectors
```

2. Remove the knowledge graph file:

```bash
rm data/knowledge_graph.json
```

3. Set `ENABLE_UNIFIED_EMBEDDINGS=false` in `.env` to fall back to legacy collections.

The system will revert to V1 behavior using the original `image_vectors` and `pdf_text_vectors` collections.

---

## Performance Expectations

| Dataset Size | Text Migration | Image Migration | Graph Building | Total |
|-------------|---------------|----------------|----------------|-------|
| 100 chunks, 50 images | ~30 seconds | ~1 minute | ~2 minutes | ~3.5 minutes |
| 1,000 chunks, 500 images | ~3 minutes | ~10 minutes | ~15 minutes | ~28 minutes |
| 10,000 chunks, 2,000 images | ~30 minutes | ~1 hour | ~2 hours | ~3.5 hours |

Times are approximate and depend on hardware, model loading, and LLM API latency for entity extraction.

### Optimization Tips

- Run text and image migration first (`--skip-graph`), verify results, then run graph building separately.
- Use a larger batch size for text migration if memory allows.
- If VLM captioning is slow, disable it during migration and add captions later via the upload endpoint.
- Ensure GPU is available for faster embedding (if applicable).
