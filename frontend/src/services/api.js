/**
 * API Service - Backend Integration
 * 
 * Connects to the Multi-Modal RAG backend for:
 * - Text-to-image search
 * - Image-to-image similarity search
 * - Image upload and indexing
 * - PDF upload and indexing
 * - RAG chat with SSE streaming
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Search images by text query
 * @param {string} query - Natural language search query
 * @param {object} options - Search options
 * @returns {Promise<SearchResponse>}
 */
export async function searchByText(query, options = {}) {
    const {
        topK = 10,
        scoreThreshold = 0.0,
        filters = null,
        includeExplanation = false
    } = options;

    const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query,
            top_k: topK,
            score_threshold: scoreThreshold,
            filters,
            include_explanation: includeExplanation
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Search failed' }));
        throw new Error(error.detail || 'Search failed');
    }

    return response.json();
}

/**
 * Search images by uploading an image file
 * @param {File} imageFile - Image file to search with
 * @param {object} options - Search options
 * @returns {Promise<ImageSearchResponse>}
 */
export async function searchByImage(imageFile, options = {}) {
    const {
        topK = 10,
        scoreThreshold = 0.0
    } = options;

    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('top_k', topK.toString());
    formData.append('score_threshold', scoreThreshold.toString());

    const response = await fetch(`${API_BASE_URL}/search/image`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Image search failed' }));
        throw new Error(error.detail || 'Image search failed');
    }

    return response.json();
}

/**
 * Upload and index an image
 * @param {File} imageFile - Image file to upload
 * @param {string} category - Optional category tag
 * @returns {Promise<UploadResponse>}
 */
export async function uploadImage(imageFile, category = null) {
    const formData = new FormData();
    formData.append('file', imageFile);
    if (category) {
        formData.append('category', category);
    }

    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
}

/**
 * Check backend health status
 * @returns {Promise<HealthResponse>}
 */
export async function checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
        throw new Error('Health check failed');
    }

    return response.json();
}

/**
 * Get backend statistics
 * @returns {Promise<StatsResponse>}
 */
export async function getStats() {
    const response = await fetch(`${API_BASE_URL}/stats`);
    
    if (!response.ok) {
        throw new Error('Stats fetch failed');
    }

    return response.json();
}

/**
 * Get the full URL for an image from metadata
 * @param {object} metadata - Image metadata from search result
 * @returns {string} Full image URL
 */
export function getImageUrl(metadata) {
    // If it's an S3 URL, return as-is (for production)
    if (metadata.s3_key && metadata.s3_key.startsWith('http')) {
        return metadata.s3_key;
    }
    
    // For local development, construct URL from file path
    if (metadata.file_path) {
        // Extract relative path from file_path
        // Handles both "data/test_subset/..." and "/abs/path/data/test_subset/..."
        const relativePath = metadata.file_path
            .replace(/\\/g, '/')
            .replace(/^(.*\/)?data\//, '');
        return `${API_BASE_URL}/images/${relativePath}`;
    }

    // Fallback to file_name in uploads directory
    if (metadata.file_name) {
        return `${API_BASE_URL}/images/uploads/${metadata.file_name}`;
    }

    return '';
}

/**
 * Upload and index a PDF file
 * @param {File} pdfFile - PDF file to upload
 * @param {function} onProgress - Optional progress callback
 * @returns {Promise<PDFUploadResponse>}
 */
export async function uploadPdf(pdfFile, onProgress = null) {
    const formData = new FormData();
    formData.append('file', pdfFile);

    const response = await fetch(`${API_BASE_URL}/upload/pdf`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'PDF upload failed' }));
        throw new Error(error.detail || 'PDF upload failed');
    }

    return response.json();
}

/**
 * Stream RAG chat response via Server-Sent Events
 * @param {string} query - User question
 * @param {object} options - Chat options
 * @param {function} onRetrieval - Callback for Phase 1 retrieval results
 * @param {function} onToken - Callback for each LLM token
 * @param {function} onDone - Callback when streaming completes
 * @param {function} onError - Callback on error
 * @returns {Promise<void>}
 */
export async function streamChat(query, options = {}, { onRetrieval, onToken, onDone, onError } = {}) {
    const {
        topK = 5,
        includeImages = true,
        scoreThreshold = 0.2,
        filters = null,
    } = options;

    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query,
            top_k: topK,
            include_images: includeImages,
            score_threshold: scoreThreshold,
            filters,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Chat request failed' }));
        throw new Error(error.detail || 'Chat request failed');
    }

    // Parse SSE stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';  // Keep incomplete last line in buffer

        let currentEvent = null;
        for (const line of lines) {
            if (line.startsWith('event:')) {
                currentEvent = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
                const dataStr = line.slice(5).trim();
                if (!dataStr) continue;

                try {
                    const data = JSON.parse(dataStr);

                    switch (currentEvent) {
                        case 'retrieval':
                            onRetrieval?.(data);
                            break;
                        case 'token':
                            onToken?.(data.token);
                            break;
                        case 'done':
                            onDone?.(data);
                            break;
                        case 'error':
                            onError?.(data.error);
                            break;
                        default:
                            break;
                    }
                } catch (e) {
                    // Skip malformed JSON
                }
                currentEvent = null;
            }
        }
    }
}

/**
 * List uploaded PDFs
 * @returns {Promise<{pdfs: Array}>}
 */
export async function listPdfs() {
    const response = await fetch(`${API_BASE_URL}/pdfs`);
    if (!response.ok) {
        throw new Error('Failed to list PDFs');
    }
    return response.json();
}


// ── V2: Web Ingestion ──────────────────────────────────────

/**
 * Index a web URL — scrapes, chunks, embeds, indexes.
 * @param {string} url - URL to index
 * @param {object} options - Options
 * @returns {Promise<WebIndexResponse>}
 */
export async function indexUrl(url, options = {}) {
    const { recursive = false, maxPages = 1 } = options;

    const response = await fetch(`${API_BASE_URL}/web/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            url,
            recursive,
            max_pages: maxPages,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'URL indexing failed' }));
        throw new Error(error.detail || 'URL indexing failed');
    }

    return response.json();
}

/**
 * Search with web grounding — falls back to web if local results are poor.
 * @param {string} query - Search query
 * @param {number} threshold - Quality threshold
 * @returns {Promise<object>}
 */
export async function searchWithGrounding(query, threshold = 0.65) {
    const response = await fetch(
        `${API_BASE_URL}/web/search-grounding?query=${encodeURIComponent(query)}&threshold=${threshold}`,
        { method: 'POST' }
    );

    if (!response.ok) {
        throw new Error('Web grounding search failed');
    }

    return response.json();
}


// ── V2: Knowledge Graph ────────────────────────────────────

/**
 * Get knowledge graph statistics
 * @returns {Promise<object>}
 */
export async function getGraphStats() {
    const response = await fetch(`${API_BASE_URL}/graph/stats`);
    if (!response.ok) throw new Error('Graph stats failed');
    return response.json();
}

/**
 * Find entities related to a given entity
 * @param {string} entity - Entity to look up
 * @param {number} maxHops - Graph traversal depth
 * @returns {Promise<object>}
 */
export async function findRelated(entity, maxHops = 2) {
    const params = new URLSearchParams({ entity, max_hops: maxHops });
    const response = await fetch(`${API_BASE_URL}/graph/related?${params}`);
    if (!response.ok) throw new Error('Graph query failed');
    return response.json();
}

/**
 * Expand a query with graph entities (for debugging)
 * @param {string} query - Query to expand
 * @returns {Promise<object>}
 */
export async function expandQuery(query) {
    const params = new URLSearchParams({ query });
    const response = await fetch(`${API_BASE_URL}/graph/expand?${params}`);
    if (!response.ok) throw new Error('Query expansion failed');
    return response.json();
}


export default {
    searchByText,
    searchByImage,
    uploadImage,
    uploadPdf,
    streamChat,
    listPdfs,
    checkHealth,
    getStats,
    getImageUrl,
    indexUrl,
    searchWithGrounding,
    getGraphStats,
    findRelated,
    expandQuery,
    API_BASE_URL,
};
