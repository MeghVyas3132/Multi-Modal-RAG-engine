/**
 * API Service - Backend Integration
 * 
 * Connects to the Multi-Modal RAG backend for:
 * - Text-to-image search
 * - Image-to-image similarity search
 * - Image upload and indexing
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

export default {
    searchByText,
    searchByImage,
    uploadImage,
    checkHealth,
    getStats,
    getImageUrl,
    API_BASE_URL
};
