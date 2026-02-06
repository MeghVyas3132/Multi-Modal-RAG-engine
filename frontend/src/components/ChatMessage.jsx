import React, { useEffect, useState } from 'react';
import clsx from 'clsx';
import { User, FileText, Image as ImageIcon, Clock, Zap, AlertCircle, BookOpen, Loader2 } from 'lucide-react';

/**
 * Image Result Card - displays a single search result
 */
const ImageResultCard = ({ result, onClick }) => {
    const [imageError, setImageError] = useState(false);
    const [imageLoaded, setImageLoaded] = useState(false);

    return (
        <div 
            className="group relative rounded-xl overflow-hidden border border-gray-200 bg-white shadow-sm hover:shadow-md hover:border-gray-300 transition-all cursor-pointer"
            onClick={() => onClick?.(result)}
        >
            {/* Image */}
            <div className="aspect-square bg-gray-100 relative overflow-hidden">
                {!imageError ? (
                    <img
                        src={result.imageUrl}
                        alt={result.metadata?.file_name || 'Search result'}
                        className={clsx(
                            "w-full h-full object-cover transition-all duration-300",
                            imageLoaded ? "opacity-100" : "opacity-0"
                        )}
                        onLoad={() => setImageLoaded(true)}
                        onError={() => setImageError(true)}
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center bg-gray-100">
                        <ImageIcon className="w-8 h-8 text-gray-300" />
                    </div>
                )}
                
                {/* Loading skeleton */}
                {!imageLoaded && !imageError && (
                    <div className="absolute inset-0 bg-gray-200 animate-pulse" />
                )}

                {/* Score badge */}
                <div className="absolute top-2 right-2 bg-black/70 text-white text-[10px] font-bold px-2 py-1 rounded-full">
                    {Math.round(result.score * 100)}%
                </div>
            </div>

            {/* Metadata */}
            <div className="p-2 bg-white border-t border-gray-100">
                <div className="text-[11px] text-gray-600 font-medium truncate">
                    {result.metadata?.file_name || 'Unknown'}
                </div>
                {result.metadata?.category && (
                    <div className="text-[10px] text-gray-400 uppercase tracking-wider mt-0.5">
                        {result.metadata.category}
                    </div>
                )}
            </div>
        </div>
    );
};

/**
 * Image Results Grid - displays search results in a responsive grid
 */
const ImageResultsGrid = ({ results, latencyMs, cached, query }) => {
    const [selectedImage, setSelectedImage] = useState(null);

    if (!results || results.length === 0) {
        return (
            <div className="flex items-center gap-2 text-gray-500 text-sm">
                <AlertCircle className="w-4 h-4" />
                <span>No images found</span>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {/* Stats bar */}
            <div className="flex items-center gap-4 text-[11px] text-gray-500">
                <div className="flex items-center gap-1">
                    <Zap className="w-3 h-3" />
                    <span>{latencyMs}ms</span>
                </div>
                <div className="flex items-center gap-1">
                    <ImageIcon className="w-3 h-3" />
                    <span>{results.length} results</span>
                </div>
                {cached && (
                    <div className="flex items-center gap-1 text-green-600">
                        <Clock className="w-3 h-3" />
                        <span>Cached</span>
                    </div>
                )}
            </div>

            {/* Image grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {results.map((result) => (
                    <ImageResultCard 
                        key={result.id} 
                        result={result}
                        onClick={setSelectedImage}
                    />
                ))}
            </div>

            {/* Lightbox modal */}
            {selectedImage && (
                <div 
                    className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
                    onClick={() => setSelectedImage(null)}
                >
                    <div className="relative max-w-4xl max-h-[90vh] rounded-xl overflow-hidden bg-white shadow-2xl">
                        <img
                            src={selectedImage.imageUrl}
                            alt={selectedImage.metadata?.file_name || 'Selected image'}
                            className="max-w-full max-h-[80vh] object-contain"
                        />
                        <div className="p-4 bg-white border-t">
                            <div className="flex items-center justify-between">
                                <div>
                                    <div className="font-bold text-gray-900">
                                        {selectedImage.metadata?.file_name}
                                    </div>
                                    <div className="text-sm text-gray-500">
                                        Score: {Math.round(selectedImage.score * 100)}% match
                                    </div>
                                </div>
                                {selectedImage.metadata?.width && (
                                    <div className="text-sm text-gray-400">
                                        {selectedImage.metadata.width} x {selectedImage.metadata.height}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

const AttachmentItem = ({ file }) => {
    const [previewUrl, setPreviewUrl] = useState(null);
    const isImage = file.type?.startsWith('image/') || (file.name && /\.(jpg|jpeg|png|gif|webp)$/i.test(file.name));

    useEffect(() => {
        // Use previewUrl from file if available (for already processed attachments)
        if (file.previewUrl) {
            setPreviewUrl(file.previewUrl);
            return;
        }
        
        if (isImage && file instanceof File) {
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
            return () => URL.revokeObjectURL(url);
        }
    }, [file, isImage]);

    if (previewUrl) {
        return (
            <div className="rounded-xl overflow-hidden border border-gray-200 shadow-sm bg-white hover:border-gray-300 transition-all cursor-zoom-in">
                <img
                    src={previewUrl}
                    alt={file.name}
                    className="w-full h-auto max-h-[400px] object-contain block"
                />
                <div className="px-3 py-2 bg-gray-50 border-t border-gray-100 flex items-center gap-2">
                    <ImageIcon className="w-3.5 h-3.5 text-blue-500" />
                    <span className="text-[11px] text-gray-600 font-bold truncate">{file.name}</span>
                </div>
            </div>
        );
    }

    return (
        <div className="flex items-center gap-3 bg-white px-4 py-3 rounded-xl border border-gray-200 shadow-sm hover:border-gray-300 transition-all">
            <div className="w-10 h-10 rounded-lg bg-red-50 flex items-center justify-center flex-shrink-0">
                <FileText className="w-5 h-5 text-red-500" />
            </div>
            <div className="flex flex-col overflow-hidden">
                <span className="text-xs text-gray-900 font-bold truncate">{file.name}</span>
                <span className="text-[10px] text-gray-400 uppercase font-black tracking-tighter">Document</span>
            </div>
        </div>
    );
};

/**
 * Text Results Section - displays retrieved PDF text chunks
 */
const TextResultsSection = ({ results, latencyMs }) => {
    if (!results || results.length === 0) return null;

    return (
        <div className="space-y-2">
            <div className="flex items-center gap-2 text-[11px] text-gray-500">
                <BookOpen className="w-3 h-3" />
                <span>{results.length} text chunks retrieved</span>
                {latencyMs && (
                    <>
                        <span>·</span>
                        <Zap className="w-3 h-3" />
                        <span>{latencyMs}ms</span>
                    </>
                )}
            </div>
            <div className="space-y-2 max-h-60 overflow-y-auto">
                {results.slice(0, 3).map((result, idx) => (
                    <div key={idx} className="text-[12px] text-gray-600 bg-white border border-gray-100 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-1">
                            <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                                Page {result.metadata?.page_num || '?'} · {result.metadata?.source_pdf || 'PDF'}
                            </span>
                            <span className="text-[10px] text-gray-300">
                                {Math.round(result.score * 100)}% match
                            </span>
                        </div>
                        <p className="line-clamp-3">{result.metadata?.text || ''}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

/**
 * Streaming cursor - animated blinking cursor for active streaming
 */
const StreamingCursor = () => (
    <span className="inline-block w-2 h-5 bg-black/70 ml-0.5 animate-pulse" />
);

const ChatMessage = ({ message }) => {
    const isUser = message.role === 'user';
    const isError = message.type === 'error';
    const hasImageResults = (message.type === 'image_results' || message.type === 'rag_response' || message.type === 'streaming') 
        && message.searchResults?.length > 0;
    const hasTextResults = (message.type === 'rag_response' || message.type === 'streaming')
        && message.textResults?.length > 0;
    const isStreaming = message.isStreaming;
    const isPdfIndexed = message.type === 'pdf_indexed';

    // Determine assistant label
    const assistantLabel = isPdfIndexed ? 'PDF Processor' 
        : (message.type === 'rag_response' || message.type === 'streaming') ? 'RAG Assistant'
        : 'Image Search';

    return (
        <div className={clsx(
            "flex w-full px-4 py-6 md:px-8",
            isUser ? "bg-white" : "bg-[#f9f9f9]",
            isError && "bg-red-50"
        )}>
            <div className="max-w-4xl mx-auto flex gap-4 w-full text-left">
                {/* Avatar */}
                <div className={clsx(
                    "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 shadow-sm transition-all",
                    isUser ? "bg-white border border-gray-200" : isError ? "bg-red-500" : isPdfIndexed ? "bg-emerald-600" : "bg-black"
                )}>
                    {isUser ? (
                        <User className="w-4 h-4 text-gray-500" />
                    ) : isError ? (
                        <AlertCircle className="w-4 h-4 text-white" />
                    ) : isPdfIndexed ? (
                        <FileText className="w-4 h-4 text-white" />
                    ) : (
                        <div className="w-2 h-2 bg-white rounded-sm" />
                    )}
                </div>

                {/* Content */}
                <div className="flex-1 space-y-3 overflow-hidden">
                    <div className="font-bold text-[14px] text-gray-900 flex items-center gap-2">
                        {isUser ? "You" : assistantLabel}
                        {isStreaming && (
                            <Loader2 className="w-3.5 h-3.5 text-gray-400 animate-spin" />
                        )}
                    </div>

                    {/* Attachments if any (for user messages) */}
                    {message.attachments && message.attachments.length > 0 && (
                        <div className="flex flex-wrap gap-3 mb-2">
                            {message.attachments.map((file, idx) => (
                                <div key={idx} className="flex flex-col gap-2 max-w-full sm:max-w-sm">
                                    <AttachmentItem file={file} />
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Retrieved text chunks (shown before LLM answer) */}
                    {hasTextResults && (
                        <TextResultsSection
                            results={message.textResults}
                            latencyMs={message.latencyMs}
                        />
                    )}

                    {/* Text content (LLM answer or status) */}
                    {message.content && (
                        <div className={clsx(
                            "leading-relaxed whitespace-pre-wrap text-[15px] font-medium max-w-none",
                            isError ? "text-red-700" : "text-gray-800"
                        )}>
                            {message.content}
                            {isStreaming && <StreamingCursor />}
                        </div>
                    )}

                    {/* Streaming placeholder when no content yet */}
                    {isStreaming && !message.content && !hasTextResults && (
                        <div className="flex items-center gap-2 text-gray-400 text-sm">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>Thinking...</span>
                        </div>
                    )}

                    {/* Image search results grid */}
                    {hasImageResults && (
                        <ImageResultsGrid 
                            results={message.searchResults}
                            latencyMs={message.latencyMs}
                            cached={message.cached}
                            query={message.query}
                        />
                    )}
                </div>
            </div>
        </div>
    );
};

export default ChatMessage;
