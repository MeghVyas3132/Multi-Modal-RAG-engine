import React, { useEffect, useState } from 'react';
import clsx from 'clsx';
import { User, FileText, Image as ImageIcon, Clock, Zap, AlertCircle, BookOpen, Loader2 } from 'lucide-react';

/**
 * Image Result Card — soft card with rounded corners and hover lift
 */
const ImageResultCard = ({ result, onClick }) => {
    const [imageError, setImageError] = useState(false);
    const [imageLoaded, setImageLoaded] = useState(false);

    return (
        <div 
            className="group relative rounded-2xl overflow-hidden border border-gray-100 bg-white shadow-soft hover:shadow-soft-md hover:border-gray-200/80 transition-all duration-300 cursor-pointer hover:-translate-y-0.5"
            onClick={() => onClick?.(result)}
        >
            <div className="aspect-square bg-surface-100 relative overflow-hidden">
                {!imageError ? (
                    <img
                        src={result.imageUrl}
                        alt={result.metadata?.file_name || 'Search result'}
                        className={clsx(
                            "w-full h-full object-cover transition-all duration-500",
                            imageLoaded ? "opacity-100 scale-100" : "opacity-0 scale-105"
                        )}
                        onLoad={() => setImageLoaded(true)}
                        onError={() => setImageError(true)}
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center bg-surface-100">
                        <ImageIcon className="w-8 h-8 text-gray-300" />
                    </div>
                )}
                {!imageLoaded && !imageError && (
                    <div className="absolute inset-0 bg-surface-200 animate-pulse rounded-2xl" />
                )}
                <div className="absolute top-2 right-2 bg-white/80 backdrop-blur-sm text-gray-700 text-[10px] font-semibold px-2 py-0.5 rounded-full shadow-soft">
                    {Math.round(result.score * 100)}%
                </div>
            </div>
            <div className="p-2.5 bg-white">
                <div className="text-[11px] text-gray-500 font-medium truncate">
                    {result.metadata?.file_name || 'Unknown'}
                </div>
                {result.metadata?.category && (
                    <div className="text-[10px] text-gray-400 mt-0.5">
                        {result.metadata.category}
                    </div>
                )}
            </div>
        </div>
    );
};

/**
 * Image Results Grid
 */
const ImageResultsGrid = ({ results, latencyMs, cached, query }) => {
    const [selectedImage, setSelectedImage] = useState(null);

    if (!results || results.length === 0) {
        return (
            <div className="flex items-center gap-2 text-gray-400 text-[13px]">
                <AlertCircle className="w-4 h-4" />
                <span>No images found</span>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            <div className="flex items-center gap-3 text-[11px] text-gray-400">
                <div className="flex items-center gap-1">
                    <Zap className="w-3 h-3" />
                    <span>{latencyMs}ms</span>
                </div>
                <div className="flex items-center gap-1">
                    <ImageIcon className="w-3 h-3" />
                    <span>{results.length} results</span>
                </div>
                {cached && (
                    <div className="flex items-center gap-1 text-emerald-500">
                        <Clock className="w-3 h-3" />
                        <span>Cached</span>
                    </div>
                )}
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                {results.map((result) => (
                    <ImageResultCard 
                        key={result.id} 
                        result={result}
                        onClick={setSelectedImage}
                    />
                ))}
            </div>

            {/* Lightbox */}
            {selectedImage && (
                <div 
                    className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
                    onClick={() => setSelectedImage(null)}
                >
                    <div className="relative max-w-4xl max-h-[90vh] rounded-2xl overflow-hidden bg-white shadow-soft-lg">
                        <img
                            src={selectedImage.imageUrl}
                            alt={selectedImage.metadata?.file_name || 'Selected image'}
                            className="max-w-full max-h-[80vh] object-contain"
                        />
                        <div className="p-4 bg-white border-t border-gray-100">
                            <div className="flex items-center justify-between">
                                <div>
                                    <div className="font-semibold text-gray-800 text-[14px]">
                                        {selectedImage.metadata?.file_name}
                                    </div>
                                    <div className="text-[12px] text-gray-400 mt-0.5">
                                        {Math.round(selectedImage.score * 100)}% match
                                    </div>
                                </div>
                                {selectedImage.metadata?.width && (
                                    <div className="text-[12px] text-gray-400">
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
            <div className="rounded-2xl overflow-hidden border border-gray-100 shadow-soft bg-white hover:shadow-soft-md transition-all duration-300">
                <img
                    src={previewUrl}
                    alt={file.name}
                    className="w-full h-auto max-h-[400px] object-contain block"
                />
                <div className="px-3 py-2 bg-surface-50 border-t border-gray-100 flex items-center gap-2">
                    <ImageIcon className="w-3.5 h-3.5 text-blue-400" />
                    <span className="text-[11px] text-gray-500 font-medium truncate">{file.name}</span>
                </div>
            </div>
        );
    }

    return (
        <div className="flex items-center gap-3 bg-white px-4 py-3 rounded-2xl border border-gray-100 shadow-soft hover:shadow-soft-md transition-all duration-300">
            <div className="w-9 h-9 rounded-xl bg-red-50 flex items-center justify-center flex-shrink-0">
                <FileText className="w-4 h-4 text-red-400" />
            </div>
            <div className="flex flex-col overflow-hidden">
                <span className="text-[12px] text-gray-700 font-semibold truncate">{file.name}</span>
                <span className="text-[10px] text-gray-400 font-medium">Document</span>
            </div>
        </div>
    );
};

/**
 * Text Results Section — retrieved PDF chunks
 */
const TextResultsSection = ({ results, latencyMs }) => {
    if (!results || results.length === 0) return null;

    return (
        <div className="space-y-2">
            <div className="flex items-center gap-2 text-[11px] text-gray-400">
                <BookOpen className="w-3 h-3" />
                <span>{results.length} text chunks retrieved</span>
                {latencyMs && (
                    <>
                        <span className="text-gray-300">·</span>
                        <Zap className="w-3 h-3" />
                        <span>{latencyMs}ms</span>
                    </>
                )}
            </div>
            <div className="space-y-1.5 max-h-60 overflow-y-auto custom-scrollbar">
                {results.slice(0, 3).map((result, idx) => (
                    <div key={idx} className="text-[12px] text-gray-500 bg-white border border-gray-100 rounded-xl p-3 shadow-soft">
                        <div className="flex items-center gap-2 mb-1.5">
                            <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
                                Page {result.metadata?.page_num || '?'} · {result.metadata?.source_pdf || 'PDF'}
                            </span>
                            <span className="text-[10px] text-gray-300">
                                {Math.round(result.score * 100)}%
                            </span>
                        </div>
                        <p className="line-clamp-3 leading-relaxed">{result.metadata?.text || ''}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

/**
 * Streaming cursor
 */
const StreamingCursor = () => (
    <span className="inline-block w-[2px] h-4 bg-gray-600 ml-0.5 animate-pulse rounded-full" />
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

    const assistantLabel = isPdfIndexed ? 'PDF Processor' 
        : (message.type === 'rag_response' || message.type === 'streaming') ? 'RAG Assistant'
        : 'Image Search';

    return (
        <div className={clsx(
            "flex w-full px-4 md:px-6 py-5 transition-colors duration-200",
            isUser ? "bg-transparent" : "bg-white/40",
            isError && "bg-red-50/50"
        )}>
            <div className="max-w-3xl mx-auto flex gap-3 w-full text-left">
                {/* Avatar */}
                <div className={clsx(
                    "w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 shadow-soft transition-all",
                    isUser
                        ? "bg-white border border-gray-100"
                        : isError
                            ? "bg-red-400"
                            : isPdfIndexed
                                ? "bg-emerald-500"
                                : "bg-gray-800"
                )}>
                    {isUser ? (
                        <User className="w-3.5 h-3.5 text-gray-500" />
                    ) : isError ? (
                        <AlertCircle className="w-3.5 h-3.5 text-white" />
                    ) : isPdfIndexed ? (
                        <FileText className="w-3.5 h-3.5 text-white" />
                    ) : (
                        <div className="w-2 h-2 bg-white rounded-sm" />
                    )}
                </div>

                {/* Content */}
                <div className="flex-1 space-y-2.5 overflow-hidden min-w-0 pt-0.5">
                    <div className="font-semibold text-[13px] text-gray-700 flex items-center gap-2">
                        {isUser ? "You" : assistantLabel}
                        {isStreaming && (
                            <Loader2 className="w-3 h-3 text-gray-400 animate-spin" />
                        )}
                    </div>

                    {/* Attachments */}
                    {message.attachments && message.attachments.length > 0 && (
                        <div className="flex flex-wrap gap-2.5 mb-2">
                            {message.attachments.map((file, idx) => (
                                <div key={idx} className="flex flex-col gap-2 max-w-full sm:max-w-sm">
                                    <AttachmentItem file={file} />
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Text chunks */}
                    {hasTextResults && (
                        <TextResultsSection
                            results={message.textResults}
                            latencyMs={message.latencyMs}
                        />
                    )}

                    {/* Text content */}
                    {message.content && (
                        <div className={clsx(
                            "leading-[1.7] whitespace-pre-wrap text-[14px] max-w-none",
                            isError ? "text-red-600 font-medium" : "text-gray-600"
                        )}>
                            {message.content}
                            {isStreaming && <StreamingCursor />}
                        </div>
                    )}

                    {/* Streaming placeholder */}
                    {isStreaming && !message.content && !hasTextResults && (
                        <div className="flex items-center gap-2 text-gray-400 text-[13px]">
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            <span>Thinking...</span>
                        </div>
                    )}

                    {/* Image results */}
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
