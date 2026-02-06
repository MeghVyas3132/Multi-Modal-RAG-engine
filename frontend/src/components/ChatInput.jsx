import React, { useState, useRef, useEffect } from 'react';
import { Send, Plus, X, FileText, Image as ImageIcon, File as FileIcon } from 'lucide-react';
import clsx from 'clsx';

const FilePreview = ({ file, onRemove }) => {
    const [previewUrl, setPreviewUrl] = useState(null);
    const isImage = file.type?.startsWith('image/') || (file.name && /\.(jpg|jpeg|png|gif|webp)$/i.test(file.name));

    useEffect(() => {
        if (isImage && file instanceof File) {
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
            return () => URL.revokeObjectURL(url);
        }
    }, [file, isImage]);

    return (
        <div className="relative flex items-center gap-2 bg-white pl-2 pr-1.5 py-1.5 rounded-xl border border-gray-200 shadow-sm flex-shrink-0 group">
            {previewUrl ? (
                <div className="w-8 h-8 rounded-lg overflow-hidden border border-gray-100 flex-shrink-0">
                    <img src={previewUrl} alt="preview" className="w-full h-full object-cover" />
                </div>
            ) : (
                <div className="w-8 h-8 rounded-lg bg-gray-50 flex items-center justify-center flex-shrink-0 border border-gray-50">
                    <FileText className="w-4 h-4 text-gray-400" />
                </div>
            )}
            <div className="flex flex-col overflow-hidden max-w-[120px]">
                <span className="text-[11px] text-gray-700 font-bold truncate leading-tight">{file.name}</span>
                <span className="text-[9px] text-gray-400 uppercase font-black tracking-tighter leading-none">
                    {isImage ? 'Image' : 'Document'}
                </span>
            </div>
            <button
                onClick={onRemove}
                className="ml-1 p-1 text-gray-300 hover:text-red-500 transition-colors"
                title="Remove file"
            >
                <X className="w-3.5 h-3.5" />
            </button>
        </div>
    );
};

const ChatInput = ({ onSend, disabled }) => {
    const [input, setInput] = useState('');
    const [files, setFiles] = useState([]);
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const menuRef = useRef(null);
    const textareaRef = useRef(null);

    // Hidden inputs for each type
    const imageInputRef = useRef(null);
    const pdfInputRef = useRef(null);
    const docInputRef = useRef(null);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleSend = () => {
        if ((!input.trim() && files.length === 0) || disabled) return;
        onSend(input, files);
        setInput('');
        setFiles([]);
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            const newFiles = Array.from(e.target.files);
            setFiles(prev => [...prev, ...newFiles]);
        }
        setIsMenuOpen(false);
        // Clear value to allow re-selection of the same file
        e.target.value = '';
    };

    const removeFile = (index) => {
        setFiles(prev => prev.filter((_, i) => i !== index));
    };

    // Close menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setIsMenuOpen(false);
            }
        };
        if (isMenuOpen) {
            document.addEventListener('mousedown', handleClickOutside);
        }
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [isMenuOpen]);

    return (
        <div className="w-full max-w-4xl mx-auto px-4 pb-6 pt-2 relative">

            {/* Hidden Inputs */}
            <input type="file" ref={imageInputRef} className="hidden" accept="image/*" onChange={handleFileChange} multiple />
            <input type="file" ref={pdfInputRef} className="hidden" accept=".pdf" onChange={handleFileChange} multiple />
            <input type="file" ref={docInputRef} className="hidden" accept=".doc,.docx,.txt" onChange={handleFileChange} multiple />

            {/* Attachment Menu Popover */}
            {isMenuOpen && (
                <div
                    ref={menuRef}
                    className="absolute left-6 bottom-[72px] w-48 bg-white border border-gray-100 rounded-xl shadow-2xl z-50 py-1 overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-200"
                >
                    <button
                        onClick={() => pdfInputRef.current?.click()}
                        className="flex items-center gap-3 w-full px-4 py-2.5 hover:bg-gray-50 text-gray-700 text-sm transition-colors border-b border-gray-50 text-left"
                    >
                        <FileIcon className="w-4 h-4 text-red-500" />
                        <span className="font-bold">Upload PDF</span>
                    </button>
                    <button
                        onClick={() => imageInputRef.current?.click()}
                        className="flex items-center gap-3 w-full px-4 py-2.5 hover:bg-gray-50 text-gray-700 text-sm transition-colors border-b border-gray-50 text-left"
                    >
                        <ImageIcon className="w-4 h-4 text-blue-500" />
                        <span className="font-bold">Upload Image</span>
                    </button>
                    <button
                        onClick={() => docInputRef.current?.click()}
                        className="flex items-center gap-3 w-full px-4 py-2.5 hover:bg-gray-50 text-gray-700 text-sm transition-colors text-left"
                    >
                        <FileText className="w-4 h-4 text-orange-500" />
                        <span className="font-bold">Upload Document</span>
                    </button>
                </div>
            )}

            <div className="relative flex flex-col border border-gray-200 rounded-2xl shadow-sm bg-white focus-within:ring-1 focus-within:ring-black/[0.02] focus-within:border-gray-300 transition-all overflow-hidden lg:max-w-3xl lg:mx-auto">

                {/* File Previews inside the input box */}
                {files.length > 0 && (
                    <div className="flex gap-2 p-3 bg-gray-50/30 border-b border-gray-100 overflow-x-auto scrollbar-hide">
                        {files.map((file, idx) => (
                            <FilePreview key={idx} file={file} onRemove={() => removeFile(idx)} />
                        ))}
                    </div>
                )}

                {/* Input Area */}
                <div className="flex items-end p-2 px-3 gap-2 min-h-[52px]">
                    <button
                        onClick={(e) => {
                            e.preventDefault();
                            setIsMenuOpen(!isMenuOpen);
                        }}
                        className={clsx(
                            "p-2 rounded-lg transition-all flex-shrink-0 mb-0.5",
                            isMenuOpen ? "bg-gray-100 text-gray-900" : "text-gray-400 hover:text-gray-900 hover:bg-gray-50"
                        )}
                        title="Add attachment"
                    >
                        <Plus className={clsx("w-5 h-5 transition-transform duration-200", isMenuOpen && "rotate-45")} />
                    </button>

                    <textarea
                        ref={textareaRef}
                        value={input}
                        onChange={(e) => {
                            setInput(e.target.value);
                            e.target.style.height = 'auto';
                            e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
                        }}
                        onKeyDown={handleKeyDown}
                        placeholder="Message AI..."
                        className="flex-1 max-h-48 py-2 px-2 focus:outline-none resize-none bg-transparent text-[15px] text-gray-800 leading-relaxed placeholder-gray-400 border-none ring-0 focus:ring-0 overflow-hidden"
                        style={{ appearance: 'none', WebkitAppearance: 'none' }}
                        rows={1}
                        disabled={disabled}
                    />

                    <button
                        onClick={handleSend}
                        disabled={(!input.trim() && files.length === 0) || disabled}
                        className={clsx(
                            "p-2 rounded-xl transition-all duration-200 flex-shrink-0 mb-0.5",
                            (input.trim() || files.length > 0) && !disabled
                                ? "bg-black text-white hover:bg-gray-800 shadow-sm"
                                : "bg-gray-50 text-gray-200 cursor-not-allowed"
                        )}
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </div>
            </div>
            <div className="text-center mt-3">
                <p className="text-[11px] text-gray-400 font-medium">AI can make mistakes. Consider verifying important information.</p>
            </div>
        </div>
    );
};

export default ChatInput;
