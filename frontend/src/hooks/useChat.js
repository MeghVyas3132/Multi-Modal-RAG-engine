import { useState, useEffect, useRef, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { 
    getOrCreateUser, 
    loadUserChats, 
    saveUserChats,
    generateSessionToken 
} from '../data/storageService';
import { searchByText, searchByImage, uploadImage, uploadPdf, streamChat, getImageUrl } from '../services/api';

/**
 * Multi-Tenant Chat Hook with RAG Backend Integration
 * 
 * Features:
 * - UUID-based user identification
 * - Session tokens for parallel chat execution
 * - Complete tenant isolation
 * - Text-to-image search via CLIP
 * - Image-to-image similarity search
 * - Real-time image upload and indexing
 */

const useChat = (userEmail) => {
    // User state with UUID
    const [currentUser, setCurrentUser] = useState(null);
    
    // Chat state
    const [chats, setChats] = useState([]);
    const [currentChatId, setCurrentChatId] = useState(null);
    const [currentSessionToken, setCurrentSessionToken] = useState(null);
    const [messages, setMessages] = useState([]);
    const [isTyping, setIsTyping] = useState(false);
    
    // Track loaded user for change detection
    const loadedEmailRef = useRef(null);
    const isInitialMount = useRef(true);

    // Initialize user and load their chats
    useEffect(() => {
        if (!userEmail) {
            setCurrentUser(null);
            setChats([]);
            setCurrentChatId(null);
            setCurrentSessionToken(null);
            setMessages([]);
            loadedEmailRef.current = null;
            return;
        }

        // If user email changed, reload everything
        if (loadedEmailRef.current !== userEmail) {
            // Get or create user (will seed mock data for demo accounts)
            const user = getOrCreateUser(userEmail);
            setCurrentUser(user);
            
            if (user) {
                // Load user's chats with tenant isolation
                const userChats = loadUserChats(user.userId);
                setChats(userChats);
            } else {
                setChats([]);
            }
            
            setCurrentChatId(null);
            setCurrentSessionToken(null);
            setMessages([]);
            loadedEmailRef.current = userEmail;
        }
    }, [userEmail]);

    // Save chats whenever they change (skip initial mount)
    useEffect(() => {
        if (isInitialMount.current) {
            isInitialMount.current = false;
            return;
        }

        if (currentUser?.userId && loadedEmailRef.current === userEmail) {
            saveUserChats(currentUser.userId, chats);
        }
    }, [chats, currentUser, userEmail]);

    // Update messages when current chat changes
    useEffect(() => {
        if (currentChatId) {
            const currentChat = chats.find(c => c.id === currentChatId);
            if (currentChat) {
                setMessages(currentChat.messages || []);
                setCurrentSessionToken(currentChat.sessionToken);
            }
        } else {
            setMessages([]);
            setCurrentSessionToken(null);
        }
    }, [currentChatId, chats]);

    // Create new chat session
    const createNewChat = useCallback(() => {
        setCurrentChatId(null);
        setCurrentSessionToken(null);
        setMessages([]);
    }, []);

    // Send message with session token tracking
    const sendMessage = useCallback(async (text, attachments = []) => {
        const hasImageAttachment = attachments.some(file => 
            file.type?.startsWith('image/') || /\.(jpg|jpeg|png|gif|webp)$/i.test(file.name)
        );

        const newUserMessage = {
            id: uuidv4(),
            role: 'user',
            content: text,
            attachments: attachments.map(f => ({
                name: f.name,
                type: f.type,
                size: f.size,
                // Create preview URL for images
                previewUrl: f.type?.startsWith('image/') ? URL.createObjectURL(f) : null
            })),
            timestamp: Date.now(),
        };

        let activeId = currentChatId;
        let activeSessionToken = currentSessionToken;

        // Create new chat if needed
        if (!activeId) {
            activeId = uuidv4();
            activeSessionToken = `sess_${uuidv4()}`;
            
            // Determine chat title from text or attachment name
            let chatTitle;
            if (text.trim()) {
                chatTitle = text.slice(0, 30) + (text.length > 30 ? '...' : '');
            } else if (attachments.length > 0) {
                chatTitle = attachments[0].name || attachments[0].previewUrl || 'Attachment';
                chatTitle = chatTitle.slice(0, 40) + (chatTitle.length > 40 ? '...' : '');
            } else {
                chatTitle = 'New Chat';
            }

            const newChat = {
                id: activeId,
                sessionToken: activeSessionToken,
                title: chatTitle,
                messages: [newUserMessage],
                createdAt: Date.now(),
                updatedAt: Date.now(),
                status: 'active'
            };
            
            setChats(prev => [newChat, ...prev]);
            setCurrentChatId(activeId);
            setCurrentSessionToken(activeSessionToken);
            setMessages([newUserMessage]);
        } else {
            // Update existing chat
            setMessages(prev => [...prev, newUserMessage]);
            setChats(prev => prev.map(chat => {
                if (chat.id === activeId) {
                    return { 
                        ...chat, 
                        messages: [...chat.messages, newUserMessage],
                        updatedAt: Date.now()
                    };
                }
                return chat;
            }));
        }

        setIsTyping(true);

        try {
            let response;
            let botMessage;

            const hasPdfAttachment = attachments.some(file =>
                file.name?.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf'
            );

            // If there's a PDF attachment, upload and index it
            if (hasPdfAttachment) {
                const pdfFile = attachments.find(f =>
                    f.name?.toLowerCase().endsWith('.pdf') || f.type === 'application/pdf'
                );

                if (pdfFile) {
                    try {
                        response = await uploadPdf(pdfFile);

                        botMessage = {
                            id: uuidv4(),
                            role: 'assistant',
                            content: `✅ **${response.filename}** indexed successfully!\n\n` +
                                `📄 ${response.total_pages} pages · ${response.chunks_indexed} text chunks · ${response.images_indexed} images\n` +
                                `⚡ Processed in ${response.latency_ms}ms\n\n` +
                                `You can now ask questions about this document.`,
                            type: 'pdf_indexed',
                            pdfResult: response,
                            timestamp: Date.now(),
                        };
                    } catch (error) {
                        botMessage = {
                            id: uuidv4(),
                            role: 'assistant',
                            content: `PDF upload failed: ${error.message}`,
                            type: 'error',
                            timestamp: Date.now(),
                        };
                    }
                }
            }
            // If there's an image attachment, do image-to-image search
            else if (hasImageAttachment && attachments.length > 0) {
                const imageFile = attachments.find(f => 
                    f.type?.startsWith('image/') || /\.(jpg|jpeg|png|gif|webp)$/i.test(f.name)
                );
                
                if (imageFile) {
                    response = await searchByImage(imageFile, { topK: 6 });
                    
                    botMessage = {
                        id: uuidv4(),
                        role: 'assistant',
                        content: `Found ${response.total} similar images in ${response.latency_ms}ms`,
                        type: 'image_results',
                        searchResults: response.results.map(r => ({
                            id: r.id,
                            score: r.score,
                            imageUrl: getImageUrl(r.metadata),
                            metadata: r.metadata
                        })),
                        latencyMs: response.latency_ms,
                        timestamp: Date.now(),
                    };
                }
            } else if (text.trim()) {
                // Text query → RAG chat with streaming
                // Create a placeholder bot message for streaming
                const botId = uuidv4();
                botMessage = {
                    id: botId,
                    role: 'assistant',
                    content: '',
                    type: 'streaming',
                    textResults: [],
                    searchResults: [],
                    latencyMs: null,
                    isStreaming: true,
                    timestamp: Date.now(),
                };

                // Add the placeholder immediately
                setMessages(prev => [...prev, botMessage]);
                setChats(prev => prev.map(chat => {
                    if (chat.id === activeId) {
                        return {
                            ...chat,
                            messages: [...chat.messages, botMessage],
                            updatedAt: Date.now()
                        };
                    }
                    return chat;
                }));

                // Stream the response
                let streamedContent = '';

                await streamChat(text, { topK: 5, includeImages: true }, {
                    onRetrieval: (data) => {
                        // Phase 1: update with retrieval results
                        const imageResults = (data.image_results || []).map(r => ({
                            id: r.id,
                            score: r.score,
                            imageUrl: getImageUrl(r.metadata),
                            metadata: r.metadata
                        }));

                        setMessages(prev => prev.map(m => {
                            if (m.id === botId) {
                                return {
                                    ...m,
                                    textResults: data.text_results || [],
                                    searchResults: imageResults,
                                    latencyMs: data.latency_ms,
                                };
                            }
                            return m;
                        }));
                        setChats(prev => prev.map(chat => {
                            if (chat.id === activeId) {
                                return {
                                    ...chat,
                                    messages: chat.messages.map(m => {
                                        if (m.id === botId) {
                                            return {
                                                ...m,
                                                textResults: data.text_results || [],
                                                searchResults: imageResults,
                                                latencyMs: data.latency_ms,
                                            };
                                        }
                                        return m;
                                    }),
                                    updatedAt: Date.now()
                                };
                            }
                            return chat;
                        }));
                    },
                    onToken: (token) => {
                        // Phase 2: append token to content
                        streamedContent += token;
                        setMessages(prev => prev.map(m => {
                            if (m.id === botId) {
                                return { ...m, content: streamedContent };
                            }
                            return m;
                        }));
                    },
                    onDone: () => {
                        // Mark streaming complete
                        setMessages(prev => prev.map(m => {
                            if (m.id === botId) {
                                return {
                                    ...m,
                                    content: streamedContent,
                                    isStreaming: false,
                                    type: streamedContent ? 'rag_response' : 'image_results',
                                };
                            }
                            return m;
                        }));
                        setChats(prev => prev.map(chat => {
                            if (chat.id === activeId) {
                                return {
                                    ...chat,
                                    messages: chat.messages.map(m => {
                                        if (m.id === botId) {
                                            return {
                                                ...m,
                                                content: streamedContent,
                                                isStreaming: false,
                                                type: streamedContent ? 'rag_response' : 'image_results',
                                            };
                                        }
                                        return m;
                                    }),
                                    updatedAt: Date.now()
                                };
                            }
                            return chat;
                        }));
                    },
                    onError: (error) => {
                        setMessages(prev => prev.map(m => {
                            if (m.id === botId) {
                                return {
                                    ...m,
                                    content: streamedContent || `Error: ${error}`,
                                    isStreaming: false,
                                    type: streamedContent ? 'rag_response' : 'error',
                                };
                            }
                            return m;
                        }));
                    },
                });

                // Already handled — skip the generic add below
                botMessage = null;
            }

            if (botMessage) {
                setMessages(prev => [...prev, botMessage]);
                setChats(prev => prev.map(chat => {
                    if (chat.id === activeId) {
                        return { 
                            ...chat, 
                            messages: [...chat.messages, botMessage],
                            updatedAt: Date.now()
                        };
                    }
                    return chat;
                }));
            }

        } catch (error) {
            console.error('Search failed:', error);
            
            // Error message
            const errorMessage = {
                id: uuidv4(),
                role: 'assistant',
                content: `Search failed: ${error.message}. Make sure the backend is running at http://localhost:8000`,
                type: 'error',
                timestamp: Date.now(),
            };

            setMessages(prev => [...prev, errorMessage]);
            setChats(prev => prev.map(chat => {
                if (chat.id === activeId) {
                    return { 
                        ...chat, 
                        messages: [...chat.messages, errorMessage],
                        updatedAt: Date.now()
                    };
                }
                return chat;
            }));
        }

        setIsTyping(false);
    }, [currentChatId, currentSessionToken]);

    // Select existing chat
    const selectChat = useCallback((id) => {
        setCurrentChatId(id);
    }, []);

    // Delete chat
    const deleteChat = useCallback((e, id) => {
        e.stopPropagation();
        setChats(prev => prev.filter(c => c.id !== id));
        if (currentChatId === id) {
            setCurrentChatId(null);
            setCurrentSessionToken(null);
        }
    }, [currentChatId]);

    // Get current session info (useful for parallel execution tracking)
    const getSessionInfo = useCallback(() => {
        if (!currentChatId) return null;
        const chat = chats.find(c => c.id === currentChatId);
        return chat ? {
            chatId: chat.id,
            sessionToken: chat.sessionToken,
            userId: currentUser?.userId,
            userEmail: currentUser?.email
        } : null;
    }, [currentChatId, chats, currentUser]);

    return {
        // User info
        currentUser,
        
        // Chat state
        chats,
        currentChatId,
        currentSessionToken,
        messages,
        isTyping,
        
        // Actions
        createNewChat,
        sendMessage,
        selectChat,
        deleteChat,
        
        // Utilities
        getSessionInfo
    };
};

export default useChat;
