import React, { useRef, useEffect, useState } from 'react';
import Sidebar from '../components/Sidebar';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput';
import useChat from '../hooks/useChat';
import { Menu } from 'lucide-react';

const Dashboard = () => {
    const userEmail = localStorage.getItem('userEmail');
    const {
        chats,
        currentChatId,
        messages,
        isTyping,
        createNewChat,
        sendMessage,
        selectChat,
        deleteChat
    } = useChat(userEmail);

    const messagesEndRef = useRef(null);
    const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isTyping]);

    // Only auto-select on mount if there's no active chat and history exists
    useEffect(() => {
        const storedAuth = localStorage.getItem('isAuthenticated');
        if (storedAuth && !currentChatId && chats.length > 0) {
            // We check local storage to only run this on fresh session entry or similar
            // But for now, let's keep it simple: if you have history and land here, 
            // you probably want to see the latest. 
            // However, the user wants "New Chat" behavior like ChatGPT.
            // ChatGPT usually lands on a new chat.
            // So let's actually leave it as NULL (New Chat) by default.
        }
    }, []);

    return (
        <div className="flex h-screen bg-white overflow-hidden text-gray-900 font-sans antialiased">
            {/* Sidebar - Desktop */}
            <div className="hidden md:flex">
                <Sidebar
                    chats={chats}
                    currentChatId={currentChatId}
                    onSelectChat={selectChat}
                    onNewChat={createNewChat}
                    onDeleteChat={deleteChat}
                />
            </div>

            {/* Sidebar - Mobile Overlay */}
            {isMobileSidebarOpen && (
                <div className="fixed inset-0 z-50 flex md:hidden">
                    <div className="relative w-64 h-full">
                        <Sidebar
                            chats={chats}
                            currentChatId={currentChatId}
                            onSelectChat={(id) => { selectChat(id); setIsMobileSidebarOpen(false); }}
                            onNewChat={() => { createNewChat(); setIsMobileSidebarOpen(false); }}
                            onDeleteChat={deleteChat}
                        />
                    </div>
                    <div className="flex-1 bg-black/50" onClick={() => setIsMobileSidebarOpen(false)}></div>
                </div>
            )}

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col h-full bg-white relative w-full">
                {/* Simple Header */}
                <div className="h-14 border-b border-gray-100 flex items-center px-4 justify-between md:hidden bg-white z-10">
                    <button onClick={() => setIsMobileSidebarOpen(true)} className="p-2 -ml-2 text-gray-600">
                        <Menu className="w-6 h-6" />
                    </button>
                    <span className="font-semibold text-gray-800">Chatbot</span>
                    <div className="w-8" /> {/* Spacer */}
                </div>

                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {messages.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center p-8 text-center text-gray-500 pb-20">
                            <div className="w-16 h-16 bg-gray-50 rounded-2xl flex items-center justify-center mb-6 border border-gray-100 shadow-sm">
                                <span className="text-3xl">✨</span>
                            </div>
                            <h3 className="text-[22px] font-bold text-gray-900 mb-3 tracking-tight">How can I help you today?</h3>
                            <p className="max-w-[400px] text-[15px] text-gray-500 leading-relaxed font-medium">Start a conversation or upload a document to get started.</p>
                        </div>
                    ) : (
                        <div className="flex flex-col py-4">
                            {messages.map((msg) => (
                                <ChatMessage key={msg.id} message={msg} />
                            ))}
                            {isTyping && (
                                <div className="px-4 md:px-8 py-6 bg-gray-50/50">
                                    <div className="max-w-4xl mx-auto flex gap-4 w-full">
                                        <div className="w-8 h-8 rounded flex items-center justify-center bg-white border border-gray-200 flex-shrink-0">
                                            <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-pulse"></div>
                                        </div>
                                        <div className="flex items-center">
                                            <span className="text-sm text-gray-500 font-medium">Thinking...</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* Input Area */}
                <ChatInput onSend={sendMessage} disabled={isTyping} />
            </div>
        </div>
    );
};

export default Dashboard;
