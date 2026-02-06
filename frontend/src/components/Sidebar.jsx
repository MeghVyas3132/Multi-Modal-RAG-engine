import React, { useState } from 'react';
import { MessageSquare, Plus, Trash2, LogOut, Search } from 'lucide-react';
import clsx from 'clsx';
import { useNavigate } from 'react-router-dom';

const Sidebar = ({ chats, currentChatId, onSelectChat, onNewChat, onDeleteChat }) => {
    const navigate = useNavigate();
    const userEmail = localStorage.getItem('userEmail') || 'user@example.com';
    const [searchTerm, setSearchTerm] = useState('');

    const handleLogout = () => {
        localStorage.removeItem('isAuthenticated');
        localStorage.removeItem('userEmail');
        navigate('/login');
    };

    // Filter out empty chats (0 messages) and apply search term
    const filteredChats = chats
        .filter(chat => chat.messages && chat.messages.length > 0)
        .filter(chat => (chat.title || '').toLowerCase().includes(searchTerm.toLowerCase()));

    return (
        <div className="flex flex-col h-full w-64 bg-[#171717] text-gray-200 border-r border-[#262626] flex-shrink-0">
            {/* Sidebar Header */}
            <div className="flex items-center h-14 px-4">
                <div className="flex items-center gap-2 overflow-hidden">
                    <div className="w-6 h-6 bg-white rounded flex items-center justify-center flex-shrink-0">
                        <div className="w-2.5 h-2.5 bg-black rounded-sm" />
                    </div>
                    <span className="font-semibold text-white text-base tracking-tight truncate">Chatbot</span>
                </div>
            </div>

            <div className="px-3 pt-2 space-y-4">
                {/* New Chat Button */}
                <button
                    onClick={onNewChat}
                    className={clsx(
                        "flex items-center justify-start gap-3 w-full px-3 py-2 rounded-lg transition-all duration-200 border border-[#333]",
                        currentChatId === null ? "bg-[#2f2f2f] text-white border-white/10" : "bg-transparent hover:bg-[#2f2f2f] text-gray-200"
                    )}
                >
                    <Plus className="w-4 h-4" />
                    <span className="text-sm font-medium">New chat</span>
                </button>

                {/* Search Input */}
                <div className="relative">
                    <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-500" />
                    <input
                        type="text"
                        placeholder="Search chats..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-[#212121] text-sm text-gray-200 pl-9 pr-3 py-2 rounded-lg border border-[#333] focus:outline-none focus:border-[#444] placeholder-gray-550"
                    />
                </div>
            </div>

            <div className="flex-1 overflow-y-auto px-2 py-4 mt-2">
                <div className="text-[11px] font-bold text-gray-500 px-3 mb-2 uppercase tracking-widest">
                    Your Chats
                </div>
                <div className="space-y-0.5">
                    {filteredChats.map((chat) => (
                        <div
                            key={chat.id}
                            onClick={() => onSelectChat(chat.id)}
                            className={clsx(
                                "group flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-200 relative",
                                currentChatId === chat.id
                                    ? "bg-[#2f2f2f] text-white"
                                    : "text-gray-400 hover:bg-[#212121] hover:text-gray-200"
                            )}
                        >
                            <MessageSquare className="w-4 h-4 flex-shrink-0 opacity-70" />
                            <div className="flex-1 max-w-full overflow-hidden">
                                <div className="text-[14px] truncate select-none font-medium">
                                    {chat.title || "New Chat"}
                                </div>
                            </div>

                            <button
                                onClick={(e) => onDeleteChat(e, chat.id)}
                                className="absolute right-2 opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity"
                            >
                                <Trash2 className="w-3.5 h-3.5" />
                            </button>
                        </div>
                    ))}
                    {filteredChats.length === 0 && (
                        <div className="px-3 text-xs text-gray-600 italic mt-2">No chats found</div>
                    )}
                </div>
            </div>

            {/* User Profile Section */}
            <div className="p-3 border-t border-[#262626]">
                <div className="flex items-center gap-3 px-2 py-2 mb-1 rounded-lg hover:bg-[#212121] transition-colors cursor-default">
                    <div className="w-7 h-7 rounded bg-blue-600 flex items-center justify-center text-[10px] font-bold text-white uppercase flex-shrink-0">
                        {userEmail.substring(0, 2)}
                    </div>
                    <div className="flex-1 overflow-hidden min-w-0">
                        <div className="text-sm font-medium text-gray-200 truncate">{userEmail}</div>
                    </div>
                </div>
                <button
                    onClick={handleLogout}
                    className="flex items-center gap-3 w-full px-2 py-2 hover:bg-[#212121] rounded-lg transition-colors text-gray-400 hover:text-gray-200"
                >
                    <LogOut className="w-4 h-4" />
                    <span className="text-sm">Log out</span>
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
