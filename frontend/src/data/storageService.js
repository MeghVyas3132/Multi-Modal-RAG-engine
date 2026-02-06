import { v4 as uuidv4 } from 'uuid';
import { MOCK_USERS, seedMockDataForUser, createNewUser, generateSessionToken } from './mockData';

/**
 * Multi-Tenant Storage Service
 * 
 * Provides complete isolation between users with:
 * - User-level data segregation (by userId)
 * - Session tokens for parallel chat execution
 * - Optimistic updates with localStorage persistence
 */

const STORAGE_PREFIX = 'chatbot_tenant_';
const USER_INDEX_KEY = 'chatbot_user_index';

/**
 * Get or create user by email
 * Returns user object with userId for tenant isolation
 */
export const getOrCreateUser = (email) => {
    if (!email) return null;
    
    const normalizedEmail = email.toLowerCase();
    
    // Check if user exists in index
    let userIndex = {};
    try {
        const indexData = localStorage.getItem(USER_INDEX_KEY);
        userIndex = indexData ? JSON.parse(indexData) : {};
    } catch (e) {
        console.error('Failed to load user index:', e);
    }
    
    // Return existing user
    if (userIndex[normalizedEmail]) {
        const tenantData = loadTenantData(userIndex[normalizedEmail]);
        if (tenantData?.user) {
            return tenantData.user;
        }
    }
    
    // Check for mock user
    if (MOCK_USERS[normalizedEmail]) {
        const mockUser = MOCK_USERS[normalizedEmail];
        userIndex[normalizedEmail] = mockUser.userId;
        localStorage.setItem(USER_INDEX_KEY, JSON.stringify(userIndex));
        seedMockDataForUser(normalizedEmail);
        return mockUser;
    }
    
    // Create new user
    const newUser = createNewUser(normalizedEmail);
    userIndex[normalizedEmail] = newUser.userId;
    localStorage.setItem(USER_INDEX_KEY, JSON.stringify(userIndex));
    
    // Initialize tenant data
    const tenantData = {
        user: newUser,
        chats: []
    };
    saveTenantData(newUser.userId, tenantData);
    
    return newUser;
};

/**
 * Load tenant data for a specific user
 */
export const loadTenantData = (userId) => {
    if (!userId) return null;
    
    try {
        const data = localStorage.getItem(`${STORAGE_PREFIX}${userId}`);
        if (data) {
            return JSON.parse(data);
        }
    } catch (e) {
        console.error('Failed to load tenant data:', e);
    }
    return null;
};

/**
 * Save tenant data for a specific user
 */
export const saveTenantData = (userId, data) => {
    if (!userId) return false;
    
    try {
        localStorage.setItem(`${STORAGE_PREFIX}${userId}`, JSON.stringify(data));
        return true;
    } catch (e) {
        console.error('Failed to save tenant data:', e);
        return false;
    }
};

/**
 * Load all chats for a user
 */
export const loadUserChats = (userId) => {
    const tenantData = loadTenantData(userId);
    return tenantData?.chats || [];
};

/**
 * Save chats for a user
 */
export const saveUserChats = (userId, chats) => {
    const tenantData = loadTenantData(userId) || { user: null, chats: [] };
    tenantData.chats = chats;
    return saveTenantData(userId, tenantData);
};

/**
 * Create a new chat session with isolated session token
 */
export const createChatSession = (userId, title, initialMessage = null) => {
    const chatId = uuidv4();
    const sessionToken = generateSessionToken();
    const now = Date.now();
    
    const newChat = {
        id: chatId,
        sessionToken,
        title: title || 'New Chat',
        messages: initialMessage ? [initialMessage] : [],
        createdAt: now,
        updatedAt: now,
        status: 'active'
    };
    
    const chats = loadUserChats(userId);
    chats.unshift(newChat);
    saveUserChats(userId, chats);
    
    return newChat;
};

/**
 * Update a specific chat session
 */
export const updateChatSession = (userId, chatId, updates) => {
    const chats = loadUserChats(userId);
    const chatIndex = chats.findIndex(c => c.id === chatId);
    
    if (chatIndex === -1) return null;
    
    chats[chatIndex] = {
        ...chats[chatIndex],
        ...updates,
        updatedAt: Date.now()
    };
    
    saveUserChats(userId, chats);
    return chats[chatIndex];
};

/**
 * Delete a chat session
 */
export const deleteChatSession = (userId, chatId) => {
    const chats = loadUserChats(userId);
    const filtered = chats.filter(c => c.id !== chatId);
    saveUserChats(userId, filtered);
    return filtered;
};

/**
 * Add message to a chat session
 */
export const addMessageToChat = (userId, chatId, message) => {
    const chats = loadUserChats(userId);
    const chatIndex = chats.findIndex(c => c.id === chatId);
    
    if (chatIndex === -1) return null;
    
    const messageWithId = {
        id: message.id || uuidv4(),
        ...message,
        timestamp: message.timestamp || Date.now()
    };
    
    chats[chatIndex].messages.push(messageWithId);
    chats[chatIndex].updatedAt = Date.now();
    
    // Update title from first user message if still default
    if (chats[chatIndex].title === 'New Chat' && message.role === 'user') {
        if (message.content && message.content.trim()) {
            chats[chatIndex].title = message.content.slice(0, 30) + (message.content.length > 30 ? '...' : '');
        } else if (message.attachments && message.attachments.length > 0) {
            const fileName = message.attachments[0].name || message.attachments[0].previewUrl || 'Attachment';
            chats[chatIndex].title = fileName.slice(0, 40) + (fileName.length > 40 ? '...' : '');
        }
    }
    
    saveUserChats(userId, chats);
    return messageWithId;
};

/**
 * Get active sessions (for parallel execution tracking)
 */
export const getActiveSessions = (userId) => {
    const chats = loadUserChats(userId);
    return chats
        .filter(c => c.status === 'active')
        .map(c => ({
            chatId: c.id,
            sessionToken: c.sessionToken,
            title: c.title,
            messageCount: c.messages.length
        }));
};

/**
 * Clear all data for a user (for testing/logout)
 */
export const clearUserData = (userId) => {
    if (!userId) return;
    localStorage.removeItem(`${STORAGE_PREFIX}${userId}`);
};

/**
 * Export all user data (for backup/migration)
 */
export const exportUserData = (userId) => {
    return loadTenantData(userId);
};

/**
 * Import user data (for restore/migration)
 */
export const importUserData = (userId, data) => {
    return saveTenantData(userId, data);
};

// Re-export generateSessionToken for external use
export { generateSessionToken } from './mockData';
