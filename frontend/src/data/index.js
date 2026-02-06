/**
 * Data Layer - Multi-Tenant Chat Storage
 * 
 * This module provides a complete data layer for multi-tenant chat storage
 * with user isolation, session tracking, and mock data for development.
 * 
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                        User Layer                               │
 * │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
 * │  │ User A (UUID)   │  │ User B (UUID)   │  │ User C (UUID)   │ │
 * │  │ demo@example    │  │ alice@example   │  │ bob@example     │ │
 * │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
 * └───────────┼────────────────────┼────────────────────┼──────────┘
 *             │                    │                    │
 * ┌───────────┼────────────────────┼────────────────────┼──────────┐
 * │           ▼                    ▼                    ▼          │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │               Chat Sessions (Isolated Per User)          │   │
 * │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
 * │  │  │ Chat 1      │  │ Chat 2      │  │ Chat 3      │      │   │
 * │  │  │ Session: X  │  │ Session: Y  │  │ Session: Z  │      │   │
 * │  │  │ Messages[]  │  │ Messages[]  │  │ Messages[]  │      │   │
 * │  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
 * │  └─────────────────────────────────────────────────────────┘   │
 * │                     localStorage (Tenant Isolated)              │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * Demo Accounts (auto-seeded with mock data):
 * - demo@example.com - 3 chat sessions
 * - alice@example.com - 2 chat sessions  
 * - bob@example.com - 1 chat session
 * 
 * Usage:
 * ```js
 * import { getOrCreateUser, loadUserChats, saveUserChats } from './data';
 * 
 * // Get user (creates if new, seeds mock data for demo accounts)
 * const user = getOrCreateUser('demo@example.com');
 * 
 * // Load user's chats (tenant-isolated)
 * const chats = loadUserChats(user.userId);
 * 
 * // Save chats
 * saveUserChats(user.userId, updatedChats);
 * ```
 */

// Storage Service - Main API
export {
    getOrCreateUser,
    loadTenantData,
    saveTenantData,
    loadUserChats,
    saveUserChats,
    createChatSession,
    updateChatSession,
    deleteChatSession,
    addMessageToChat,
    getActiveSessions,
    clearUserData,
    exportUserData,
    importUserData,
    generateSessionToken
} from './storageService';

// Mock Data - For development/testing
export {
    MOCK_USERS,
    MOCK_CHAT_SESSIONS,
    seedMockDataForUser,
    createNewUser
} from './mockData';
