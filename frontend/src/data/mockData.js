import { v4 as uuidv4 } from 'uuid';

/**
 * Multi-Tenant Chat Data Architecture
 * 
 * Structure:
 * - Users (identified by UUID, keyed by email for lookup)
 * - Each user has multiple chat sessions
 * - Each chat session has its own sessionToken for parallel execution
 * - Messages are nested within chat sessions
 * 
 * Storage Schema:
 * {
 *   users: {
 *     [email]: {
 *       userId: UUID,
 *       email: string,
 *       displayName: string,
 *       createdAt: timestamp,
 *       preferences: {...}
 *     }
 *   },
 *   chatSessions: {
 *     [userId]: {
 *       [chatId]: {
 *         chatId: UUID,
 *         sessionToken: UUID (for parallel execution tracking),
 *         title: string,
 *         createdAt: timestamp,
 *         updatedAt: timestamp,
 *         status: 'active' | 'archived',
 *         messages: [...]
 *       }
 *     }
 *   }
 * }
 */

// Demo user accounts with pre-seeded data
export const MOCK_USERS = {
    'demo@example.com': {
        userId: 'usr_a1b2c3d4-e5f6-7890-abcd-ef1234567890',
        email: 'demo@example.com',
        displayName: 'Demo User',
        createdAt: Date.now() - 86400000 * 30, // 30 days ago
        preferences: {
            theme: 'dark',
            language: 'en'
        }
    },
    'alice@example.com': {
        userId: 'usr_b2c3d4e5-f6a7-8901-bcde-f12345678901',
        email: 'alice@example.com',
        displayName: 'Alice Johnson',
        createdAt: Date.now() - 86400000 * 15,
        preferences: {
            theme: 'dark',
            language: 'en'
        }
    },
    'bob@example.com': {
        userId: 'usr_c3d4e5f6-a7b8-9012-cdef-123456789012',
        email: 'bob@example.com',
        displayName: 'Bob Smith',
        createdAt: Date.now() - 86400000 * 7,
        preferences: {
            theme: 'light',
            language: 'en'
        }
    }
};

// Generate mock chat sessions with messages
const generateMockMessage = (role, content, timestamp) => ({
    id: uuidv4(),
    role,
    content,
    timestamp,
    attachments: []
});

const createChatSession = (title, messages, createdAt) => {
    const chatId = uuidv4();
    return {
        chatId,
        sessionToken: `sess_${uuidv4()}`, // Unique token for parallel execution
        title,
        createdAt,
        updatedAt: messages.length > 0 ? messages[messages.length - 1].timestamp : createdAt,
        status: 'active',
        messages
    };
};

// Pre-seeded chat history for demo users
export const MOCK_CHAT_SESSIONS = {
    // Demo User's chats
    'usr_a1b2c3d4-e5f6-7890-abcd-ef1234567890': [
        createChatSession(
            'Help with React hooks',
            [
                generateMockMessage('user', 'Can you explain useEffect to me?', Date.now() - 86400000 * 2),
                generateMockMessage('assistant', 'useEffect is a React Hook that lets you synchronize a component with an external system. It runs after every render by default, but you can control when it runs using the dependency array.', Date.now() - 86400000 * 2 + 1000),
                generateMockMessage('user', 'What about cleanup functions?', Date.now() - 86400000 * 2 + 60000),
                generateMockMessage('assistant', 'Cleanup functions run before the effect runs again and when the component unmounts. You return a function from useEffect to specify cleanup logic. This is useful for subscriptions, timers, or event listeners.', Date.now() - 86400000 * 2 + 61000)
            ],
            Date.now() - 86400000 * 2
        ),
        createChatSession(
            'API integration patterns',
            [
                generateMockMessage('user', 'What\'s the best way to handle API calls in React?', Date.now() - 86400000),
                generateMockMessage('assistant', 'There are several patterns: 1) useEffect with fetch/axios for simple cases, 2) React Query or SWR for caching and automatic refetching, 3) Redux with middleware like thunk or saga for complex state. I recommend React Query for most use cases.', Date.now() - 86400000 + 1000)
            ],
            Date.now() - 86400000
        ),
        createChatSession(
            'Tailwind CSS tips',
            [
                generateMockMessage('user', 'How do I create responsive designs with Tailwind?', Date.now() - 3600000),
                generateMockMessage('assistant', 'Tailwind uses mobile-first breakpoints: sm (640px), md (768px), lg (1024px), xl (1280px), 2xl (1536px). Prefix any utility with these breakpoints. Example: "text-sm md:text-base lg:text-lg" for responsive text sizing.', Date.now() - 3600000 + 1000)
            ],
            Date.now() - 3600000
        )
    ],
    
    // Alice's chats
    'usr_b2c3d4e5-f6a7-8901-bcde-f12345678901': [
        createChatSession(
            'Python data analysis',
            [
                generateMockMessage('user', 'How do I read a CSV file in pandas?', Date.now() - 86400000 * 3),
                generateMockMessage('assistant', 'Use pd.read_csv("file.csv"). You can specify parameters like delimiter, header, encoding, and dtype for column types. Example: df = pd.read_csv("data.csv", delimiter=",", header=0)', Date.now() - 86400000 * 3 + 1000)
            ],
            Date.now() - 86400000 * 3
        ),
        createChatSession(
            'Machine learning basics',
            [
                generateMockMessage('user', 'What\'s the difference between supervised and unsupervised learning?', Date.now() - 86400000),
                generateMockMessage('assistant', 'Supervised learning uses labeled data to train models for prediction (classification, regression). Unsupervised learning finds patterns in unlabeled data (clustering, dimensionality reduction). Semi-supervised combines both approaches.', Date.now() - 86400000 + 1000)
            ],
            Date.now() - 86400000
        )
    ],
    
    // Bob's chats
    'usr_c3d4e5f6-a7b8-9012-cdef-123456789012': [
        createChatSession(
            'Docker containerization',
            [
                generateMockMessage('user', 'How do I create a Dockerfile for a Node.js app?', Date.now() - 86400000 * 2),
                generateMockMessage('assistant', 'Basic Dockerfile:\n\nFROM node:18-alpine\nWORKDIR /app\nCOPY package*.json ./\nRUN npm ci --only=production\nCOPY . .\nEXPOSE 3000\nCMD ["node", "index.js"]\n\nUse multi-stage builds for smaller images.', Date.now() - 86400000 * 2 + 1000)
            ],
            Date.now() - 86400000 * 2
        )
    ]
};

/**
 * Seed mock data into localStorage for a specific user
 * Only seeds if user has no existing data
 */
export const seedMockDataForUser = (email) => {
    const mockUser = MOCK_USERS[email.toLowerCase()];
    if (!mockUser) return false;
    
    const storageKey = `chatbot_tenant_${mockUser.userId}`;
    const existing = localStorage.getItem(storageKey);
    
    // Only seed if no existing data
    if (!existing) {
        const mockChats = MOCK_CHAT_SESSIONS[mockUser.userId] || [];
        const tenantData = {
            user: mockUser,
            chats: mockChats.map(chat => ({
                id: chat.chatId,
                sessionToken: chat.sessionToken,
                title: chat.title,
                messages: chat.messages,
                createdAt: chat.createdAt,
                updatedAt: chat.updatedAt,
                status: chat.status
            }))
        };
        localStorage.setItem(storageKey, JSON.stringify(tenantData));
        return true;
    }
    return false;
};

/**
 * Generate a new session token for parallel chat execution
 */
export const generateSessionToken = () => `sess_${uuidv4()}`;

/**
 * Create a new user entry (for new signups)
 */
export const createNewUser = (email, displayName) => ({
    userId: `usr_${uuidv4()}`,
    email: email.toLowerCase(),
    displayName: displayName || email.split('@')[0],
    createdAt: Date.now(),
    preferences: {
        theme: 'dark',
        language: 'en'
    }
});
