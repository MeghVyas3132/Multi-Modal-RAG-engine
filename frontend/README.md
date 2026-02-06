# Chatbot Frontend

A modern, multi-tenant AI chatbot interface built with React and Vite. Features persistent chat history, user isolation, and a clean conversational UI inspired by ChatGPT.

---

## Overview

This project provides a production-ready frontend for AI-powered chat applications. It implements complete user isolation with localStorage-based persistence, allowing multiple users to maintain separate chat histories on the same device. The architecture is designed for easy backend integration when needed.

---

## Features

- Multi-tenant architecture with UUID-based user identification
- Persistent chat history across sessions using localStorage
- Session tokens for parallel chat execution tracking
- Responsive design with mobile sidebar overlay
- File attachment support (images, PDFs, documents)
- Real-time typing indicators
- Chat search and filtering
- Demo accounts with pre-seeded conversation data

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Framework | React 18 |
| Build Tool | Vite |
| Styling | Tailwind CSS |
| Routing | React Router v6 |
| Icons | Lucide React |
| State | React Hooks |

---

## Project Structure

```
chatbot/
├── src/
│   ├── components/
│   │   ├── ChatInput.jsx      # Message input with file attachments
│   │   ├── ChatMessage.jsx    # Message bubble rendering
│   │   └── Sidebar.jsx        # Chat history and navigation
│   ├── data/
│   │   ├── index.js           # Data layer exports
│   │   ├── mockData.js        # Demo users and seeded chats
│   │   └── storageService.js  # Multi-tenant localStorage CRUD
│   ├── hooks/
│   │   └── useChat.js         # Chat state management hook
│   ├── pages/
│   │   ├── Dashboard.jsx      # Main chat interface
│   │   └── Login.jsx          # Authentication screen
│   ├── App.jsx                # Router configuration
│   ├── index.css              # Global styles
│   └── main.jsx               # Application entry point
├── index.html
├── package.json
├── tailwind.config.js
├── postcss.config.js
└── vite.config.js
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mili609/chatbot.git
cd chatbot

# Install dependencies
npm install

# Start development server
npm run dev
```

---

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |

---

## Demo Accounts

The following accounts are pre-seeded with chat history for testing:

| Email | Chat Sessions |
|-------|---------------|
| demo@example.com | 3 conversations |
| alice@example.com | 2 conversations |
| bob@example.com | 1 conversation |

New email addresses create fresh accounts with empty history.

---

## Data Architecture

```
User (UUID)
└── Chat Sessions (isolated per user)
    └── Session Token (for parallel execution)
        └── Messages[]
```

Storage keys follow the pattern `chatbot_tenant_{userId}` for complete tenant isolation.

---

## License

See LICENSE file for details.
