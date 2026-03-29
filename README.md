# NexusCore — AI Gaming Support Platform

NexusCore is an AI-powered gaming support platform that answers player questions by combining real-time data from Steam, Reddit, and RAWG with GPT-4o. Instead of static FAQs, it pulls live patch notes, community discussions, and game metadata to give contextually accurate answers.

---

## Features

- **AI Chat (FAQ)** — Ask any gaming question in natural language; the AI extracts the game name automatically
- **Game Support Page** — Game-specific chat with live Reddit community posts surfaced alongside the answer
- **Category-aware responses** — Separate modes for Gameplay, Technical, Updates, Community, Account/Billing, General
- **Steam integration** — Pulls official patch notes and changelogs directly from the Steam News API
- **Reddit integration** — Searches the correct game subreddit with a 3-tier fallback (search → top → hot) using a browser User-Agent to bypass bot detection
- **Semantic memory** — Reddit posts are stored as vector embeddings in ChromaDB; answers are grounded in meaning-based search, not just keywords
- **Franchise detection** — Automatically distinguishes "Call of Duty" (franchise → top 5 games) from "Call of Duty: Black Ops 6" (single game)
- **Multi-turn chat** — Full conversation history is passed to GPT-4o so follow-up questions work naturally
- **Auth system** — JWT-based user registration and login
- **Forum** — Community forum backed by PostgreSQL
- **PDF upload** — Upload game manuals or patch notes as PDFs; they get chunked and stored in the vector database

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| AI / LLM | OpenAI GPT-4o + GPT-4o-mini |
| Vector Database | ChromaDB + OpenAI `text-embedding-3-large` |
| Relational Database | PostgreSQL |
| Frontend | React + Vite + TypeScript |
| Game Metadata | RAWG API |
| Patch Notes | Steam ISteamNews API (free, no key) |
| Community Data | Reddit JSON API (no key, browser UA) |
| ORM | SQLAlchemy |
| Auth | JWT (python-jose + passlib) |
| HTTP Client | httpx (async) |
| Containerization | Docker + Docker Compose |

---

## Architecture — How a Request Works

```
User question
    │
    ▼
1. GPT-4o-mini   → Normalize game name ("warhammer 3" → "Total War: Warhammer III")
2. GPT-4o-mini   → Is this a franchise or a single game?
3. RAWG API      → Game metadata + Steam App ID
    │
    ├── Category: updates
    │       └── Steam ISteamNews API → official patch notes
    │
    └── Category: gameplay / technical / community / general
            ├── GPT-4o-mini  → find correct subreddit
            ├── GPT-4o-mini  → extract search keywords
            ├── Reddit API   → 3-tier fallback fetch (search → top → hot)
            ├── Keyword filter → keep top 5 relevant posts
            └── ChromaDB     → store posts as vectors, semantic search
    │
    ▼
4. GPT-4o → generate final answer with all context injected
5. Return answer + Reddit posts to frontend
```

---

## Project Structure

```
project2/               ← Backend (FastAPI)
├── main.py             ← App entry point, ChromaDB setup, PDF upload
├── chat.py             ← /chat endpoint (FAQ page)
├── community_search.py ← /community/chat endpoint (Game Support page)
├── auth.py             ← /auth/register, /auth/login (JWT)
├── forum.py            ← /forum endpoints
├── models.py           ← SQLAlchemy models (User, Post, etc.)
├── database.py         ← PostgreSQL connection
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example

frontend/               ← Frontend (React + Vite + TypeScript)
├── src/
│   ├── api.ts          ← All API calls (proxied through Vite → FastAPI)
│   ├── pages/
│   │   ├── FAQPage
│   │   └── GameSupportPage
│   └── ...
└── vite.config.ts      ← Proxy: /api → localhost:8000
```

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker Desktop
- OpenAI API key
- RAWG API key (free at [rawg.io](https://rawg.io/apidocs))

### 1. Clone

```bash
git clone https://github.com/yourname/nexuscore-backend.git
cd nexuscore-backend
```

### 2. Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

### 3. Start Databases

```bash
docker compose up -d
```

This starts ChromaDB (port 8001) and PostgreSQL (port 5433).

### 4. Start Backend

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

API runs at `http://localhost:8000`

### 5. Start Frontend

```bash
cd ../frontend
npm install
npm run dev
```

App runs at `http://localhost:5173`

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Powers GPT-4o, GPT-4o-mini, and embeddings |
| `RAWG_API_KEY` | Yes | Game metadata (free tier available) |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `SECRET_KEY` | Yes | JWT signing secret |

Create a `.env` file — never commit it. See `.env.example` for the template.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | FAQ chat — game name extracted from question |
| POST | `/community/chat` | Game-specific chat with Reddit posts |
| POST | `/auth/register` | Register a new user |
| POST | `/auth/login` | Login, returns JWT |
| POST | `/upload-pdf` | Upload PDF to vector database |
| GET | `/search` | Semantic search across stored documents |
| GET | `/documents` | List all stored documents |
| POST | `/reset-db` | Clear the vector database |

---

## External APIs

| API | Key Required | Used For |
|---|---|---|
| OpenAI GPT-4o | Yes | Final answer generation |
| OpenAI GPT-4o-mini | Yes | Game name extraction, subreddit lookup, keyword extraction |
| RAWG | Yes (free) | Game metadata, genres, platforms, Steam slug |
| Steam ISteamNews | No | Official patch notes and update announcements |
| Steam Search Suggest | No | Fallback Steam App ID lookup |
| Reddit JSON | No | Community posts, guides, reviews |

---

## Deployment

| Part | Platform | Notes |
|---|---|---|
| Frontend | Vercel | Auto-detects Vite, free tier |
| Backend | Railway | Python/Docker support, free tier |
| PostgreSQL | Railway plugin | Auto-provides `DATABASE_URL` |
| ChromaDB | Railway (Docker image `chromadb/chroma`) | Add as second service |

Before deploying the frontend, update `api.ts`:
```ts
const BASE_URL = "https://your-backend.railway.app";
```

And add your Vercel domain to FastAPI's CORS `allow_origins` in `main.py`.
