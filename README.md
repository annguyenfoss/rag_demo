# RAG Q&A Chatbot Monorepo

- Frontend: React + TypeScript (Vite)
- Backend: Python (FastAPI, LlamaIndex, Ollama)
- Vector DB: Embedded Milvus (Milvus Lite)
- Local LLM via Ollama (`qwen3:30b`)

## Prerequisites

- Node.js 18+
- Python 3.10+
- Ollama running locally with model `qwen3:30b`
  - Install: `curl -fsSL https://ollama.com/install.sh | sh`
  - Pull: `ollama pull qwen3:30b`
  - For embeddings: `ollama pull nomic-embed-text`

## Backend

Create a virtual environment and install dependencies:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ingest data (put files in `backend/data/` first):

```bash
python ingest.py
```

Run the API server:

```bash
python main.py
```

**You will see detailed logs in your terminal including:**
- Configuration values on startup
- "FastAPI application starting/ready" messages
- Request details for each chat message
- Query responses or error messages

Backend runs on **http://localhost:9100**

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the app at **http://localhost:9200**

## Configuration

All configuration is optional with sensible defaults. To customize:

1. Copy the example environment file:
   ```bash
   # For backend configuration
   cp backend/.env.example backend/.env
   
   # Or use the root-level example
   cp .env.example .env
   ```

2. Edit `.env` with your custom values (see `.env.example` for all options)

**Available Environment Variables:**

- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (default `qwen3:30b`)
- `EMBED_MODEL` (default `nomic-embed-text`)
- `MILVUS_URI` (default `milvus.db` for Milvus Lite)
- `MILVUS_COLLECTION` (default `rag_collection`)
- `STORAGE_DIR` (default `backend/storage`)
- `DATA_DIR` (default `backend/data`)
- `ALLOWED_ORIGINS` (default `http://localhost:9200,http://127.0.0.1:9200`)

**Note:** `.env` files are excluded from git (see `.gitignore`). Only `.env.example` files are tracked.

## Notes

- Simple flat structure - all Python files in `backend/` directory
- Run everything from the `backend/` directory
- Milvus Lite stores vectors in a local file (no separate service)
- The backend will work without ingestion but answers will be generic
