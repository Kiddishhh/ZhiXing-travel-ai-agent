# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run RAG initialization
python scripts/init_rag.py

# Test RAG retrieval + reranking
python scripts/test_rag.py

# Test Qwen LLM connection
python scripts/test_llm.py

# Initialize Postgres Checkpointer + Store
python scripts/init_db.py

# Run all tests
python -m pytest

# Run a specific test file
python -m pytest tests/test_rag/

# Check syntax of all Python files in the project
python -c "import ast, sys, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').rglob('*.py')]"
```

## Project Architecture

A LangGraph-based Chinese travel planning assistant with a layered design:

### Layers (bottom → top)

**1. Data Storage** (`app/core/ChromaDB/`) — `ChromaManager` wraps ChromaDB `PersistentClient` with DashScope embeddings (`text-embedding-v2`), persisted to `data/chroma_db/`.

**2. RAG Pipeline** (`app/rag/`) — Document loading (`.md` files from `data/documents/`), parent-child text splitting (parent=1000ch/200overlap, child=200ch/50overlap), hybrid retrieval (BM25+jieba + Dense + RRF fusion), and LLM reranking (Qwen-turbo pointwise scoring).

**3. Agent Orchestration** (`app/agents/`) — LangGraph-based agents. Currently placeholder structure with `handoffs/`, `routers/`, `subagents/` subpackages awaiting implementation.

**4. API** (`app/api/`) — FastAPI. Currently placeholder.

### Key Design Patterns

- **Settings** (`app/config.py`): pydantic-settings, loads from `.env` at module level via `get_settings()` cache.
- **Logger** (`app/utils/logger.py`): loguru with console (colorized, DEBUG+) + file rotation (JSON, INFO+) + error-only file.
- **No test framework config yet**: No `pytest.ini` or `conftest.py`.

### Data Flow

```
data/documents/*.md
  → DocumentManager.load_*()
    → ParentDocumentSplitter (parent + child chunks)
      → HybridRetriever.initialize() (BM25 index + ChromaDB vector store)
        → HybridRetriever.invoke(query) (BM25 + Dense → RRF fusion)
          → LLMReranker.rerank(query, docs) (Qwen pointwise scoring)
```

### Environment

- Python >= 3.11, package manager: `uv`
- Dependencies in `pyproject.toml` + `uv.lock`
- `.env` required with at minimum: `DASHSCOPE_API_KEY`, `LANGSMITH_API_KEY`, Postgres + Redis connection strings
- Other services: PostgreSQL (checkpointer), Redis, ChromaDB (local file)
- On Windows, scripts set `asyncio.WindowsSelectorEventLoopPolicy()`

### Directory Layout

```
app/
├── agents/          # LangGraph agents (placeholder)
├── api/             # FastAPI routes (placeholder)
├── core/            # Core services (ChromaDB)
├── rag/             # RAG pipeline (loader, splitter, retriever, reranker)
├── schemas/         # Pydantic models (placeholder)
├── tools/           # MCP tools, langgraph state transition
└── utils/           # Logging, shared utilities
scripts/             # Standalone scripts (init, test)
data/
├── chroma_db/       # Persistent vector store
└── documents/       # Source markdown files (destinations, food, accommodation)
docs/superpowers/    # Design specs and implementation plans
tests/               # Test stubs (no real tests yet)
```
