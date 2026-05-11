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

# Run agent tests
python -m pytest tests/test_agents/ -v

# Run a single test
python -m pytest tests/test_agents/test_destination_router.py::test_explore_only -v

# Check syntax of all Python files in the project
python -c "import ast, sys, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.py')]"
```

## Project Architecture

A LangGraph-based Chinese travel planning assistant with a layered design:

### Layers (bottom → top)

**1. Data Storage** (`app/core/ChromaDB/`) — `ChromaManager` wraps ChromaDB `PersistentClient` with DashScope embeddings (`text-embedding-v2`), persisted to `data/chroma_db/`.

**2. RAG Pipeline** (`app/rag/`) — Document loading (`.md` files from `data/documents/`), parent-child text splitting (parent=1000ch/200overlap, child=200ch/50overlap), hybrid retrieval (BM25+jieba + Dense + RRF fusion), and LLM reranking (Qwen-turbo pointwise scoring). Only `LLMReranker` is exported from `__init__.py`; `HybridRetriever` and `DocumentManager` are imported directly from their modules.

**3. Agent Orchestration** (`app/agents/`) — LangGraph StateGraph agents. `routers/destination_router.py` implements a Router pattern: `classifier_node` (qwen-max structured output) → `route_to_agents` (Send 并行分发) → `agent_node` (explore 走 ChromaDB 检索 / weather 占位) → `compile_report` (汇总报告)。`handoffs/` 和 `subagents/` 仍为占位。

**4. API** (`app/api/`) — FastAPI. Currently placeholder.

### Key Design Patterns

- **Settings** (`app/config.py`): pydantic-settings, loads from `.env` at module level via `get_settings()` cache.
- **Logger** (`app/utils/logger.py`): loguru with console (colorized, DEBUG+) + file rotation (JSON, INFO+) + error-only file.
- **LangGraph State**: TypedDict + `Annotated[list, add]` reducer 实现并行 Agent 结果累加；`Send` API 实现条件并行分发。
- **No test framework config yet**: No `pytest.ini` or `conftest.py`.

### Data Flow

```
# RAG Pipeline
data/documents/*.md
  → DocumentManager.load_*()
    → ParentDocumentSplitter (parent + child chunks)
      → HybridRetriever.initialize() (BM25 index + ChromaDB vector store)
        → HybridRetriever.invoke(query) (BM25 + Dense → RRF fusion)
          → LLMReranker.rerank(query, docs) (Qwen pointwise scoring)

# Agent Router
用户查询 → classifier_node (LLM 分类 explore/weather)
              → route_to_agents (Send 并行)
                → agent_node (explore: ChromaDB / weather: 占位)
              → compile_report (汇总 Markdown 报告)
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
├── agents/          # LangGraph agents (routers/ 已实现，handoffs/subagents 占位)
│   ├── routers/     # destination_router.py — 目的地查询并行路由
│   ├── handoffs/    # 占位
│   └── subagents/   # 占位
├── api/             # FastAPI routes (占位)
├── core/            # Core services (ChromaDB)
├── mcp_core/        # MCP 核心 (占位)
├── models/          # ORM/数据模型 (占位)
├── rag/             # RAG pipeline (loader, splitter, retriever, reranker)
├── schemas/         # Pydantic models (占位)
├── tools/           # MCP tools, langgraph state transition (占位)
└── utils/           # Logging, shared utilities
scripts/             # Standalone scripts (init, test)
data/
├── chroma_db/       # Persistent vector store
└── documents/       # Source markdown files (destinations, food, accommodation)
docs/superpowers/    # Design specs and implementation plans
tests/
├── test_agents/     # Agent 集成测试 (3 async tests)
├── test_rag/        # RAG 测试 (占位)
└── test_api/        # API 测试 (占位)
```
