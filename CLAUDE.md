# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run RAG initialization (load docs → split → BM25 + ChromaDB index)
python scripts/init_rag.py

# Test RAG retrieval + reranking
python scripts/test_rag.py

# Test Qwen LLM connection
python scripts/test_llm.py

# Initialize Postgres Checkpointer + Store tables
python scripts/init_db.py

# Run all tests
python -m pytest

# Run agent tests
python -m pytest tests/test_agents/ -v

# Run MCP integration tests
python -m pytest tests/test_mcp/ -v

# Run a single test
python -m pytest tests/test_agents/test_destination_router.py::test_explore_only -v

# Check syntax of all Python files
python -c "import ast, sys, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.py')]"
```

## Architecture

A LangGraph-based Chinese travel planning assistant with an 8-step sequential workflow and MCP-based tool integration.

### Two Graph Systems

**1. Handoffs Main Flow** (`app/agents/handoffs/graph.py`) — The primary agent. Single `agent` node + `ToolNode` loop (`START → agent ⇄ tools → END`). Uses `StepConfigResolver` middleware to inject step-specific prompts and tools at each turn based on `current_step`. The agent progresses through 8 steps by calling state transition tools that return `Command(goto=...)`.

**2. Destination Router** (`app/agents/routers/destination_router.py`) — Parallel dispatch. LLM classifier → `Send` API parallel dispatch → explore (ChromaDB RAG) / weather agents → compile markdown report. Used as a tool (`query_destination_info`) by the main flow's step 2.

### 8-Step Workflow

Defined in `app/core/state.py` as `PlanningStep` literal, with `STEP_CLEANUP_MAP` for rollback:

| # | Step Key | Purpose |
|---|----------|---------|
| 1 | `requirement_collection` | Gather user needs (departure, dates, budget, style) |
| 2 | `destination_recommendation` | Recommend 3 destinations via RAG |
| 3 | `transport_planning` | Flight/train/driving via transport coordinator |
| 4 | `accommodation_planning` | Hotels/hostels |
| 5 | `food_planning` | Restaurants/local food |
| 6 | `itinerary_generation` | LLM generates daily itinerary JSON |
| 7 | `budget_summarization` | Aggregate costs, validate against budget |
| 8 | `order_generation` | Final order, ends flow |

### Step Configuration System

`app/agents/handoffs/step_config.py` — Central config hub. `get_step_config()` returns a dict with 8 step configs, each containing:
- `prompt`: System prompt template with `{field}` placeholders rendered from state
- `tools`: List of callable tools available at this step
- `requires`: State fields that must exist before this step can run

`app/core/middleware.py` — `StepConfigResolver` reads `current_step` from state, validates prerequisites, renders the prompt template, and returns the prompt + tools for the LLM call.

### Transport Layer (Coordinator + Subagents)

`app/agents/subagents/transport_coordinator.py` — Supervisor agent that wraps three subagents as `@tool` functions (`query_flights`, `query_trains`, `plan_driving_route`). Called by the unified `query_transport_options` tool in step 3.

Each subagent filters relevant tools from the MCP client:
- `flight_agent.py` — VariFlight-Aviation MCP tools
- `train_agent.py` — 12306 MCP tools (ModelScope-hosted)
- `driving_agent.py` — Amap MCP tools (geocoding + directions)

### MCP Core (`app/mcp_core/`)

`MCPClientManager` (singleton) manages 6 MCP servers:

| Server | Transport | Purpose |
|--------|-----------|---------|
| `weather` | stdio (local FastMCP) | Amap weather API |
| `search` | stdio (local FastMCP) | Tavily search API |
| `amap` | HTTP | Amap geocoding/directions |
| `12306-mcp` | streamable_http | 12306 train tickets (ModelScope) |
| `VariFlight-Aviation` | streamable_http | Flight data |
| `aigohotel-mcp` | streamable_http | Accommodation search |

`get_mcp_client()` (no args) returns the singleton with all 6 servers. Pass `servers=[...]` to limit.

### Tools Registry

`app/tools/__init__.py` — `TOOL_REGISTRY` dict with `register_tool()`. All 24 tools registered here; the handoffs graph collects all values as the ToolNode. State transition tools in `app/tools/state_transition.py` use LangGraph's `Command(goto=...)` pattern to advance steps.

### RAG Pipeline

```
data/documents/*.md → DocumentManager → ParentDocumentSplitter (parent 1000ch/child 200ch)
  → HybridRetriever (BM25 + Dense + RRF fusion, k=60)
    → LLMReranker (Qwen-turbo pointwise scoring, top_k=5)
```

### Key Design Patterns

- **Settings** (`app/config.py`): pydantic-settings, loaded from `.env` via `get_settings()` cache.
- **Logger** (`app/utils/logger.py`): loguru — console (colorized, DEBUG+), file rotation (JSON, INFO+, 10MB/7-day), error-only file.
- **State**: `TravelState(MessagesState)` — all fields `NotRequired`. `Command(goto=...)` drives step transitions. `Annotated[list, add]` reducer for parallel agent results in the router.
- **Testing**: `pytest` + `pytest-asyncio` (`@pytest.mark.asyncio`). No `pytest.ini` or `conftest.py`. MCP singleton tests use `MCPClientManager.reset_instance()` fixture.

### Known Issues

- `app/tools/accommodation_tools.py`, `budget_tools.py`, `food_tools.py`, `order_tools.py` have been cleared (0 bytes) but are still imported in `app/tools/__init__.py` and referenced in `app/agents/handoffs/step_config.py`. These imports will fail at runtime.
- `TOOL_REGISTRY` still contains entries (`query_hotels`, `query_hostels`, `query_restaurants`, `query_local_food`, `calculate_budget`, `create_order`) whose implementations no longer exist.
- Duplicate `TransportState` definitions in both `app/core/transport_state.py` and `app/agents/subagents/transport_state.py`.

### Environment

- Python >= 3.11, package manager: `uv`
- Dependencies in `pyproject.toml` + `uv.lock`
- `.env` required with: `DASHSCOPE_API_KEY`, `LANGSMITH_API_KEY`, Postgres + Redis connection strings, `AMAP_API_KEY`, `TAVILY_API_KEY`, `VARIFLIGHT_API_KEY`, `AIGOHOTEL_MCP_API`
- Other services: PostgreSQL (checkpointer), Redis, ChromaDB (local file)
- On Windows, scripts set `asyncio.WindowsSelectorEventLoopPolicy()`
