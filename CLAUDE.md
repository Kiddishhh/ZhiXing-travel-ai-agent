# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run RAG initialization (load docs → split → BM25 + ChromaDB index)
python scripts/init_rag.py

# Initialize Postgres Checkpointer + Store tables
python scripts/init_db.py

# Run all unit tests (no external services required)
python -m pytest tests/rag/ tests/tools/ tests/agents/ -v -s

# Run RAG unit tests
python -m pytest tests/rag/ -v -s

# Run tools unit tests
python -m pytest tests/tools/ -v -s

# Run agent unit tests
python -m pytest tests/agents/ -v -s

# Run a single test
python -m pytest tests/agents/test_context_compression.py::TestGuardCompression::test_compresses_when_exceeds_threshold -v

# Interactive tests (require real LLM/MCP/API)
python tests/interactive/interactive_llm.py
python tests/interactive/interactive_rag.py
python tests/interactive/interactive_flow.py
python tests/interactive/interactive_destination.py
python tests/interactive/interactive_mcp.py
python tests/interactive/interactive_weather.py
python tests/interactive/interactive_search.py
python tests/interactive/interactive_transport.py
python tests/interactive/interactive_accommodation.py
python tests/interactive/interactive_food.py

# Check syntax of all Python files
python -c "import ast; [ast.parse(open(p, encoding='utf-8').read()) for p in __import__('pathlib').Path('.').rglob('*.py') if 'venv' not in str(p)]"
```

## Architecture

A LangGraph-based Chinese travel planning assistant with an 8-step sequential workflow, context compression, and MCP-based tool integration.

### Main Graph (`app/agents/handoffs/graph.py`)

Built with `langchain.agents.create_agent` + `TravelPlannerMiddleware`. The internal loop is managed by `create_agent`:

```
abefore_model(压缩) → awrap_model_call(注入 prompt/tools) → LLM → ToolNode → 循环
```

Three middleware hooks in `TravelPlannerMiddleware(AgentMiddleware)`:
- **abefore_model** — Token-aware context compression. Counts tokens via `count_tokens_approximately`, compresses old messages into a `context_summary` when exceeding 12000 tokens. Keeps last 10 messages. Fallback to truncation on LLM failure.
- **awrap_model_call** — Reads `current_step`, validates prerequisites, renders prompt template, injects user profile, appends `context_summary`, then calls `request.override(system_message=..., tools=...)` to set step-specific prompt and tools.
- **awrap_tool_call** — Wraps Pydantic validation errors into friendly guidance prompts; re-raises non-validation errors.

### Context Compression (`middleware.py`)

`TravelPlannerMiddleware.abefore_model()` — runs before every model call. When token count exceeds `COMPRESSION_MAX_TOKENS` (12000), splits messages into old (compress) and recent (keep 10), calls `ChatOpenAI` to generate a facts-only summary (≤800 chars, no AI behavior description), removes old messages via `RemoveMessage`, and stores summary in `state["context_summary"]`. Previous summaries are merged on re-compression.

### State (`app/core/state.py`)

`TravelState(AgentState)` — `messages` holds pure conversation (Human/AI/Tool only, no SystemMessages). Key fields:
- `current_step` — one of 8 `PlanningStep` values
- `context_summary` — compressed history (set by guard, used ephemerally by agent)
- `user_requirement` — dict with `budget_level` and `travel_styles` as `Optional` (tool auto-computes them)
- `STEP_CLEANUP_MAP` — defines which fields to nullify per step on rollback

### 8-Step Workflow

Defined in `app/core/state.py` as `PlanningStep` literal:

| # | Step Key | Purpose |
|---|----------|---------|
| 1 | `requirement_collection` | Gather user needs |
| 2 | `destination_recommendation` | Recommend 3 destinations |
| 3 | `transport_planning` | Flight/train/driving |
| 4 | `accommodation_planning` | Hotels/hostels |
| 5 | `food_planning` | Restaurants/local food |
| 6 | `itinerary_generation` | LLM generates daily itinerary |
| 7 | `budget_summarization` | Aggregate + validate costs |
| 8 | `order_generation` | Final order, ends flow |

### Step Configuration System

`app/agents/handoffs/step_config.py` — Each step's prompt has three sections:
- **⚠️ 关键规则** — hard gate prohibiting auto-advancing without user confirmation
- **📋 查询工具 / 🔒 确认工具 / ↩️ 回退工具** — tools categorized by permission level
- **任务** — step-specific instructions

`app/core/middleware.py` — `TravelPlannerMiddleware(AgentMiddleware)` reads `current_step`, validates prerequisites, renders `{field}` placeholders from state, injects `context_summary` and user profile, then uses `request.override()` to set step-specific `system_message` and `tools`.

### State Transition Tools (`app/tools/state_transition.py`)

17 tools use `Command(update={...})` **without `goto`** — routing is handled by `create_agent`'s internal loop. The only exception: `generate_order_tool` uses `goto="__end__"` to terminate the graph. All transition tool ToolMessages contain "brake signals" that guide the LLM to introduce the next step to the user and wait for confirmation.

### Query Tools

Data-fetching tools return `str` (Markdown-formatted results):
- `query_destination_info` — RAG + weather via destination router
- `query_transport_options` — flight/train/driving via transport coordinator
- `query_accommodation` — hotels via aigohotel-mcp
- `query_food` — Amap POI + Tavily search (direct httpx API calls)
- `calculate_budget` / `create_order` — compute from TravelState

### Transport Layer

`app/agents/subagents/transport_coordinator.py` — Supervisor wrapping three subagents:
- `flight_agent.py` — VariFlight-Aviation MCP
- `train_agent.py` — 12306 MCP (ModelScope-hosted)
- `driving_agent.py` — Amap MCP (geocoding + directions)

### MCP Core (`app/mcp_core/`)

`MCPClientManager` (singleton) manages 6 MCP servers: weather (stdlib), search (stdlib), amap (HTTP), 12306-mcp (streamable_http), VariFlight-Aviation (streamable_http), aigohotel-mcp (streamable_http).

### LLM Configuration

All LLM calls use `ChatOpenAI` with DashScope's OpenAI-compatible endpoint:
```python
ChatOpenAI(
    model="qwen3.6-flash",
    api_key=settings.dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```
Model name default in `app/config.py` → `settings.qwen_model_name`. Structured output in destination router uses `method="function_calling"` to avoid `json_object` mode issues.

### 测试原则

- **测试必须可视化流程**：每个测试用例必须包含 `print()` 输出关键步骤和模拟输入注入，让运行者能看到完整测试流水线，而不是沉默地 pass
- **模拟输入的可见注入**：mock 对象的返回值、side_effect 设置必须在 print 中明确展示（如 `print("[注入] optimizer.optimize → strategy=none")`）
- **分阶段标注**：用 `[1/4]`, `[2/4]` 等标注管线阶段，`[OK]` 标注断言通过
- **运行方式**：用 `python -m pytest tests/test_rag/ -v -s`（`-s` 不捕获 stdout）来查看测试流程
- **对比输出**：管线测试应打印输入和输出的对比，让人直观看到数据如何流转

### Key Patterns

- **Settings**: pydantic-settings from `.env`, cached `get_settings()`
- **Logger**: loguru — console (colorized), file rotation (JSON, 10MB/7-day), error-only file
- **State updates**: `Annotated[list, add]` reducer for parallel results; `Command(update)` without goto for step transitions
- **Error handling**: `awrap_tool_call` in middleware.py wraps Pydantic validation errors as guidance prompts, re-raises other exceptions; fallback compression on LLM failure
- **Testing**: `pytest` + `pytest-asyncio` (`@pytest.mark.asyncio`), mock LLM via `AsyncMock`

### Known Issues

- Duplicate `TransportState` definitions in `app/core/transport_state.py` and `app/agents/subagents/transport_state.py`
- MCP servers (aigohotel-mcp, 12306-mcp) have intermittent connectivity issues — accommodation/transport tests may fail on network errors
- DashScope free tier quota limits — `test_destination_router.py` may fail with 403 when quota exhausted

### Environment

- Python >= 3.11, package manager: `uv`
- `.env` required: `DASHSCOPE_API_KEY`, `LANGSMITH_API_KEY`, Postgres + Redis connection strings, `AMAP_API_KEY`, `TAVILY_API_KEY`, `VARIFLIGHT_API_KEY`, `AIGOHOTEL_MCP_API`
- Services: PostgreSQL (checkpointer), Redis, ChromaDB (local file)
- On Windows, scripts set `asyncio.WindowsSelectorEventLoopPolicy()`
