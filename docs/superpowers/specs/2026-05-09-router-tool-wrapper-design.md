# Router Agent Tool 包装设计

## 背景

当前 `app/agents/routers/destination_router.py` 中的 Router agent 已经实现了完整的 LangGraph 工作流（分类→并行分发→结果汇总），但只能通过直接调用 `router.ainvoke()` 使用。将其包装为 LangChain `@tool`，可以在未来的 LangGraph agent 中通过 `ToolNode` 直接调用，或绑定到 LLM 进行 function calling。

## 设计

### 新增文件: `app/tools/router_query.py`

```python
from langchain_core.tools import tool
from app.agents.routers.destination_router import create_destination_router
from app.utils.logger import app_logger


@tool
async def query_destination_info(destination: str, query: str = "") -> str:
    """
    查询目的地详细信息（并行查询多个源）

    此工具会调用 Router，并行执行：
    1. 探索 Agent: 从 RAG 系统检索景点攻略
    2. 天气 Agent: 查询实时天气信息

    参数:
    - destination: 目的地名称, 如 "西安"
    - query: 具体查询 (可选), 如 "景点推荐"

    返回:
    - 综合的目的地信息 (景点 + 天气)
    """
    app_logger.info(f"调用目的地 Router: {destination}")

    router = create_destination_router()

    if not query:
        query = f"推荐{destination}旅游"

    result = await router.ainvoke({
        "original_query": query,
        "destination": destination
    })

    return result["final_report"]
```

### 更新 `app/tools/__init__.py`

```python
from .router_query import query_destination_info

__all__ = ["query_destination_info"]
```

### 依赖关系

- `langchain_core.tools.tool` — `@tool` 装饰器
- `app.agents.routers.destination_router.create_destination_router` — 复用现有 Router
- `app.utils.logger.app_logger` — 日志记录

## 关键决策

| 决策 | 选择 | 理由 |
|------|------|------|
| graph 缓存 | 不缓存，每次新建 | graph 构建纯 Python 对象，无 I/O，开销可忽略 |
| 错误处理 | 不在 tool 层捕获 | Router 内部 `_explore_agent` 已有 try/except |
| 异步 | `async def` | Router 使用 `ainvoke` |
| 返回类型 | `str` | 返回 `final_report` Markdown 字符串 |

## 不涉及

- 不修改 `app/agents/routers/destination_router.py`
- 不动 `app/tools/state_transition.py`（空占位）
- 不涉及 MCP 暴露（后续单独规划）

## 验证

1. Python 语法检查: `python -c "import ast; ast.parse(open('app/tools/router_query.py').read())"`
2. 手动导入测试: `python -c "from app.tools.router_query import query_destination_info; print(type(query_destination_info))"`
3. 调用测试: 运行 `python scripts/test_tool.py`（新建脚本，异步调用 tool.ainvoke）
