# Router Agent Tool 包装 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 destination_router agent 包装为 LangChain `@tool`，使其可被 LangGraph ToolNode 调用

**Architecture:** 在 `app/tools/router_query.py` 中新建一个 `@tool` 装饰的异步函数，内部调用 `create_destination_router().ainvoke()`，返回 `final_report` Markdown 字符串。更新 `app/tools/__init__.py` 导出。

**Tech Stack:** `langchain-core` (tool decorator), 现有 `destination_router` (LangGraph StateGraph)

---

### Task 1: 创建 router_query.py 工具文件

**Files:**
- Create: `app/tools/router_query.py`

- [ ] **Step 1: 写入 router_query.py**

```python
"""
Router Agent Tool 包装

将 destination_router LangGraph 工作流包装为 LangChain @tool,
可通过 ToolNode 直接调用或绑定到 LLM 做 function calling。
"""
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

- [ ] **Step 2: 验证语法正确**

```bash
python -c "import ast; ast.parse(open('app/tools/router_query.py', encoding='utf-8').read()); print('语法检查通过')"
```
Expected: `语法检查通过`

- [ ] **Step 3: 提交**

```bash
git add app/tools/router_query.py
git commit -m "feat: add query_destination_info tool wrapping destination router"
```

---

### Task 2: 更新 tools/__init__.py 导出

**Files:**
- Modify: `app/tools/__init__.py` (当前为空)

- [ ] **Step 1: 写入 __init__.py**

```python
from .router_query import query_destination_info

__all__ = ["query_destination_info"]
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from app.tools import query_destination_info; print(f'导入成功: {query_destination_info.name}')"
```
Expected: `导入成功: query_destination_info`

- [ ] **Step 3: 提交**

```bash
git add app/tools/__init__.py
git commit -m "feat: export query_destination_info from tools package"
```

---

### Task 3: 端到端验证

**Files:**
- Create: `scripts/test_tool.py`（验证后删除）

- [ ] **Step 1: 创建临时验证脚本**

```python
"""
验证 query_destination_info 工具端到端可用
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.tools import query_destination_info


async def main():
    # 验证 1: tool 元数据
    print(f"Tool 名称: {query_destination_info.name}")
    print(f"Tool 描述: {query_destination_info.description[:50]}...")
    print(f"Tool 参数: {query_destination_info.args_schema.model_json_schema()['properties']}")
    print()

    # 验证 2: 实际调用
    print("调用工具: query_destination_info('西安', '景点推荐')")
    result = await query_destination_info.ainvoke({
        "destination": "西安",
        "query": "有什么好玩的景点"
    })
    print(f"\n返回结果 (前200字符):\n{result[:200]}...")
    print(f"\n结果长度: {len(result)} 字符")

    # 验证 3: 不传 query 参数
    print("\n--- 测试不传 query ---")
    result2 = await query_destination_info.ainvoke({"destination": "杭州"})
    print(f"返回结果 (前200字符):\n{result2[:200]}...")

    print("\n所有验证通过!")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 运行验证脚本**

```bash
python scripts/test_tool.py
```
Expected: 输出 tool 名称/描述/参数信息 + 两次 Router 调用结果

- [ ] **Step 3: 删除临时验证脚本**

```bash
rm scripts/test_tool.py
```

- [ ] **Step 4: 运行已有测试确保无回归**

```bash
python -m pytest tests/test_agents/test_destination_router.py -v
```
Expected: 3 tests PASS

- [ ] **Step 5: 最终提交**

```bash
git add .
git commit -m "test: verify query_destination_info tool end-to-end"
```
