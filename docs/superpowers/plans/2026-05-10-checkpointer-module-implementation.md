# Checkpointer 模块实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建 PostgreSQL Checkpointer 模块，为 LangGraph agent 提供会话状态持久化能力。

**Architecture:** 新建 `app/core/checkpointer.py`（CheckpointerManager 单例 + 便捷函数 + lifespan），修改 `graph.py` 的 `create_travel_planner()` 接受可选 checkpointer 参数。

**Tech Stack:** psycopg_pool.AsyncConnectionPool, langgraph.checkpoint.postgres.aio.AsyncPostgresSaver

---

### Task 1: 创建 Checkpointer 模块

**Files:**
- Create: `app/core/checkpointer.py`

- [ ] **Step 1: 创建 checkpointer.py**

```python
"""
PostgreSQL Checkpointer 管理器

为 LangGraph agent 提供短期会话记忆（对话状态持久化）。
单例模式管理 AsyncConnectionPool + AsyncPostgresSaver。
"""
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings
from app.utils.logger import app_logger


class CheckpointerManager:
    """Checkpointer 管理器（单例模式）"""

    _instance: Optional["CheckpointerManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.pool: Optional[AsyncConnectionPool] = None
        self.checkpointer: Optional[AsyncPostgresSaver] = None

    @classmethod
    async def get_instance(cls) -> "CheckpointerManager":
        """获取单例实例（异步安全）"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize()
        return cls._instance

    async def initialize(self):
        """初始化连接池和 Checkpointer"""
        if self.checkpointer is not None:
            app_logger.warning("Checkpointer 已初始化，跳过")
            return

        try:
            app_logger.info("初始化 PostgreSQL Checkpointer...")

            self.pool = AsyncConnectionPool(
                conninfo=settings.database_url,
                min_size=settings.db_min_conn,
                max_size=settings.db_max_conn,
                timeout=settings.db_timeout,
            )

            await self.pool.open()

            self.checkpointer = AsyncPostgresSaver(self.pool)
            await self.checkpointer.setup()

            app_logger.info("Checkpointer 初始化完成")
        except Exception as e:
            app_logger.error(f"Checkpointer 初始化失败: {e}")
            raise

    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            app_logger.info("Checkpointer 连接池已关闭")

    def get_checkpointer(self) -> AsyncPostgresSaver:
        """获取 Checkpointer 实例"""
        if self.checkpointer is None:
            raise RuntimeError("Checkpointer 未初始化，请先调用 initialize()")
        return self.checkpointer


async def get_checkpointer() -> AsyncPostgresSaver:
    """获取全局 Checkpointer 实例"""
    manager = await CheckpointerManager.get_instance()
    return manager.get_checkpointer()


@asynccontextmanager
async def checkpointer_lifespan():
    """Checkpointer 生命周期管理器（用于 FastAPI lifespan）"""
    manager = await CheckpointerManager.get_instance()
    try:
        yield manager.get_checkpointer()
    finally:
        await manager.close()
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/core/checkpointer.py', encoding='utf-8').read()); print('Syntax OK')"
```

- [ ] **Step 3: 提交**

```bash
git add app/core/checkpointer.py
git commit -m "feat: add CheckpointerManager singleton for PostgreSQL session persistence"
```

---

### Task 2: 集成 Checkpointer 到 Graph

**Files:**
- Modify: `app/agents/handoffs/graph.py`

- [ ] **Step 1: 修改 `create_travel_planner()` 签名和 compile 调用**

将 `graph.py` 中的函数签名从：

```python
async def create_travel_planner() -> StateGraph:
```

改为：

```python
from langgraph.checkpoint.base import BaseCheckpointSaver

async def create_travel_planner(checkpointer: BaseCheckpointSaver = None):
```

将 `return builder.compile()` 改为：

```python
    return builder.compile(checkpointer=checkpointer)
```

注意：这里 `create_travel_planner` 的返回类型不再是 `StateGraph` 而是 `CompiledStateGraph`，移除返回类型注解或改为 `CompiledStateGraph`。

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/agents/handoffs/graph.py', encoding='utf-8').read()); print('Syntax OK')"
```

- [ ] **Step 3: 验证图编译**

```bash
python -c "
import asyncio, sys
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from app.agents.handoffs.graph import create_travel_planner
async def test():
    graph = await create_travel_planner()  # 无 checkpointer 也能编译
    print(f'Graph compiled: {graph}')
asyncio.run(test())
"
```

- [ ] **Step 4: 提交**

```bash
git add app/agents/handoffs/graph.py
git commit -m "feat: wire checkpointer into create_travel_planner"
```
