# Checkpointer 模块设计

## 概述

在 `app/core/checkpointer.py` 中创建 PostgreSQL Checkpointer 管理器，为 LangGraph agent 提供短期会话记忆（对话状态持久化）。采用单例模式管理 `AsyncConnectionPool` + `AsyncPostgresSaver`，支持 FastAPI lifespan 集成。

## 架构

```
app/core/checkpointer.py
  ├── CheckpointerManager (单例)
  │     ├── pool: AsyncConnectionPool
  │     └── checkpointer: AsyncPostgresSaver
  ├── get_checkpointer()          # 便捷函数
  └── checkpointer_lifespan()     # FastAPI lifespan
```

依赖关系：
- `app/config.py` → `settings.database_url` / `settings.db_min_conn` / `settings.db_max_conn` / `settings.db_timeout`
- `psycopg_pool.AsyncConnectionPool` → PostgreSQL 连接池
- `langgraph.checkpoint.postgres.aio.AsyncPostgresSaver` → LangGraph checkpointer

## 组件设计

### CheckpointerManager

单例模式，双重检查锁定（`asyncio.Lock`），确保全应用共享一个连接池。

**初始化流程：**
1. 创建 `AsyncConnectionPool`，使用 `settings.database_url` 和 `db_min_conn/db_max_conn/db_timeout`
2. 调用 `await self.pool.open()` 打开连接池
3. 创建 `AsyncPostgresSaver(self.pool)`
4. 调用 `await self.checkpointer.setup()` 自动建表

**关闭流程：**
- `close()` → `await self.pool.close()`

### 便捷函数

- `get_checkpointer() → AsyncPostgresSaver`：获取全局 checkpointer 实例，供 `builder.compile(checkpointer=...)` 使用
- `checkpointer_lifespan()`：`@asynccontextmanager`，用于 FastAPI lifespan，确保应用退出时关闭连接池

## Graph 集成

修改 `app/agents/handoffs/graph.py` 的 `create_travel_planner()`：

```python
async def create_travel_planner(checkpointer=None) -> StateGraph:
    ...
    graph = builder.compile(checkpointer=checkpointer)
    return graph
```

调用方：
```python
checkpointer = await get_checkpointer()
graph = await create_travel_planner(checkpointer=checkpointer)
```

## 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `database_url` | (自动拼接) | PostgreSQL 连接字符串 |
| `DB_MIN_CONN` | 2 | 连接池最小连接数 |
| `DB_MAX_CONN` | 20 | 连接池最大连接数 |
| `DB_TIMEOUT` | 30 | 连接超时（秒） |

## 错误处理

- 初始化失败时 `app_logger.error()` 记录详情并 `raise`
- 重复初始化时 `app_logger.warning()` 警告并跳过
- `get_checkpointer()` 在未初始化时抛出 `RuntimeError`

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `app/core/checkpointer.py` | **新建** | CheckpointerManager + 便捷函数 |
| `app/agents/handoffs/graph.py` | **修改** | `create_travel_planner()` 接受可选 checkpointer 参数 |
