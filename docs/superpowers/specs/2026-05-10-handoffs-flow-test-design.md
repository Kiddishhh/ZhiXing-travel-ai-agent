# Handoffs Flow 交互式测试脚本设计

## 概述

在 `tests/handoffs_flow_test.py` 中创建交互式 CLI 测试脚本，运行完整 8 步行规划流程。通过 `python tests/handoffs_flow_test.py` 直接启动。

## 架构

```
main()
  ├── 1. 生成 UUID session_id / user_id
  ├── 2. 初始化 Checkpointer (PostgreSQL)
  ├── 3. 编译 Graph (带 checkpointer)
  ├── 4. 构建初始 State + 首条 HumanMessage
  ├── 5. while True 循环 ─── 6. graph.astream(stream_mode="values")
  └── 7. 关闭 Checkpointer
```

## 组件设计

### 会话标识

```python
session_id = str(uuid.uuid4())
user_id = "test_user"
```

`config = {"configurable": {"thread_id": session_id}}` 传给 `astream`，checkpointer 按 `thread_id` 隔离会话。

### Checkpointer 初始化

```python
from app.core.checkpointer import get_checkpointer
checkpointer = await get_checkpointer()
```

启动时自动创建连接池 + 建表。

### Graph 编译

```python
from app.agents.handoffs.graph import create_travel_planner
graph = await create_travel_planner(checkpointer=checkpointer)
```

### 首次调用

```python
initial_state = create_initial_state(user_id, session_id)
initial_state["messages"].append(HumanMessage(content=first_input))
async for event in graph.astream(initial_state, config, stream_mode="values"):
    # event 是完整的 TravelState dict
```

### 后续调用

```python
# 只需要传入新消息，checkpointer 自动从 PostgreSQL 恢复历史状态
update = {"messages": [HumanMessage(content=user_input)]}
async for event in graph.astream(update, config, stream_mode="values"):
    pass
```

### Stream 输出

`stream_mode="values"` 每次 yield 当前完整 `TravelState`。输出信息：
- `current_step` — 当前步骤
- `messages[-1]` — 最新消息（LLM 回复 / ToolMessage）

### 退出

输入 `quit` / `exit` 退出循环，`finally` 中关闭 checkpointer。

## 错误处理

- PostgreSQL 不可达时 `get_checkpointer()` 抛出异常，脚本终止并提示检查数据库
- LLM API 异常由 LangGraph 内部处理，不特殊处理

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `tests/handoffs_flow_test.py` | **新建** | 交互式测试脚本 |
