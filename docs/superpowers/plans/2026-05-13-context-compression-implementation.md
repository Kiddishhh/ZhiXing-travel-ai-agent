# 上下文压缩 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 main flow agent 每次调用前，基于 token 阈值自动触发 LLM 压缩历史对话为摘要，降低后续 LLM 调用的 token 消耗。

**Architecture:** 在 `graph.py` 新增 guard 节点（`START → guard → agent ⇄ tools`），guard 通过 `count_tokens_approximately` 检测 token 量，超阈值时调用 qwen-max 生成摘要，通过 `RemoveMessage` 清除旧消息并注入摘要 `SystemMessage`。压缩 LLM 失败时降级为简单截断。

**Tech Stack:** `langchain_core.messages.utils.count_tokens_approximately`, `langgraph.graph.message.RemoveMessage`, `ChatTongyi(qwen-max)`

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `app/agents/handoffs/graph.py` | 修改 | 新增常量 + guard 节点 + 更新图结构 |
| `tests/test_agents/test_context_compression.py` | 新建 | guard 节点全部测试（mock LLM） |

---

### Task 1: 编写测试文件（TDD — 红阶段）

**Files:**
- Create: `tests/test_agents/test_context_compression.py`

- [ ] **Step 1: 创建测试文件，写入全部 6 个测试用例**

```python
"""上下文压缩 guard 节点测试 — 使用 mock LLM，无需真实 API Key"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langgraph.graph.message import RemoveMessage


def make_mock_llm(response_text: str = "[压缩摘要] 测试摘要内容"):
    """创建 mock LLM，返回指定摘要文本"""
    mock = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.content = response_text
    mock.ainvoke = AsyncMock(return_value=mock_resp)
    return mock


def make_messages(count: int) -> list:
    """生成指定数量的 HumanMessage + AIMessage 交替消息对"""
    msgs = []
    for i in range(count):
        msgs.append(HumanMessage(content=f"用户消息 {i} " + "extra " * 50))
        msgs.append(AIMessage(content=f"AI回复 {i} " + "extra " * 50))
    return msgs


class TestGuardPassthrough:
    """低于阈值时 guard 透传不修改状态"""

    @pytest.mark.asyncio
    async def test_few_messages_no_compression(self):
        """消息少时返回空 dict，不触发压缩"""
        from app.agents.handoffs.graph import _make_guard_node

        llm = make_mock_llm()
        guard = _make_guard_node(llm)

        state = {
            "messages": [
                SystemMessage(content="步骤 1 的 system prompt"),
                HumanMessage(content="我想去北京"),
                AIMessage(content="好的，北京是个不错的选择"),
            ]
        }
        result = await guard(state)
        assert result == {}
        # 确认未调用压缩 LLM
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_system_messages_no_compression(self):
        """全部是 SystemMessage 时无可压缩内容，返回空 dict"""
        from app.agents.handoffs.graph import _make_guard_node

        llm = make_mock_llm()
        guard = _make_guard_node(llm)

        state = {
            "messages": [
                SystemMessage(content="步骤 prompt"),
                SystemMessage(content="历史摘要"),
            ]
        }
        result = await guard(state)
        assert result == {}


class TestGuardCompression:
    """超阈值时 guard 压缩旧消息并注入摘要"""

    @pytest.mark.asyncio
    async def test_compresses_when_exceeds_threshold(self):
        """超阈值时返回 RemoveMessage + SystemMessage(摘要)"""
        from app.agents.handoffs.graph import _make_guard_node, COMPRESSION_KEEP_RECENT

        llm = make_mock_llm("[摘要] 用户想去北京旅行，预算5000元")
        guard = _make_guard_node(llm, max_tokens=100)

        system_msg = SystemMessage(content="步骤 prompt")
        # 生成大量消息触发压缩
        old_msgs = make_messages(20)
        recent_msgs = make_messages(2)  # 最近 4 条 = 2 对

        state = {"messages": [system_msg] + old_msgs + recent_msgs}
        result = await guard(state)

        # 返回了 messages 列表
        assert "messages" in result
        result_msgs = result["messages"]

        # 包含 RemoveMessage（删除旧消息）+ SystemMessage（摘要）
        has_remove = any(isinstance(m, RemoveMessage) for m in result_msgs)
        has_summary = any(isinstance(m, SystemMessage) for m in result_msgs)
        assert has_remove, "应包含 RemoveMessage"
        assert has_summary, "应包含摘要 SystemMessage"

        # 验证 LLM 被调用
        llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_preserves_system_messages(self):
        """SystemMessage 不被压缩或删除"""
        from app.agents.handoffs.graph import _make_guard_node

        llm = make_mock_llm("[摘要] 测试")
        guard = _make_guard_node(llm, max_tokens=100)

        sys1 = SystemMessage(content="步骤 1 prompt")
        sys2 = SystemMessage(content="之前的摘要")
        old_msgs = make_messages(20)
        recent_msgs = make_messages(2)

        state = {"messages": [sys1, sys2] + old_msgs + recent_msgs}
        result = await guard(state)

        result_msgs = result["messages"]
        # 确认 SystemMessage 的 id 不在 RemoveMessage 中
        remove_ids = {m.id for m in result_msgs if isinstance(m, RemoveMessage)}
        assert sys1.id not in remove_ids
        assert sys2.id not in remove_ids

    @pytest.mark.asyncio
    async def test_keeps_recent_messages(self):
        """最近 COMPRESSION_KEEP_RECENT 条消息不被删除"""
        from app.agents.handoffs.graph import _make_guard_node, COMPRESSION_KEEP_RECENT

        llm = make_mock_llm("[摘要] 测试")
        guard = _make_guard_node(llm, max_tokens=100)

        sys_msg = SystemMessage(content="步骤 prompt")
        old_msgs = make_messages(20)
        recent = [
            HumanMessage(content="最新的用户消息"),
            AIMessage(content="最新的AI回复"),
            HumanMessage(content="再一条用户消息"),
            AIMessage(content="再一条AI回复"),
        ]

        state = {"messages": [sys_msg] + old_msgs + recent}
        result = await guard(state)

        result_msgs = result["messages"]
        remove_ids = {m.id for m in result_msgs if isinstance(m, RemoveMessage)}
        for msg in recent:
            assert msg.id not in remove_ids, f"最近消息 {msg.id} 不应被删除"


class TestGuardFallback:
    """压缩 LLM 失败时的降级行为"""

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        """LLM 调用失败时降级为简单截断，仍删除旧消息但不添加摘要"""
        from app.agents.handoffs.graph import _make_guard_node

        llm = make_mock_llm()
        llm.ainvoke = AsyncMock(side_effect=Exception("API 错误"))

        guard = _make_guard_node(llm, max_tokens=100)

        sys_msg = SystemMessage(content="步骤 prompt")
        old_msgs = make_messages(20)
        recent_msgs = make_messages(2)

        state = {"messages": [sys_msg] + old_msgs + recent_msgs}
        result = await guard(state)

        result_msgs = result["messages"]
        # 降级时仍删除旧消息
        has_remove = any(isinstance(m, RemoveMessage) for m in result_msgs)
        assert has_remove, "降级时仍应删除旧消息"

        # 降级时不添加摘要 SystemMessage
        has_summary = any(
            isinstance(m, SystemMessage) for m in result_msgs
        )
        assert not has_summary, "降级时不应添加摘要"
```

- [ ] **Step 2: 运行测试确认全部失败**

```bash
python -m pytest tests/test_agents/test_context_compression.py -v
```

Expected: 6 tests FAIL — `_make_guard_node` 尚未实现 (ImportError / AttributeError)

---

### Task 2: 实现常量 + `_make_guard_node`（TDD — 绿阶段）

**Files:**
- Modify: `app/agents/handoffs/graph.py`

- [ ] **Step 1: 在 `graph.py` 顶部 imports 区域追加以下导入**

在 `from langgraph.prebuilt import ToolNode, tools_condition` 之后追加：
```python
from langgraph.graph.message import RemoveMessage
```

在 `from langchain_core.messages import SystemMessage` 行追加 `HumanMessage`：
```python
from langchain_core.messages import SystemMessage, HumanMessage
```

追加 token 估算导入：
```python
from langchain_core.messages.utils import count_tokens_approximately
```

变更后的完整 imports 区域：
```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import RemoveMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_community.chat_models import ChatTongyi
from app.core.state import TravelState
from app.core.middleware import create_step_config_resolver, StepConfigResolver
from app.tools import TOOL_REGISTRY
from app.config import settings
from app.utils.logger import app_logger
```

- [ ] **Step 2: 在 imports 之后、`create_travel_planner` 之前，插入常量定义**

```python
# ── 上下文压缩配置 ──

COMPRESSION_MAX_TOKENS = 8000
COMPRESSION_KEEP_RECENT = 4

COMPRESSION_SYSTEM_PROMPT = """你是一个对话摘要专家。请将以下旅行规划对话压缩为简洁摘要。

压缩规则：
1. 保留所有关键事实：日期、目的地、人数、预算、已选选项（交通/住宿/餐饮）
2. 保留用户的特殊需求和偏好
3. 保留工具调用返回的具体数据（航班号、酒店名、价格、菜品等）
4. 去除闲聊、重复确认、冗余表达
5. 使用中文，不超过 500 字

待压缩对话：
{messages_to_compress}

请生成摘要："""
```

- [ ] **Step 3: 在常量定义之后、`create_travel_planner` 之前，插入 `_make_guard_node` 工厂函数**

```python
def _make_guard_node(llm: ChatTongyi, max_tokens: int = None):
    """创建 guard 节点闭包 — 每次 agent 调用前检测并压缩上下文"""

    threshold = max_tokens if max_tokens is not None else COMPRESSION_MAX_TOKENS

    async def guard_node(state: TravelState) -> dict:
        messages = list(state["messages"])
        token_count = count_tokens_approximately(messages)

        if token_count <= threshold:
            return {}

        # 分离 SystemMessage（保留不动）和可压缩消息
        system_msgs = []
        compressible = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_msgs.append(msg)
            else:
                compressible.append(msg)

        # 如果可压缩消息不足，不解压缩
        if len(compressible) <= COMPRESSION_KEEP_RECENT:
            return {}

        # 分离旧消息和最近消息
        old_msgs = compressible[:-COMPRESSION_KEEP_RECENT]

        # 构建压缩请求文本
        messages_text = "\n".join([
            f"[{type(m).__name__}]: {m.content if hasattr(m, 'content') else str(m)}"
            for m in old_msgs
        ])
        compression_prompt = COMPRESSION_SYSTEM_PROMPT.format(
            messages_to_compress=messages_text
        )

        try:
            response = await llm.ainvoke([HumanMessage(content=compression_prompt)])
            summary = response.content
            app_logger.info(f"上下文压缩完成: {len(old_msgs)} 条消息 → 摘要 ({len(summary)} 字符)")
        except Exception as e:
            app_logger.error(f"上下文压缩失败: {e}, 降级为简单截断")
            summary = None

        # 构建返回结果：RemoveMessage 删除旧消息 + 可选的摘要 SystemMessage
        result_messages = []
        for msg in old_msgs:
            msg_id = getattr(msg, 'id', None) or getattr(msg, 'message_id', None)
            if msg_id:
                result_messages.append(RemoveMessage(id=msg_id))

        if summary:
            result_messages.append(
                SystemMessage(content=f"[对话历史摘要]\n\n{summary}")
            )

        return {"messages": result_messages}

    return guard_node
```

- [ ] **Step 4: 运行测试确认 4 个通过（不含降级测试）**

```bash
python -m pytest tests/test_agents/test_context_compression.py -v -k "not fallback"
```

Expected: 4 tests PASS, 1 test FAIL (test_fallback_on_llm_error — 降级逻辑中 summary 的判断需确认)

---

### Task 3: 更新 `create_travel_planner` 注册 guard 节点

**Files:**
- Modify: `app/agents/handoffs/graph.py`

- [ ] **Step 1: 修改 `create_travel_planner` 函数，插入 guard 节点**

将现有图结构：
```python
builder = StateGraph(TravelState)
builder.add_node("agent", _make_agent_node(llm, resolver))
builder.add_node("tools", ToolNode(all_tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
```

改为：
```python
builder = StateGraph(TravelState)
builder.add_node("guard", _make_guard_node(llm))
builder.add_node("agent", _make_agent_node(llm, resolver))
builder.add_node("tools", ToolNode(all_tools))

builder.add_edge(START, "guard")
builder.add_edge("guard", "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "guard")
```

注意：`tools → agent` 改为 `tools → guard`，确保每次 agent 调用前都经过 guard 检查。

- [ ] **Step 2: 验证语法正确**

```bash
python -c "import ast; ast.parse(open('app/agents/handoffs/graph.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

---

### Task 4: 运行全量测试（TDD — 绿阶段确认）

- [ ] **Step 1: 运行 guard 测试**

```bash
python -m pytest tests/test_agents/test_context_compression.py -v
```

Expected: 6/6 tests PASS

- [ ] **Step 2: 运行已有测试确保无回归**

```bash
python -m pytest tests/test_agents/ -v
```

Expected: 所有已有测试仍然 PASS（含 test_destination_router.py 的 3 个测试）

- [ ] **Step 3: 运行全量测试**

```bash
python -m pytest -v
```

Expected: 无回归

---

### Task 5: Commit

- [ ] **Step 1: 提交所有变更**

```bash
git add app/agents/handoffs/graph.py tests/test_agents/test_context_compression.py
git commit -m "$(cat <<'EOF'
feat: add guard node for context compression in main flow

Insert guard node between START/tools and agent to detect and compress
conversation history when tokens exceed 8000. Uses qwen-max for LLM
summarization with fallback to simple truncation on failure.
EOF
)"
```
