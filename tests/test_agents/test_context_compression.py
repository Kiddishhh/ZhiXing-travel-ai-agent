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
    """生成指定数量的 HumanMessage + AIMessage 交替消息对，带填充使 token 数显著增长"""
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
        from app.agents.handoffs.graph import _make_guard_node

        llm = make_mock_llm("[摘要] 用户想去北京旅行，预算5000元")
        guard = _make_guard_node(llm, max_tokens=100)

        system_msg = SystemMessage(content="步骤 prompt")
        old_msgs = make_messages(20)
        recent_msgs = make_messages(2)

        state = {"messages": [system_msg] + old_msgs + recent_msgs}
        result = await guard(state)

        assert "messages" in result
        result_msgs = result["messages"]

        has_remove = any(isinstance(m, RemoveMessage) for m in result_msgs)
        has_summary = any(isinstance(m, SystemMessage) for m in result_msgs)
        assert has_remove, "应包含 RemoveMessage"
        assert has_summary, "应包含摘要 SystemMessage"

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
        has_remove = any(isinstance(m, RemoveMessage) for m in result_msgs)
        assert has_remove, "降级时仍应删除旧消息"

        has_summary = any(
            isinstance(m, SystemMessage) for m in result_msgs
        )
        assert not has_summary, "降级时不应添加摘要"
