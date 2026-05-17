"""上下文压缩 guard 节点测试 — 使用 mock LLM，无需真实 API Key

已适配 TravelPlannerMiddleware (替代原来的 _make_guard_node + StateGraph)
"""
import uuid as _uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import RemoveMessage


def _print_stage(stage: str, total: int, current: int):
    print(f"\n{'─'*50}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'─'*50}")


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
        msgs.append(HumanMessage(
            content=f"用户消息 {i} " + "extra " * 50,
            id=str(_uuid.uuid4()),
        ))
        msgs.append(AIMessage(
            content=f"AI回复 {i} " + "extra " * 50,
            id=str(_uuid.uuid4()),
        ))
    return msgs


class TestGuardPassthrough:
    """低于阈值时 middleware 透传不修改状态"""

    @pytest.mark.asyncio
    async def test_few_messages_no_compression(self):
        """消息少时返回 None，不触发压缩"""
        _print_stage("Guard 透传", 1, 1)
        from app.core.middleware import TravelPlannerMiddleware

        print("[注入] 仅2条消息, 不触发压缩")
        llm = make_mock_llm()
        middleware = TravelPlannerMiddleware(step_config={}, compression_llm=llm)

        state = {
            "messages": [
                HumanMessage(content="我想去北京"),
                AIMessage(content="好的，北京是个不错的选择"),
            ]
        }
        result = await middleware.abefore_model(state, MagicMock())
        assert result is None
        llm.ainvoke.assert_not_called()



class TestGuardCompression:
    """超阈值时 middleware 压缩旧消息并注入摘要"""

    @pytest.mark.asyncio
    async def test_compresses_when_exceeds_threshold(self):
        """超阈值时返回 RemoveMessage + context_summary"""
        _print_stage("Guard 压缩", 2, 1)
        from app.core.middleware import TravelPlannerMiddleware

        llm = make_mock_llm("[摘要] 用户想去北京旅行，预算5000元")
        middleware = TravelPlannerMiddleware(
            step_config={},
            compression_llm=llm,
            compression_max_tokens=100,
            compression_keep_recent=10,
        )

        old_msgs = make_messages(20)
        recent_msgs = make_messages(2)

        state = {"messages": old_msgs + recent_msgs}
        result = await middleware.abefore_model(state, MagicMock())

        assert "messages" in result
        result_msgs = result["messages"]

        has_remove = any(isinstance(m, RemoveMessage) for m in result_msgs)
        assert has_remove, "应包含 RemoveMessage"

        # 摘要存入 context_summary 字段，而非 messages 中的 SystemMessage
        assert "context_summary" in result
        assert result["context_summary"] == "[摘要] 用户想去北京旅行，预算5000元"

        llm.ainvoke.assert_called_once()

        old_count = len(old_msgs)
        recent_count = len(recent_msgs)
        removed_count = len([m for m in result_msgs if isinstance(m, RemoveMessage)])
        print(f"[压缩] 原始消息: {old_count + recent_count}, 保留: {recent_count}, 删除: {removed_count}")


    @pytest.mark.asyncio
    async def test_keeps_recent_messages(self):
        """最近 COMPRESSION_KEEP_RECENT 条消息不被删除"""
        _print_stage("Guard 压缩", 2, 2)
        from app.core.middleware import TravelPlannerMiddleware, COMPRESSION_KEEP_RECENT

        llm = make_mock_llm("[摘要] 测试")
        middleware = TravelPlannerMiddleware(
            step_config={},
            compression_llm=llm,
            compression_max_tokens=100,
            compression_keep_recent=COMPRESSION_KEEP_RECENT,
        )

        old_msgs = make_messages(20)
        recent = [
            HumanMessage(content="最新的用户消息"),
            AIMessage(content="最新的AI回复"),
            HumanMessage(content="再一条用户消息"),
            AIMessage(content="再一条AI回复"),
        ]

        state = {"messages": old_msgs + recent}
        result = await middleware.abefore_model(state, MagicMock())

        result_msgs = result["messages"]
        remove_ids = {m.id for m in result_msgs if isinstance(m, RemoveMessage)}
        for msg in recent:
            assert msg.id not in remove_ids, f"最近消息 {msg.id} 不应被删除"


class TestGuardFallback:
    """压缩 LLM 失败时的降级行为"""

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        """LLM 调用失败时降级为简单截断，仍删除旧消息但不设置 context_summary"""
        _print_stage("Guard 降级", 1, 1)
        from app.core.middleware import TravelPlannerMiddleware

        llm = make_mock_llm()
        print("[注入] LLM.ainvoke → Exception('API 错误')")
        llm.ainvoke = AsyncMock(side_effect=Exception("API 错误"))

        middleware = TravelPlannerMiddleware(
            step_config={},
            compression_llm=llm,
            compression_max_tokens=100,
            compression_keep_recent=10,
        )

        old_msgs = make_messages(20)
        recent_msgs = make_messages(2)

        state = {"messages": old_msgs + recent_msgs}
        result = await middleware.abefore_model(state, MagicMock())

        result_msgs = result["messages"]
        has_remove = any(isinstance(m, RemoveMessage) for m in result_msgs)
        assert has_remove, "降级时仍应删除旧消息"

        # 降级时不设置 context_summary
        assert "context_summary" not in result
