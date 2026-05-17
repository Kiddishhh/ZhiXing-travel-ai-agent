"""TravelPlannerMiddleware 单元测试 — mock handler + mock LLM"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langgraph.graph.message import RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately


def _print_stage(stage: str, total: int, current: int):
    print(f"\n{'─'*50}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'─'*50}")


class TestBeforeModelPassthrough:
    """低于阈值时 abefore_model 透传不修改状态"""

    @pytest.mark.asyncio
    async def test_no_compression_under_threshold(self):
        """token 不足时不压缩，返回 None"""
        _print_stage("压缩跳过", 1, 1)
        from app.core.middleware import TravelPlannerMiddleware

        middleware = TravelPlannerMiddleware(step_config={}, compression_llm=None)

        state = {
            "messages": [
                HumanMessage(content="你好"),
                AIMessage(content="你好，有什么可以帮你？"),
            ]
        }
        result = await middleware.abefore_model(state, MagicMock())
        assert result is None
        print("[OK] 低于阈值 → 返回 None，不触发压缩")


class TestBeforeModelCompression:
    """超阈值时 abefore_model 压缩旧消息并注入摘要"""

    def _make_messages(self, count: int) -> list:
        msgs = []
        for i in range(count):
            msgs.append(HumanMessage(content=f"用户消息 {i} " + "extra " * 50))
            msgs.append(AIMessage(content=f"AI回复 {i} " + "extra " * 50))
        return msgs

    @pytest.mark.asyncio
    async def test_compresses_when_exceeds_threshold(self):
        """超阈值时返回 RemoveMessage + context_summary"""
        _print_stage("压缩触发", 1, 2)
        from app.core.middleware import TravelPlannerMiddleware

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="[摘要] 用户想去北京旅行"
        ))

        middleware = TravelPlannerMiddleware(
            step_config={},
            compression_llm=mock_llm,
            compression_max_tokens=100,
            compression_keep_recent=4,
        )

        old_msgs = self._make_messages(10)
        recent = [
            HumanMessage(content="最新的用户消息"),
            AIMessage(content="最新的AI回复"),
        ]
        state = {"messages": old_msgs + recent}

        result = await middleware.abefore_model(state, MagicMock())

        assert "messages" in result
        has_remove = any(isinstance(m, RemoveMessage) for m in result["messages"])
        assert has_remove, "应包含 RemoveMessage 删除旧消息"

        assert "context_summary" in result
        assert result["context_summary"] == "[摘要] 用户想去北京旅行"

        mock_llm.ainvoke.assert_called_once()

        recent_ids = {m.id for m in recent}
        remove_ids = {m.id for m in result["messages"] if isinstance(m, RemoveMessage)}
        assert recent_ids.isdisjoint(remove_ids), "最近消息不应被删除"
        print(f"[OK] 旧消息删除: {len(remove_ids)} 条, 摘要已生成")

    @pytest.mark.asyncio
    async def test_compression_merges_previous_summary(self):
        """已有 context_summary 时合并重压缩"""
        _print_stage("压缩合并", 2, 2)
        from app.core.middleware import TravelPlannerMiddleware

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="[合并摘要] 新旧信息合并"
        ))

        middleware = TravelPlannerMiddleware(
            step_config={},
            compression_llm=mock_llm,
            compression_max_tokens=100,
            compression_keep_recent=4,
        )

        state = {
            "messages": self._make_messages(10),
            "context_summary": "[旧摘要] 之前已确定去西安",
        }

        result = await middleware.abefore_model(state, MagicMock())

        assert "context_summary" in result
        assert result["context_summary"] == "[合并摘要] 新旧信息合并"

        call_arg = mock_llm.ainvoke.call_args[0][0]
        call_text = str(call_arg)
        assert "旧摘要" in call_text or "西安" in call_text
        print("[OK] 旧摘要已合并重压缩")

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        """LLM 调用失败时降级为简单截断，仍删除旧消息但不设置 context_summary"""
        _print_stage("压缩降级", 1, 1)
        from app.core.middleware import TravelPlannerMiddleware

        mock_llm = AsyncMock()
        print("[注入] LLM.ainvoke → Exception('API 错误')")
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API 错误"))

        middleware = TravelPlannerMiddleware(
            step_config={},
            compression_llm=mock_llm,
            compression_max_tokens=100,
            compression_keep_recent=4,
        )

        state = {"messages": self._make_messages(10)}
        result = await middleware.abefore_model(state, MagicMock())

        assert "messages" in result
        has_remove = any(isinstance(m, RemoveMessage) for m in result["messages"])
        assert has_remove, "降级时仍应删除旧消息"
        assert "context_summary" not in result
        print("[OK] 降级截断: 删除旧消息, 无摘要")


class TestWrapModelCall:
    """awrap_model_call 根据 current_step 注入 prompt 和 tools"""

    @pytest.mark.asyncio
    async def test_injects_prompt_and_tools(self):
        """awrap_model_call 根据 current_step 注入正确 prompt 和 tools"""
        _print_stage("配置注入", 3, 1)
        from app.core.middleware import TravelPlannerMiddleware

        step_config = {
            "transport_planning": {
                "prompt": "你是交通规划专家。目的地: {selected_destination}",
                "tools": ["tool_query_transport", "tool_select_transport"],
                "requires": ["user_requirement", "selected_destination"],
            }
        }

        middleware = TravelPlannerMiddleware(
            step_config=step_config,
            compression_llm=None,
        )

        state = {
            "current_step": "transport_planning",
            "user_requirement": {"destination": "西安"},
            "selected_destination": "西安",
            "messages": [HumanMessage(content="帮我查航班")],
        }

        handler = AsyncMock()
        handler.return_value = ModelResponse(
            result=[AIMessage(content="好的，查到以下航班...")]
        )

        request = ModelRequest(
            model=MagicMock(),
            messages=state["messages"],
            system_message=SystemMessage(content="默认prompt"),
            tools=[],
            state=state,
        )

        result = await middleware.awrap_model_call(request, handler)

        handler.assert_called_once()

        called_request = handler.call_args[0][0]
        assert "交通规划专家" in called_request.system_message.content
        assert "西安" in called_request.system_message.content
        assert called_request.tools == ["tool_query_transport", "tool_select_transport"]
        print("[OK] prompt 已渲染 + tools 已注入")

    @pytest.mark.asyncio
    async def test_rejects_missing_prerequisite(self):
        """前置依赖缺失时抛出 ValueError"""
        _print_stage("配置注入", 3, 2)
        from app.core.middleware import TravelPlannerMiddleware

        step_config = {
            "destination_recommendation": {
                "prompt": "你是目的地推荐专家",
                "tools": [],
                "requires": ["user_requirement"],
            }
        }

        middleware = TravelPlannerMiddleware(
            step_config=step_config,
            compression_llm=None,
        )

        state = {
            "current_step": "destination_recommendation",
            "user_requirement": None,
            "messages": [],
        }

        handler = AsyncMock()

        request = ModelRequest(
            model=MagicMock(),
            messages=state["messages"],
            system_message=SystemMessage(content=""),
            tools=[],
            state=state,
        )

        with pytest.raises(ValueError, match="需要 'user_requirement'"):
            await middleware.awrap_model_call(request, handler)

        handler.assert_not_called()
        print("[OK] 前置依赖缺失 → ValueError")

    @pytest.mark.asyncio
    async def test_prompt_rendering_fallback(self):
        """prompt 占位符无法渲染时降级为原始模板"""
        _print_stage("配置注入", 3, 3)
        from app.core.middleware import TravelPlannerMiddleware

        step_config = {
            "requirement_collection": {
                "prompt": "你是规划顾问。预算: {budget_level}",
                "tools": [],
                "requires": [],
            }
        }

        middleware = TravelPlannerMiddleware(
            step_config=step_config,
            compression_llm=None,
        )

        state = {
            "current_step": "requirement_collection",
            "messages": [],
        }

        handler = AsyncMock()
        handler.return_value = ModelResponse(
            result=[AIMessage(content="你好，请告诉我你的需求")]
        )

        request = ModelRequest(
            model=MagicMock(),
            messages=state["messages"],
            system_message=SystemMessage(content=""),
            tools=[],
            state=state,
        )

        result = await middleware.awrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert "{budget_level}" in called_request.system_message.content
        print("[OK] 占位符渲染失败 → 降级为原始模板")


class TestProfileAndContextInjection:
    """context_summary 和画像注入到 system_prompt"""

    @pytest.mark.asyncio
    async def test_appends_context_summary_to_prompt(self):
        """context_summary 存在时追加到 system_prompt"""
        _print_stage("摘要注入", 2, 1)
        from app.core.middleware import TravelPlannerMiddleware

        step_config = {
            "transport_planning": {
                "prompt": "你是交通专家",
                "tools": [],
                "requires": ["user_requirement", "selected_destination"],
            }
        }

        middleware = TravelPlannerMiddleware(
            step_config=step_config,
            compression_llm=None,
        )

        state = {
            "current_step": "transport_planning",
            "user_requirement": {"destination": "西安"},
            "selected_destination": "西安",
            "context_summary": "用户想去西安，预算5000",
            "messages": [],
        }

        handler = AsyncMock()
        handler.return_value = ModelResponse(
            result=[AIMessage(content="好的")]
        )

        request = ModelRequest(
            model=MagicMock(),
            messages=state["messages"],
            system_message=SystemMessage(content=""),
            tools=[],
            state=state,
        )

        result = await middleware.awrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert "已收集的旅行信息" in called_request.system_message.content
        assert "用户想去西安，预算5000" in called_request.system_message.content
        print("[OK] context_summary 已追加到 prompt")

    @pytest.mark.asyncio
    async def test_profile_injection_skips_on_error(self):
        """画像注入失败时静默跳过，不阻塞 LLM 调用"""
        _print_stage("摘要注入", 2, 2)
        from app.core.middleware import TravelPlannerMiddleware
        from langchain_core.messages import SystemMessage

        step_config = {
            "requirement_collection": {
                "prompt": "你是顾问",
                "tools": [],
                "requires": [],
            }
        }

        middleware = TravelPlannerMiddleware(
            step_config=step_config,
            compression_llm=None,
        )

        state = {
            "current_step": "requirement_collection",
            "user_id": "user_999",
            "messages": [],
        }

        handler = AsyncMock()
        handler.return_value = ModelResponse(
            result=[AIMessage(content="你好")]
        )

        request = ModelRequest(
            model=MagicMock(),
            messages=state["messages"],
            system_message=SystemMessage(content=""),
            tools=[],
            state=state,
        )

        result = await middleware.awrap_model_call(request, handler)

        handler.assert_called_once()
        print("[OK] 画像注入跳过 → handler 仍被正常调用")
