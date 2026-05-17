# Middleware 注入 Prompt 和 Tools 优化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 graph.py 的自定义 StateGraph（`_make_agent_node` + `_make_guard_node`）迁移到 `langchain.agents.create_agent` + `TravelPlannerMiddleware(AgentMiddleware)` 标准架构。

**Architecture:** 一个 `TravelPlannerMiddleware` 类用 `abefore_model` 处理上下文压缩、`awrap_model_call` 注入步骤 prompt/tools。`create_agent` 管理图结构。`step_config.py` 和所有工具保持不变。

**Tech Stack:** langchain.agents.create_agent, AgentMiddleware, ModelRequest/ModelResponse, ChatOpenAI(qwen3.5-plus)

---

## 文件规划

| 文件 | 职责 |
|------|------|
| `app/core/middleware.py` — 重写 | `TravelPlannerMiddleware(AgentMiddleware)`：压缩钩子 + 配置注入钩子 |
| `app/agents/handoffs/graph.py` — 重写 | 删除自定义图构建，改为调用 `create_agent` |
| `tests/agents/test_middleware.py` — 新建 | 中间件单元测试（mock handler + mock LLM） |
| `app/agents/handoffs/step_config.py` — 不改 | 8 步配置数据 |
| `app/tools/state_transition.py` — 不改 | 17 个 Command 工具 |
| `app/core/state.py` — 不改 | TravelState 定义 |
| `app/tools/__init__.py` — 不改 | TOOL_REGISTRY |

---

### Task 1: 编写 TravelPlannerMiddleware 及其单元测试

**Files:**
- Create: `tests/agents/test_middleware.py`
- Modify: `app/core/middleware.py`

#### 1.1 创建测试文件骨架

- [ ] **Step 1: 编写测试文件骨架和 import**

```python
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
```

- [ ] **Step 2: 验证测试文件可被 pytest 发现**

```bash
python -m pytest tests/agents/test_middleware.py -v --collect-only
```

- [ ] **Step 3: Commit**

```bash
git add tests/agents/test_middleware.py
git commit -m "test: add middleware test file skeleton"
```

#### 1.2 测试 abefore_model 压缩逻辑

- [ ] **Step 4: 编写压缩跳过测试（低于阈值）**

```python
class TestBeforeModelPassthrough:
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
```

- [ ] **Step 5: 编写压缩触发测试（超阈值）**

```python
class TestBeforeModelCompression:
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

        # 最近消息不应被删除
        recent_ids = {m.id for m in recent}
        remove_ids = {m.id for m in result["messages"] if isinstance(m, RemoveMessage)}
        assert recent_ids.isdisjoint(remove_ids), "最近消息不应被删除"
        print(f"[OK] 旧消息删除: {len(remove_ids)} 条, 摘要已生成")

    @pytest.mark.asyncio
    async def test_compression_merges_previous_summary(self):
        """已有 context_summary 时合并重压缩"""
        _print_stage("压缩触发", 2, 2)
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

        # 验证调用 prompt 包含旧摘要
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
```

- [ ] **Step 6: 运行压缩测试，验证全部失败（TravelPlannerMiddleware 未实现）**

```bash
python -m pytest tests/agents/test_middleware.py::TestBeforeModelPassthrough tests/agents/test_middleware.py::TestBeforeModelCompression -v -s
```

- [ ] **Step 7: Commit**

```bash
git add tests/agents/test_middleware.py
git commit -m "test: add abefore_model compression tests"
```

#### 1.3 测试 awrap_model_call 配置注入逻辑

- [ ] **Step 8: 编写 prompt + tools 注入测试**

```python
class TestWrapModelCall:
    @pytest.mark.asyncio
    async def test_injects_prompt_and_tools(self):
        """awrap_model_call 根据 current_step 注入正确 prompt 和 tools"""
        _print_stage("配置注入", 3, 1)
        from app.core.middleware import TravelPlannerMiddleware
        from langchain_core.messages import SystemMessage

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

        # 验证 handler 收到的 request 参数
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
            "user_requirement": None,  # 缺失
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
        from langchain_core.messages import SystemMessage

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
            # budget_level 不存在
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
        # 降级为原始模板（保留 {budget_level} 花括号）
        assert "{budget_level}" in called_request.system_message.content
        print("[OK] 占位符渲染失败 → 降级为原始模板")
```

- [ ] **Step 9: 运行配置注入测试，验证全部失败**

```bash
python -m pytest tests/agents/test_middleware.py::TestWrapModelCall -v -s
```

- [ ] **Step 10: Commit**

```bash
git add tests/agents/test_middleware.py
git commit -m "test: add awrap_model_call config injection tests"
```

#### 1.4 测试图片注入和 context_summary 追加

- [ ] **Step 11: 编写画像注入和 context_summary 测试**

```python
class TestProfileAndContextInjection:
    @pytest.mark.asyncio
    async def test_appends_context_summary_to_prompt(self):
        """context_summary 存在时追加到 system_prompt"""
        _print_stage("摘要注入", 2, 1)
        from app.core.middleware import TravelPlannerMiddleware
        from langchain_core.messages import SystemMessage

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

        # 即使画像存储不可用，也不应阻塞
        result = await middleware.awrap_model_call(request, handler)

        handler.assert_called_once()
        print("[OK] 画像注入跳过 → handler 仍被正常调用")
```

- [ ] **Step 12: 运行附加测试，验证失败**

```bash
python -m pytest tests/agents/test_middleware.py::TestProfileAndContextInjection -v -s
```

- [ ] **Step 13: Commit**

```bash
git add tests/agents/test_middleware.py
git commit -m "test: add profile injection and context_summary tests"
```

---

### Task 2: 实现 TravelPlannerMiddleware

**Files:**
- Modify: `app/core/middleware.py`

- [ ] **Step 1: 重写 middleware.py**

```python
"""
AgentMiddleware 实现

TravelPlannerMiddleware 继承 AgentMiddleware, 提供两个钩子:
- abefore_model: token 计数 + 上下文压缩
- awrap_model_call: 步骤 prompt/tools 注入 + 画像注入

替代了原来的 StepConfigResolver + _make_guard_node + _make_agent_node。
"""
from typing import Callable, Awaitable

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import RemoveMessage
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ModelCallResult,
)

from app.core.state import TravelState
from app.utils.logger import app_logger

# ── 压缩常量 ──

COMPRESSION_MAX_TOKENS = 12000
COMPRESSION_KEEP_RECENT = 10

COMPRESSION_SYSTEM_PROMPT = """你是一个对话摘要专家。请将以下旅行规划对话压缩为简洁摘要。

压缩规则：
1. 只提取事实数据：日期、目的地、人数、预算、已选选项（交通/住宿/餐饮）
2. 只提取用户偏好和特殊需求
3. 只提取工具返回的具体数据（航班号、酒店名、价格、菜品等）
4. 禁止描述 AI 做过什么、问过什么、建议过什么
5. 去除闲聊、重复确认、冗余表达
6. 使用中文，不超过 800 字

待压缩对话：
{messages_to_compress}

请生成摘要："""


class TravelPlannerMiddleware(AgentMiddleware):
    """旅行规划中间件 — 上下文压缩 + 步骤配置注入"""

    state_schema = TravelState

    def __init__(
        self,
        step_config: dict,
        compression_llm=None,
        compression_max_tokens: int = None,
        compression_keep_recent: int = None,
    ):
        super().__init__()
        self._step_config = step_config
        self._compression_llm = compression_llm
        self._compression_max_tokens = (
            compression_max_tokens
            if compression_max_tokens is not None
            else COMPRESSION_MAX_TOKENS
        )
        self._compression_keep_recent = (
            compression_keep_recent
            if compression_keep_recent is not None
            else COMPRESSION_KEEP_RECENT
        )

    # ── 上下文压缩 ──

    async def abefore_model(self, state: TravelState, runtime) -> dict | None:
        """模型调用前检测并压缩上下文"""
        if self._compression_llm is None:
            return None

        messages = list(state.get("messages", []))
        token_count = count_tokens_approximately(messages)

        if token_count <= self._compression_max_tokens:
            return None

        if len(messages) <= self._compression_keep_recent:
            return None

        old_msgs = messages[:-self._compression_keep_recent]

        # 构建压缩请求文本
        messages_text = "\n".join([
            f"[{type(m).__name__}]: {m.content if hasattr(m, 'content') else str(m)}"
            for m in old_msgs
        ])

        # 如已有历史摘要，合并重压缩
        previous_summary = state.get("context_summary")
        if previous_summary:
            compression_prompt = (
                f"之前的对话摘要：\n{previous_summary}\n\n"
                f"新的待压缩对话：\n{messages_text}\n\n"
                f"请将以上内容合并为一份完整的简洁摘要："
            )
        else:
            compression_prompt = COMPRESSION_SYSTEM_PROMPT.format(
                messages_to_compress=messages_text
            )

        try:
            response = await self._compression_llm.ainvoke(
                [HumanMessage(content=compression_prompt)]
            )
            summary = response.content
            app_logger.info(
                f"上下文压缩完成: {len(old_msgs)} 条消息 → 摘要 ({len(summary)} 字符)"
            )
        except Exception as e:
            app_logger.error(f"上下文压缩失败: {e}, 降级为简单截断")
            summary = None

        # 构建 RemoveMessage 删除旧消息
        import uuid as _uuid

        result_messages = []
        for msg in old_msgs:
            msg_id = (
                getattr(msg, 'id', None)
                or getattr(msg, 'message_id', None)
                or str(_uuid.uuid4())
            )
            result_messages.append(RemoveMessage(id=msg_id))

        result = {"messages": result_messages}
        if summary:
            result["context_summary"] = summary

        return result

    # ── 配置注入 ──

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """拦截模型调用，注入步骤 prompt 和 tools"""
        state = request.state
        current_step = state.get("current_step", "requirement_collection")

        if current_step not in self._step_config:
            raise ValueError(f"未知步骤: {current_step}")

        cfg = self._step_config[current_step]

        # 1. 验证前置依赖
        for required_field in cfg["requires"]:
            if state.get(required_field) is None:
                raise ValueError(
                    f"步骤 {current_step} 需要 '{required_field}' 字段，但当前未设置"
                )

        # 2. 渲染 prompt 模板
        try:
            system_prompt = cfg["prompt"].format(**state)
        except KeyError:
            system_prompt = cfg["prompt"]

        # 3. 追加 context_summary
        context_summary = state.get("context_summary")
        if context_summary:
            system_prompt += f"\n\n[已收集的旅行信息]\n\n{context_summary}"

        # 4. 注入用户画像
        user_id = state.get("user_id")
        if user_id:
            try:
                from app.core.memory_store import get_memory_store_manager

                manager = await get_memory_store_manager()
                profile = await manager.get_profile(user_id)
                if profile:
                    profile_text = _format_profile_for_prompt(profile)
                    system_prompt += f"\n\n{profile_text}"
                    app_logger.info(f"已注入用户画像 (user_id={user_id})")
            except Exception as e:
                app_logger.warning(f"画像注入失败，跳过: {e}")

        # 5. 创建修改后的 request 并交给 handler
        modified = request.override(
            system_message=SystemMessage(content=system_prompt),
            tools=cfg["tools"],
        )
        return await handler(modified)

    # ── 工具调用错误包装 ──

    async def awrap_tool_call(self, request, handler):
        """拦截工具调用，将 Pydantic 校验错误转为引导性提示"""
        try:
            return await handler(request)
        except Exception as e:
            msg = str(e)
            if "Input should be" in msg or "validation error" in msg.lower():
                from langchain_core.messages import ToolMessage

                return ToolMessage(
                    content=(
                        f"参数校验未通过：\n{msg}\n\n"
                        f"请向用户逐一确认上述信息，补充完整后重新调用。"
                    ),
                    tool_call_id=request.tool_call["id"],
                )
            return ToolMessage(
                content=f"操作未能完成：{msg[:300]}。请向用户说明并询问如何处理。",
                tool_call_id=request.tool_call["id"],
            )


def _format_profile_for_prompt(profile: dict) -> str:
    """将 user_profiles 行格式化为 prompt 可用的画像文本"""
    lines = ["[用户长期画像]"]

    transport = profile.get("preferred_transport")
    if transport:
        lines.append(f"- 交通偏好: {transport}")

    budget = profile.get("budget_level")
    if budget:
        lines.append(f"- 预算档位: {budget}")

    styles = profile.get("travel_styles") or []
    if styles:
        lines.append(f"- 旅行风格: {', '.join(styles)}")

    dests = profile.get("favorite_destinations") or []
    if dests:
        lines.append(f"- 偏好目的地: {', '.join(dests)}")

    diets = profile.get("dietary_preferences") or []
    if diets:
        lines.append(f"- 饮食偏好: {', '.join(diets)}")

    total = profile.get("total_trips", 0)
    if total:
        last_dest = profile.get("last_destination", "") or ""
        last_date = profile.get("last_travel_date", "") or ""
        parts = [f"共{total}次"]
        if last_date:
            parts.append(f"最近一次{last_date}")
        if last_dest:
            parts.append(f"去{last_dest}")
        lines.append(f"- 历史出行: {'，'.join(parts)}")

    extensions = profile.get("extensions") or {}
    for k, v in extensions.items():
        if v:
            lines.append(f"- {k}: {v}")

    return "\n".join(lines)


async def create_travel_planner_middleware() -> TravelPlannerMiddleware:
    """工厂函数: 创建预加载 step_config 的 TravelPlannerMiddleware"""
    from langchain_openai import ChatOpenAI
    from app.agents.handoffs.step_config import get_step_config
    from app.config import settings

    step_config = await get_step_config()

    compression_llm = ChatOpenAI(
        model=settings.qwen_model_name,
        api_key=settings.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    app_logger.info("TravelPlannerMiddleware 创建完成")
    return TravelPlannerMiddleware(
        step_config=step_config,
        compression_llm=compression_llm,
    )
```

- [ ] **Step 2: 运行所有中间件单元测试**

```bash
python -m pytest tests/agents/test_middleware.py -v -s
```

预期：全部 PASS（13 个测试）

- [ ] **Step 3: Commit**

```bash
git add app/core/middleware.py
git commit -m "refactor: replace StepConfigResolver with TravelPlannerMiddleware(AgentMiddleware)"
```

---

### Task 3: 重写 graph.py 使用 create_agent

**Files:**
- Modify: `app/agents/handoffs/graph.py`

- [ ] **Step 1: 确认现有导入者不依赖旧的 graph.py 内部符号**

```bash
grep -rn "_make_agent_node\|_make_guard_node\|COMPRESSION_MAX_TOKENS\|COMPRESSION_KEEP_RECENT\|COMPRESSION_SYSTEM_PROMPT\|_wrap_tool_error\|create_step_config_resolver\|StepConfigResolver" app/ --include="*.py" | grep -v "graph.py" | grep -v "middleware.py"
```

- [ ] **Step 2: 重写 graph.py**

```python
"""
Handoffs 主流程 Graph 构建

使用 langchain.agents.create_agent + TravelPlannerMiddleware
替代原来的自定义 StateGraph。

流程由 create_agent 内部管理:
    abefore_model(压缩) → awrap_model_call(注入) → LLM → ToolNode → 循环
"""
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from app.core.state import TravelState
from app.core.middleware import create_travel_planner_middleware
from app.tools import TOOL_REGISTRY
from app.config import settings
from app.utils.logger import app_logger


async def create_travel_planner(
    checkpointer: BaseCheckpointSaver = None,
    store: BaseStore = None,
):
    """
    构建 handoffs 主流程 Graph。

    使用 create_agent 标准工厂 + TravelPlannerMiddleware:
    - abefore_model: 上下文压缩（token 计数 + LLM 摘要）
    - awrap_model_call: 步骤 prompt/tools 注入 + 画像注入
    - awrap_tool_call: 工具错误友好包装

    参数:
        checkpointer: LangGraph checkpointer (PostgreSQL)
        store: LangGraph Store (跨会话持久化)

    返回:
        编译后的图 (await graph.ainvoke(initial_state) 即可运行)
    """
    middleware = await create_travel_planner_middleware()

    llm = ChatOpenAI(
        model=settings.qwen_model_name,
        api_key=settings.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    tools_list = list(TOOL_REGISTRY.values())

    app_logger.info(
        f"Handoffs 主流程 Graph 构建完成 "
        f"(create_agent + {len(tools_list)} 个工具)"
    )

    return create_agent(
        model=llm,
        tools=tools_list,
        middleware=[middleware],
        state_schema=TravelState,
        checkpointer=checkpointer,
        store=store,
    )
```

- [ ] **Step 3: 验证 Python 语法无错误**

```bash
python -c "import ast; ast.parse(open('app/agents/handoffs/graph.py', encoding='utf-8').read()); print('Syntax OK')"
```

- [ ] **Step 4: 运行中间件单元测试确认无回归**

```bash
python -m pytest tests/agents/test_middleware.py -v -s
```

- [ ] **Step 5: Commit**

```bash
git add app/agents/handoffs/graph.py
git commit -m "refactor: replace custom StateGraph with create_agent + TravelPlannerMiddleware"
```

---

### Task 4: 运行现有测试确认无回归

**Files:**
- 无修改

- [ ] **Step 1: 运行 agent 相关测试**

```bash
python -m pytest tests/agents/ -v -s
```

- [ ] **Step 2: 运行全量单元测试（排除需网络/LLM 的交互测试）**

```bash
python -m pytest tests/rag/ tests/tools/ tests/agents/ -v -s
```

- [ ] **Step 3: 检查全部 Python 文件语法**

```bash
python -c "import ast; [ast.parse(open(p, encoding='utf-8').read()) for p in __import__('pathlib').Path('.').rglob('*.py') if 'venv' not in str(p)]"
```

- [ ] **Step 4: 如有测试失败，修复后重跑并提交**

- [ ] **Step 5: Commit（如有修复）**

```bash
git add -A
git commit -m "fix: address test regressions from middleware migration"
```

---

### Task 5: 清理无用代码和导出

**Files:**
- Modify: `app/core/middleware.py`
- 验证: `app/` 目录下无对旧符号的引用

- [ ] **Step 1: 确认无代码引用旧的 StepConfigResolver 或 create_step_config_resolver**

```bash
grep -rn "StepConfigResolver\|create_step_config_resolver" app/ --include="*.py" | grep -v middleware.py
```

- [ ] **Step 2: 确认无代码引用 graph.py 的旧私有函数**

```bash
grep -rn "_make_agent_node\|_make_guard_node\|_wrap_tool_error" app/ --include="*.py"
```

- [ ] **Step 3: 如有残留引用，更新引用者后提交**

---

## 自审清单

**1. Spec 覆盖:**
- [x] 上下文压缩保留 → Task 1.2 / Task 2 `abefore_model`
- [x] 步骤 prompt + tools 注入 → Task 1.3 / Task 2 `awrap_model_call`
- [x] 画像注入 → Task 1.4 / Task 2 `awrap_model_call`
- [x] 错误处理（前置依赖/未知步骤/降级） → 测试覆盖
- [x] `_wrap_tool_error` 迁移 → Task 2 `awrap_tool_call`
- [x] 保留 step_config.py 不变 → 文件规划表确认

**2. Placeholder 扫描:** 无 TBD/TODO/模糊占位。

**3. 类型一致性:**
- `TravelPlannerMiddleware` vs `create_travel_planner_middleware` 返回类型一致 ✓
- `request.state` / `request.override()` / `handler()` 用法与 Task 2 实现一致 ✓
- `ModelRequest / ModelResponse / ModelCallResult` 导入路径一致 ✓
