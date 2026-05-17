"""
AgentMiddleware 实现

TravelPlannerMiddleware 继承 AgentMiddleware, 提供三个钩子:
- abefore_model: token 计数 + 上下文压缩
- awrap_model_call: 步骤 prompt/tools 注入 + 画像注入
- awrap_tool_call: 工具调用错误包装

替代了原来的 StepConfigResolver + _make_guard_node + _make_agent_node。
"""
from typing import Callable, Awaitable

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import RemoveMessage
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langgraph.prebuilt.tool_node import ToolCallRequest

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

        messages_text = "\n".join([
            f"[{type(m).__name__}]: {m.content if hasattr(m, 'content') else str(m)}"
            for m in old_msgs
        ])

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
    ) -> ModelResponse:
        """拦截模型调用，注入步骤 prompt 和 tools"""
        state = request.state
        current_step = state.get("current_step", "requirement_collection")

        if current_step not in self._step_config:
            raise ValueError(f"未知步骤: {current_step}")

        cfg = self._step_config[current_step]

        for required_field in cfg["requires"]:
            if state.get(required_field) is None:
                raise ValueError(
                    f"步骤 {current_step} 需要 '{required_field}' 字段，但当前未设置"
                )

        try:
            system_prompt = cfg["prompt"].format(**state)
        except KeyError:
            system_prompt = cfg["prompt"]

        context_summary = state.get("context_summary")
        if context_summary:
            system_prompt += f"\n\n[已收集的旅行信息]\n\n{context_summary}"

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

        modified = request.override(
            system_message=SystemMessage(content=system_prompt),
            tools=cfg["tools"],
        )
        return await handler(modified)

    # ── 工具调用错误包装 ──

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        """拦截工具调用，将 Pydantic 校验错误转为引导性提示"""
        try:
            return await handler(request)
        except Exception as e:
            msg = str(e)
            if "Input should be" in msg or "validation error" in msg.lower():
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
