"""
Handoffs 主流程 Graph 构建

单 agent 节点 + ToolNode + Command 跳转。

流程:
    START → guard → agent ──┬── (有 tool_calls) → tools → guard ─┐
                              │                                     │
                              └── (无 tool_calls) → END             └── 循环

每次 LLM 调用前, agent_node 通过 StepConfigResolver 根据 current_step
注入对应 prompt + tools。工具返回 Command 后由 LangGraph 自动处理 update 和 goto。
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import RemoveMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI
from app.core.state import TravelState
from app.core.middleware import create_step_config_resolver, StepConfigResolver
from app.tools import TOOL_REGISTRY
from app.config import settings
from app.utils.logger import app_logger


def _wrap_tool_error(error: Exception) -> str:
    """将工具调用异常包装为引导性提示，让 LLM 自然向用户补充提问"""
    msg = str(error)
    if "Input should be" in msg or "validation error" in msg.lower():
        return (
            f"参数校验未通过：\n{msg}\n\n"
            f"请向用户逐一确认上述信息，补充完整后重新调用。"
        )
    return f"操作未能完成：{msg[:300]}。请向用户说明并询问如何处理。"


# ── 上下文压缩配置 ──

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


def _make_guard_node(llm: ChatOpenAI, max_tokens: int = None):
    """创建 guard 节点闭包 — 每次 agent 调用前检测并压缩上下文"""

    threshold = max_tokens if max_tokens is not None else COMPRESSION_MAX_TOKENS

    async def guard_node(state: TravelState) -> dict:
        messages = list(state["messages"])
        token_count = count_tokens_approximately(messages)

        if token_count <= threshold:
            return {}

        # 如果消息不足，不压缩
        if len(messages) <= COMPRESSION_KEEP_RECENT:
            return {}

        # 分离：旧消息(压缩) + 最近消息(保留)
        old_msgs = messages[:-COMPRESSION_KEEP_RECENT]

        # 构建压缩请求文本
        messages_text = "\n".join([
            f"[{type(m).__name__}]: {m.content if hasattr(m, 'content') else str(m)}"
            for m in old_msgs
        ])

        # 如果已有历史摘要，合并重压缩
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
            response = await llm.ainvoke([HumanMessage(content=compression_prompt)])
            summary = response.content
            app_logger.info(
                f"上下文压缩完成: {len(old_msgs)} 条消息 → 摘要 ({len(summary)} 字符)"
            )
        except Exception as e:
            app_logger.error(f"上下文压缩失败: {e}, 降级为简单截断")
            summary = None

        # 构建返回结果：RemoveMessage 删除旧消息
        import uuid as _uuid

        result_messages = []
        for msg in old_msgs:
            msg_id = getattr(msg, 'id', None) or getattr(msg, 'message_id', None)
            if msg_id is None:
                msg_id = str(_uuid.uuid4())
            result_messages.append(RemoveMessage(id=msg_id))

        result = {"messages": result_messages}
        if summary:
            result["context_summary"] = summary

        return result

    return guard_node


async def create_travel_planner(checkpointer: BaseCheckpointSaver = None):
    """
    构建 handoffs 主流程 Graph。

    图结构:
        START → guard → agent ──┬── (有 tool_calls) → tools → guard (循环)
                                  │
                                  └── (无 tool_calls) → END

    返回编译后的图 (await graph.ainvoke(initial_state) 即可运行)
    """
    resolver = await create_step_config_resolver()

    llm = ChatOpenAI(
        model="qwen3.5-plus",
        api_key=settings.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 从全局注册表收集所有工具
    all_tools = list(TOOL_REGISTRY.values())

    builder = StateGraph(TravelState)
    builder.add_node("guard", _make_guard_node(llm))
    builder.add_node("agent", _make_agent_node(llm, resolver))
    builder.add_node("tools", ToolNode(all_tools, handle_tool_errors=_wrap_tool_error))

    builder.add_edge(START, "guard")
    builder.add_edge("guard", "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "guard")

    app_logger.info(
        f"Handoffs 主流程 Graph 构建完成 (agent + {len(all_tools)} 个工具)"
    )
    return builder.compile(checkpointer=checkpointer)


def _make_agent_node(llm: ChatOpenAI, resolver: StepConfigResolver):
    """创建 agent 调用节点 (闭包捕获 llm 和 resolver 实例)"""

    async def agent_node(state: TravelState) -> dict:
        # 根据 current_step 解析 prompt + tools
        system_prompt, tools = await resolver.resolve(state)

        # 构建注入 LLM 的三层消息（临时，不存入 state）
        messages = []

        # 第1层: 指令层 — 当前步骤 prompt（优先级最高，最先被 LLM 读取）
        messages.append(SystemMessage(content=system_prompt))

        # 第2层: 记忆层 — 已收集的旅行事实数据（辅助参考）
        context_summary = state.get("context_summary")
        if context_summary:
            messages.append(
                SystemMessage(content=f"[已收集的旅行信息]\n\n{context_summary}")
            )

        # 第3层: 对话层 — 交互历史
        messages.extend(state["messages"])

        # 绑定工具到 LLM，发起调用
        llm_with_tools = llm.bind_tools(tools)
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    return agent_node
