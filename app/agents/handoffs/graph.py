"""
Handoffs 主流程 Graph 构建

单 agent 节点 + ToolNode + Command 跳转。

流程:
    START → agent ──┬── (有 tool_calls) → tools → agent ─┐
                     │                                     │
                     └── (无 tool_calls) → END             └── 循环

每次 LLM 调用前, agent_node 通过 StepConfigResolver 根据 current_step
注入对应 prompt + tools。工具返回 Command 后由 LangGraph 自动处理 update 和 goto。
"""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage
from langchain_community.chat_models import ChatTongyi
from app.core.state import TravelState
from app.core.middleware import create_step_config_resolver, StepConfigResolver
from app.tools import TOOL_REGISTRY
from app.config import settings
from app.utils.logger import app_logger


async def create_travel_planner() -> StateGraph:
    """
    构建 handoffs 主流程 Graph。

    图结构:
        START → agent ──┬── (有 tool_calls) → tools → agent (循环)
                         │
                         └── (无 tool_calls) → END

    返回编译后的图 (await graph.ainvoke(initial_state) 即可运行)
    """
    resolver = await create_step_config_resolver()

    llm = ChatTongyi(
        model="qwen-max",
        api_key=settings.dashscope_api_key,
    )

    # 从全局注册表收集所有工具
    all_tools = list(TOOL_REGISTRY.values())

    builder = StateGraph(TravelState)
    builder.add_node("agent", _make_agent_node(llm, resolver))
    builder.add_node("tools", ToolNode(all_tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route_after_agent)
    builder.add_edge("tools", "agent")

    app_logger.info(
        f"Handoffs 主流程 Graph 构建完成 (agent + {len(all_tools)} 个工具)"
    )
    return builder.compile()


def _route_after_agent(state: TravelState) -> str:
    """检查最后一条 AI 消息是否有 tool_calls，决定走 tools 节点还是结束"""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END


def _make_agent_node(llm: ChatTongyi, resolver: StepConfigResolver):
    """创建 agent 调用节点 (闭包捕获 llm 和 resolver 实例)"""

    async def agent_node(state: TravelState) -> dict:
        # 根据 current_step 解析 prompt + tools
        system_prompt, tools = resolver.resolve(state)

        # 将 system_prompt 作为 SystemMessage 注入消息列表
        messages = list(state["messages"])
        system_msg = SystemMessage(content=system_prompt)
        messages.insert(0, system_msg)

        # 绑定工具到 LLM，发起调用
        llm_with_tools = llm.bind_tools(tools)
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    return agent_node
