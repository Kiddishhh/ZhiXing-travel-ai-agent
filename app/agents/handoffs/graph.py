"""
Handoffs 主流程 Graph 构建

单 agent 节点 + Command 跳转。
每次 LLM 调用前, AgentMiddleware 根据 current_step 注入对应 prompt + tools。
工具返回 Command 直接跳回 agent 节点 (或 __end__ 终止)。
"""
from langgraph.graph import StateGraph, START
from langchain_community.chat_models import ChatTongyi
from app.core.state import TravelState
from app.core.middleware import create_step_config_middleware
from app.config import settings
from app.utils.logger import app_logger


async def create_travel_planner() -> StateGraph:
    """
    构建 handoffs 主流程 Graph。

    图结构:
        START → agent → END
                  ↑   │
                  │   │ LLM 调用工具 → Command(goto="agent" / "__end__")
                  └───┘

    返回编译后的图 (await graph.ainvoke(initial_state) 即可运行)
    """
    middleware = await create_step_config_middleware()

    llm = ChatTongyi(
        model="qwen-max",
        api_key=settings.dashscope_api_key,
    )

    builder = StateGraph(TravelState)
    builder.add_node(
        "agent",
        _make_agent_node(llm),
        middleware=[middleware],
    )
    builder.add_edge(START, "agent")

    app_logger.info("Handoffs 主流程 Graph 构建完成")
    return builder.compile()


def _make_agent_node(llm: ChatTongyi):
    """创建 agent 调用节点 (闭包捕获 llm 实例)"""

    async def agent_node(state: TravelState) -> dict:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    return agent_node
