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
        extra_body={"enable_thinking": False},
        max_retries=2,
        request_timeout=30.0,
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
