"""
Router Agent Tool 包装

将 destination_router LangGraph 工作流包装为 LangChain @tool,
可通过 ToolNode 直接调用或绑定到 LLM 做 function calling。
"""
from langchain_core.tools import tool

from app.agents.routers.destination_router import create_destination_router
from app.utils.logger import app_logger


@tool
async def query_destination_info(destination: str, query: str = "") -> str:
    """
    查询目的地详细信息（并行查询多个源）

    此工具会调用 Router，并行执行：
    1. 探索 Agent: 从 RAG 系统检索景点攻略
    2. 天气 Agent: 查询实时天气信息

    参数:
    - destination: 目的地名称, 如 "西安"
    - query: 具体查询 (可选), 如 "景点推荐"

    返回:
    - 综合的目的地信息 (景点 + 天气)
    """
    app_logger.info(f"调用目的地 Router: {destination}")

    router = create_destination_router()

    if not query:
        query = f"推荐{destination}旅游"

    result = await router.ainvoke({
        "original_query": query,
        "destination": destination
    })

    brake = (
        "\n\n---\n"
        "⚠️ 请将以上目的地信息整理后用简洁的语言向用户展示（每个目的地 2-3 句话），"
        "列出推荐理由后等待用户选择。用户明确确认目的地之后，再调用 select_destination_tool。"
    )
    return result["final_report"] + brake
