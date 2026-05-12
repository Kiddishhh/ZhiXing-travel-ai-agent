"""
交通规划协调器
统一调度航班、高铁、自驾三个 Subagent，提供综合交通规划
"""
import asyncio
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent
from langchain_core.tools import tool
from app.config import settings
from app.mcp_core.client import get_mcp_client
from app.utils.logger import app_logger


async def _get_helper_tools():
    """获取辅助工具（日期、周边搜索等）"""
    manager = await get_mcp_client()
    all_tools = await manager.get_tools()

    helper_tools = [
        tool for tool in all_tools
        if any(keyword in tool.name.lower() for keyword in [
            'get-current-date', 'maps_around_search',
            'getfutureweather'
        ])
    ]

    app_logger.info(f"辅助工具: {[t.name for t in helper_tools]}")
    return helper_tools


# ── 子 Agent 包装为 Tool ────────────────────────────────────────

@tool
async def query_flights(origin: str, destination: str, departure_date: str) -> str:
    """
    查询航班信息（适合长途，速度快）

    Args:
        origin: 出发城市，如 "北京"
        destination: 到达城市，如 "上海"
        departure_date: 出发日期，格式 YYYY-MM-DD
    """
    from app.agents.subagents.flight_agent import create_flight_subagent

    agent = await create_flight_subagent()
    query = f"帮我查一下从{origin}到{destination}的航班，出发日期：{departure_date}"
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": query}]
    })
    result = response["messages"][-1].content
    return str(result)


@tool
async def query_trains(origin: str, destination: str, departure_date: str) -> str:
    """
    查询高铁/火车信息（适合中短途，舒适便捷）

    Args:
        origin: 出发城市，如 "北京"
        destination: 到达城市，如 "上海"
        departure_date: 出发日期，格式 YYYY-MM-DD
    """
    from app.agents.subagents.train_agent import create_train_subagent

    agent = await create_train_subagent()
    query = f"帮我查一下从{origin}到{destination}的火车票，出发日期：{departure_date}"
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": query}]
    })
    result = response["messages"][-1].content
    return str(result)


@tool
async def plan_driving_route(origin: str, destination: str, departure_date: str = "") -> str:
    """
    规划自驾路线（适合深度游，自由灵活）

    Args:
        origin: 出发地/地址，如 "北京天安门"
        destination: 目的地/地址，如 "上海东方明珠"
        departure_date: 出发日期（自驾查询可忽略此参数）
    """
    from app.agents.subagents.driving_agent import create_driving_subagent

    agent = await create_driving_subagent()
    query = f"帮我规划一下从{origin}到{destination}的自驾路线"
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": query}]
    })
    result = response["messages"][-1].content
    return str(result)


# ── 协调器创建 ──────────────────────────────────────────────────

async def create_transport_coordinator():
    """创建交通规划协调器"""

    llm = ChatTongyi(
        model=settings.qwen_model_name,
        api_key=settings.dashscope_api_key,
        temperature=0.7
    )

    # 获取辅助工具
    helper_tools = await _get_helper_tools()

    # 子 Agent 工具
    subagent_tools = [query_flights, query_trains, plan_driving_route]

    # 组合所有工具
    all_tools = helper_tools + subagent_tools

    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt="""你是交通规划协调专家。

**可用工具**：

**交通查询（主要）**：
1. query_flights：查询航班信息（适合长途，速度快）
2. query_trains：查询高铁信息（适合中短途，舒适便捷）
3. plan_driving_route：规划自驾路线（适合深度游，自由灵活）

**辅助工具（按需使用）**：
- get-current-date：获取今天日期（调用需要时间的工具前，获取实时时间）
- maps_around_search：搜索周边信息（如机场周边、火车站周边）
- getFutureWeatherByAirport：查询机场未来天气

**工作流程**：
1. 理解用户的交通需求（出发地、目的地、日期、人数）
2. 如果用户明确指定交通方式，直接调用对应工具
3. 如果用户未指定，根据距离推荐：
   * < 300km：推荐高铁
   * 300-1000km：推荐高铁或航班
   * > 1000km：推荐航班
4. 调用工具后，用清晰格式展示结果
5. 可以主动询问用户偏好（时间优先还是价格优先）

**注意事项**：
- 用户说的今天和明天之类的词要以 get-current-date 获取的时间为标准
- 一定要调用工具获取实时信息，不要编造数据
- 如果查询失败，告知用户并提供替代方案
- 航班和高铁需要提供日期，自驾不需要
- 调用工具前，先调用 get-current-date 获取今天日期
"""
    )

    app_logger.info(f"交通规划协调器创建完成 ({len(all_tools)} 个工具)")
    return agent


if __name__ == "__main__":
    async def main():
        import sys
        sys.stdout.reconfigure(encoding='utf-8')

        print("\n" + "=" * 50)
        print("正在初始化交通规划协调器...")
        print("=" * 50)

        coordinator = await create_transport_coordinator()

        test_query = "我想从北京去上海，明天出发，帮我推荐交通方式"

        print(f"\n用户提问: {test_query}")
        print("-" * 30)

        response = await coordinator.ainvoke({
            "messages": [{"role": "user", "content": test_query}]
        })

        print("-" * 30)
        print("Agent 回复:")
        final_message = response["messages"][-1].content
        print(final_message)

        print("\n" + "=" * 50)
        print("测试结束")
        print("=" * 50)

    asyncio.run(main())
