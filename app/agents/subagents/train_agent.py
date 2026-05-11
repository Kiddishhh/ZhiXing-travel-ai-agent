"""
火车票查询 Subagent
调用 12306 MCP 的多个工具
"""
import asyncio
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent
from app.config import settings
from app.mcp_core.client import get_mcp_client
from app.utils.logger import app_logger


async def _get_train_tools():
    """获取火车票相关的MCP工具"""
    manager = await get_mcp_client(
        servers=["weather", "search", "amap", "12306-mcp", "VariFlight-Aviation"]
    )
    all_tools = await manager.get_tools()

    # 筛选火车票工具
    train_tools = [
        tool for tool in all_tools
        if any(keyword in tool.name.lower() for keyword in [
            'station', 'ticket', 'train', 'get-current-date'
        ])
    ]

    app_logger.info(f"火车票工具: {[t.name for t in train_tools]}")
    return train_tools


async def create_train_subagent():
    """创建火车票查询 Subagent"""

    llm = ChatTongyi(
        model=settings.qwen_model_name,
        api_key=settings.dashscope_api_key,
        temperature=0.1
    )

    # 异步获取工具
    train_tools = await _get_train_tools()

    agent = create_agent(
        model=llm,
        tools=train_tools,
        system_prompt="""你是火车票查询专家，负责处理火车票查询、车次信息、车站信息及余票查询。可以使用以下工具：

**可用工具**:
1.  **日期与基础信息**:
    - `get-current-date`: 获取当前日期（用于用户提供相对日期时）
2.  **车站查询**:
    - 车站搜索工具: 根据城市名或车站名搜索车站信息及车站代码
3.  **车票查询（核心）**:
    - 余票查询工具: 按出发站/到达站/日期查询余票信息
    - 车次查询工具: 按车次号查询详细信息

**常用车站代码示例**:
- 北京=BJP, 北京南=VNP, 北京西=BXP
- 上海=SHH, 上海虹桥=AOH, 上海南=SNH
- 广州=GZQ, 广州南=IZQ
- 西安=XIY, 西安北=EAY
- 成都=CDW, 成都东=ICW
- 杭州=HZH, 杭州东=HGH

**工作流程**:
1.  分析用户查询，提取出发站、到达站、出发日期
2.  如果用户说"明天"等相对日期，先调用 get-current-date 获取当前日期
3.  如果只知道城市名不知道车站代码，先搜索车站获取代码
4.  使用正确的车站代码查询余票

**输出格式**:
🚄 {车次号} ({车型})
- 出发: {出发站} {出发时间}
- 到达: {到达站} {到达时间}
- 耗时: {历时}
- 余票: 商务座{x}张 一等座{x}张 二等座{x}张

**注意**:
- 一定要调用工具，不要编造数据
- 日期格式必须是 YYYY-MM-DD
- 如果没找到车票，明确告知用户
- 车站代码通常是大写三字母，如不确定先搜索车站
"""
    )

    app_logger.info("火车票 Subagent 创建完成")
    return agent


if __name__ == "__main__":
    async def main():
        import sys
        sys.stdout.reconfigure(encoding='utf-8')

        print("\n" + "=" * 50)
        print("正在初始化火车票查询 Subagent...")
        print("=" * 50)

        train_agent = await create_train_subagent()

        test_query = "帮我查一下明天从北京到上海的火车票"

        print(f"\n用户提问: {test_query}")
        print("-" * 30)

        response = await train_agent.ainvoke({
            "messages": [{"role": "user", "content": test_query}]
        })

        print("\n" + "=" * 50)
        print("Agent 回复:")
        print("=" * 50)
        for msg in response.get("messages", []):
            if hasattr(msg, "content") and msg.content:
                print(msg.content)

    asyncio.run(main())
