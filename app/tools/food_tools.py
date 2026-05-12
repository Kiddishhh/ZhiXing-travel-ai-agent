"""
餐饮查询工具
调用 Amap MCP（周边搜索）+ Tavily MCP（美食攻略）
"""
from langchain.tools import tool
from app.mcp_core.client import get_mcp_client
from app.utils.logger import app_logger


@tool
async def query_food(
    destination: str,
    food_type: str = None,
    query: str = None,
) -> str:
    """
    查询餐饮选项（餐厅/本地小吃/美食攻略）

    参数说明:
    - destination: 目的地城市，如 "西安"
    - food_type: 餐饮类型（可选）。可选值: restaurant（餐厅）, local_snack（本地小吃）。不传则查询全部
    - query: 自定义搜索词（可选），如 "回民街美食"

    返回:
    - 格式化的餐饮推荐（包含 Amap 周边结果和美食攻略）
    """
    app_logger.info(f"🍜 查询餐饮: {destination}, type={food_type}, query={query}")

    manager = await get_mcp_client()
    all_tools = await manager.get_tools()

    # 筛选工具
    geo_tool = None
    around_tool = None
    search_tool = None

    for t in all_tools:
        name = t.name.lower()
        if 'maps_geo' in name:
            geo_tool = t
        elif 'maps_around_search' in name:
            around_tool = t
        elif 'search' in name:
            search_tool = t

    results = []

    # 确定搜索关键词
    if food_type == "restaurant":
        around_keyword = query or f"{destination} 餐厅"
        search_query = f"{destination} 餐厅推荐 必吃榜"
    elif food_type == "local_snack":
        around_keyword = query or f"{destination} 小吃"
        search_query = f"{destination} 本地小吃 特色美食攻略"
    else:
        around_keyword = query or f"{destination} 美食"
        search_query = f"{destination} 美食攻略 必吃推荐"

    # 1. Amap 周边搜索
    if geo_tool and around_tool:
        try:
            geo_result = await geo_tool.ainvoke({"address": destination})
            geo_str = str(geo_result) if not isinstance(geo_result, str) else geo_result
            app_logger.info(f"Geocoding 结果: {geo_str[:200]}")

            around_params = {
                "keywords": around_keyword,
                "city": destination,
            }
            around_result = await around_tool.ainvoke(around_params)
            around_str = str(around_result) if not isinstance(around_result, str) else around_result
            if around_str.strip():
                results.append(f"## 🗺️ 周边餐饮\n{around_str}")
        except Exception as e:
            app_logger.warning(f"Amap 餐饮搜索失败: {e}")

    # 2. Tavily 美食攻略搜索
    if search_tool:
        try:
            search_result = await search_tool.ainvoke({"query": search_query})
            search_str = str(search_result) if not isinstance(search_result, str) else search_result
            if search_str.strip():
                results.append(f"## 📝 美食攻略\n{search_str}")
        except Exception as e:
            app_logger.warning(f"Tavily 美食搜索失败: {e}")

    if not results:
        return f"未找到 {destination} 的餐饮推荐，请尝试更具体的搜索词。"

    return "\n\n".join(results)
