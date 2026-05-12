"""
住宿查询工具
调用 aigohotel-mcp 查询酒店/民宿
"""
from langchain_core.tools import tool
from app.mcp_core.client import get_mcp_client
from app.utils.logger import app_logger


@tool
async def query_accommodation(
    destination: str,
    check_in_date: str,
    stay_nights: int,
    accommodation_type: str = None,
    budget_min: float = 0,
    budget_max: float = 99999,
) -> str:
    """
    查询住宿选项（酒店/民宿/青旅）

    参数说明:
    - destination: 目的地城市，如 "北京"
    - check_in_date: 入住日期，格式 YYYY-MM-DD
    - stay_nights: 入住天数
    - accommodation_type: 住宿类型（可选）。可选值: hotel, hostel。不传则查询全部
    - budget_min: 最低预算（可选，默认 0）
    - budget_max: 最高预算（可选，默认 99999）

    返回:
    - 格式化的住宿选项信息（名称、价格、评分、预订链接）
    """
    app_logger.info(f"🏨 查询住宿: {destination}, {check_in_date}, {stay_nights}晚, type={accommodation_type}")

    manager = await get_mcp_client()
    all_tools = await manager.get_tools()

    # 筛选 aigohotel 工具
    hotel_tools = [
        t for t in all_tools
        if any(kw in t.name.lower() for kw in ['searchhotels', 'gethoteldetail'])
    ]

    search_tool = None
    for t in hotel_tools:
        if 'searchhotels' in t.name.lower():
            search_tool = t
            break

    if search_tool is None:
        return "⚠️ 住宿查询服务暂不可用，请稍后重试。"

    # 构建查询参数
    search_params = {
        "place": destination,
        "checkInDate": check_in_date,
        "stayNights": stay_nights,
    }

    # 映射住宿类型
    if accommodation_type == "hotel":
        search_params["starRating"] = "3,4,5"
    elif accommodation_type == "hostel":
        search_params["tags"] = "hostel,youth_hostel"

    app_logger.info(f"调用 searchHotels: {search_params}")

    try:
        result = await search_tool.ainvoke(search_params)
        result_str = str(result) if not isinstance(result, str) else result

        if not result_str or result_str.strip() == "":
            return f"未找到 {destination} 的{'酒店' if accommodation_type == 'hotel' else '住宿'}选项，请调整日期或预算后重试。"

        return result_str
    except Exception as e:
        app_logger.error(f"住宿查询失败: {e}")
        return f"住宿查询出错: {str(e)}"


# ── 向后兼容的包装函数 ──
# step_config.py 和 __init__.py 中引用的是 query_hotels / query_hostels

@tool
async def query_hotels(
    destination: str,
    check_in_date: str,
    stay_nights: int = 1,
) -> str:
    """
    查询酒店选项

    参数说明:
    - destination: 目的地城市
    - check_in_date: 入住日期，格式 YYYY-MM-DD
    - stay_nights: 入住天数

    返回:
    - 格式化的酒店选项信息
    """
    return await query_accommodation.ainvoke({
        "destination": destination,
        "check_in_date": check_in_date,
        "stay_nights": stay_nights,
        "accommodation_type": "hotel",
    })


@tool
async def query_hostels(
    destination: str,
    check_in_date: str,
    stay_nights: int = 1,
) -> str:
    """
    查询民宿/青旅选项

    参数说明:
    - destination: 目的地城市
    - check_in_date: 入住日期，格式 YYYY-MM-DD
    - stay_nights: 入住天数

    返回:
    - 格式化的民宿/青旅选项信息
    """
    return await query_accommodation.ainvoke({
        "destination": destination,
        "check_in_date": check_in_date,
        "stay_nights": stay_nights,
        "accommodation_type": "hostel",
    })
