"""
餐饮查询工具
调用 Amap MCP（周边搜索）+ Tavily MCP（美食攻略）
"""
import httpx
from app.config import settings

# ── API 端点常量 ──
AMAP_GEO_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_AROUND_URL = "https://restapi.amap.com/v5/place/around"
TAVILY_URL = "https://api.tavily.com/search"
POI_TYPE_FOOD = "050000"
SEARCH_RADIUS = "10000"
REQUEST_TIMEOUT = 15.0

from langchain.tools import tool
from app.mcp_core.client import get_mcp_client
from app.utils.logger import app_logger


# ── 辅助函数 ──

async def _geocode(client: httpx.AsyncClient, address: str) -> str | None:
    """地理编码：结构化地址 → 经纬度坐标，失败返回 None"""
    try:
        resp = await client.get(AMAP_GEO_URL, params={
            "address": address,
            "key": settings.amap_api_key,
        })
        if resp.status_code != 200:
            app_logger.warning(f"地理编码 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        if data.get("status") == "1" and data.get("geocodes"):
            return data["geocodes"][0]["location"]
        return None
    except (httpx.HTTPError, ValueError, LookupError) as e:
        app_logger.warning(f"地理编码失败: {e}")
        return None


async def _search_poi(
    client: httpx.AsyncClient, location: str, keyword: str
) -> list[dict]:
    """POI 周边搜索：坐标 + 关键词 → 结构化餐厅列表，失败返回 []"""
    try:
        resp = await client.get(AMAP_AROUND_URL, params={
            "location": location,
            "keywords": keyword,
            "types": POI_TYPE_FOOD,
            "show_fields": "business,photos",
            "radius": SEARCH_RADIUS,
            "page_size": 10,
            "key": settings.amap_api_key,
        })
        if resp.status_code != 200:
            app_logger.warning(f"POI搜索 HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        data = resp.json()
        if data.get("status") == "1" and data.get("pois"):
            pois = []
            for p in data["pois"]:
                biz = p.get("business") or {}
                pois.append({
                    "name": p.get("name", ""),
                    "address": p.get("address", ""),
                    "type": p.get("type", ""),
                    "tel": biz.get("tel", ""),
                    "opentime": biz.get("opentime", ""),
                    "photos": [ph.get("url", "") for ph in (p.get("photos") or []) if ph.get("url")],
                    "location": p.get("location", ""),
                })
            return pois
        return []
    except (httpx.HTTPError, ValueError, LookupError) as e:
        app_logger.warning(f"POI搜索失败: {e}")
        return []


async def _search_tavily(client: httpx.AsyncClient, query: str) -> dict | None:
    """Tavily 深度搜索：美食攻略查询，失败返回 None"""
    try:
        resp = await client.post(TAVILY_URL, json={
            "api_key": settings.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": 5,
            "include_answer": True,
        })
        if resp.status_code != 200:
            app_logger.warning(f"Tavily搜索 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        if data.get("results") is not None:
            return {
                "answer": data.get("answer", ""),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": (r.get("content") or "")[:300],
                    }
                    for r in data["results"]
                ],
            }
        return None
    except (httpx.HTTPError, ValueError, LookupError) as e:
        app_logger.warning(f"Tavily搜索失败: {e}")
        return None


def _format_poi_results(pois: list[dict]) -> str:
    """POI 列表 → Markdown 表格"""
    if not pois:
        return ""

    lines = ["### 🗺️ 周边餐厅", ""]
    lines.append("| 名称 | 地址 | 类型 | 电话 |")
    lines.append("|------|------|------|------|")
    for p in pois:
        name = p.get("name", "")
        addr = p.get("address", "")
        raw_type = p.get("type", "")
        ptype = raw_type.split(";")[-1] if raw_type else ""
        tel = p.get("tel", "")
        lines.append(f"| {name} | {addr} | {ptype} | {tel} |")
    return "\n".join(lines)


def _format_tavily_result(data: dict | None) -> str:
    """Tavily 搜索结果 → Markdown"""
    if not data:
        return ""

    lines = ["### 📝 美食攻略", ""]
    if data.get("answer"):
        lines.append(data["answer"])
        lines.append("")

    links = [r for r in data.get("results", []) if r.get("title") and r.get("url")]
    if links:
        lines.append("**参考链接**:")
        for r in links:
            lines.append(f"- [{r['title']}]({r['url']})")
    return "\n".join(lines)


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
        elif 'search_travel_info' in name:
            search_tool = t

    # 检查工具可用性
    if not any([geo_tool, around_tool, search_tool]):
        return "⚠️ 餐饮查询服务暂不可用，请稍后重试。"

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
