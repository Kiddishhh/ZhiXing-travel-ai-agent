"""
餐饮查询工具
直接调用 Amap API + Tavily API 获取餐饮推荐
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
                    "tag":biz.get("tag", ""),
                    "rating": biz.get("rating", ""),
                    "cost":biz.get("cost", ""),
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
    lines.append("| 名称 | 地址 | 类型 | 电话 | 特色 | 评分 | 人均 |")
    lines.append("|------|------|------|------|------|------|------|")
    for p in pois:
        name = p.get("name", "")
        addr = p.get("address", "")
        raw_type = p.get("type", "")
        ptype = raw_type.split(";")[-1] if raw_type else ""
        tel = p.get("tel", "")
        tag = p.get("tag", "")
        rating = p.get("rating", "")
        cost = p.get("cost", "")
        lines.append(f"| {name} | {addr} | {ptype} | {tel} | {tag} | {rating} | {cost} |")
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


# ── 公开工具 ──

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

    if food_type == "restaurant":
        around_keyword = query or f"{destination} 餐厅"
        search_query = f"{destination} 必吃餐厅推荐 特色菜"
    elif food_type == "local_snack":
        around_keyword = query or f"{destination} 小吃"
        search_query = f"{destination} 本地小吃 特色美食攻略"
    else:
        around_keyword = query or f"{destination} 美食"
        search_query = f"{destination} 美食攻略 必吃推荐"

    results = []
    warnings = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # 1. 地理编码
        location = None
        if settings.amap_api_key:
            location = await _geocode(client, destination)

        if not location:
            warnings.append("⚠️ 无法获取目的地坐标")
        else:
            # 2. POI 周边搜索
            pois = await _search_poi(client, location, around_keyword)
            if pois:
                results.append(_format_poi_results(pois))
            else:
                warnings.append("⚠️ 地图餐饮数据暂不可用")

        # 3. Tavily 美食攻略
        if settings.tavily_api_key:
            tavily_data = await _search_tavily(client, search_query)
            if tavily_data:
                results.append(_format_tavily_result(tavily_data))
            else:
                warnings.append("⚠️ 美食攻略数据暂不可用")

    if not results:
        msg = "⚠️ 餐饮查询服务暂不可用，请稍后重试。"
        if warnings:
            msg = "\n".join(warnings) + "\n\n" + msg
        return msg

    output = f"## 🍜 {destination} 餐饮推荐\n"
    if warnings:
        output += "\n".join(warnings) + "\n\n"
    output += "\n\n".join(results)
    return output
