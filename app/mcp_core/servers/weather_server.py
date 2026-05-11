"""
高德天气 MCP Server

暴露 get_weather_forecast(city_adcode) 工具，
根据城市 adcode 查询未来 7 天天气预报。
"""
import httpx
from app.config import settings

AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"

_FORECAST_FIELD_MAP = {
    "date": "date",
    "week": "week",
    "dayweather": "day_weather",
    "nightweather": "night_weather",
    "daytemp": "day_temp",
    "nighttemp": "night_temp",
    "daywind": "day_wind",
    "nightwind": "night_wind",
    "daypower": "day_power",
    "nightpower": "night_power",
}


def _transform_forecast(cast: dict) -> dict:
    """将高德字段名转换为下划线风格"""
    return {_FORECAST_FIELD_MAP[k]: v for k, v in cast.items() if k in _FORECAST_FIELD_MAP}


async def _fetch_amap_forecast(adcode: str) -> dict:
    """调用高德天气 API，返回原始响应 dict"""
    api_key = settings.amap_api_key
    if not api_key:
        return {"error": "天气服务未配置，请设置 AMAP_API_KEY"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(AMAP_WEATHER_URL, params={
            "key": api_key,
            "city": adcode,
            "extensions": "all",
        })
        resp.raise_for_status()
        return resp.json()


async def get_weather_forecast(city_adcode: str) -> dict:
    """查询城市未来7天天气预报（Task 4 实现）"""
    raise NotImplementedError("get_weather_forecast 将在 Task 4 实现")
