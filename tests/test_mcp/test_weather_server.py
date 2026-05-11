"""
天气 MCP Server 测试
"""
import httpx
import pytest
from unittest import mock

from app.mcp_core.servers.weather_server import (
    get_weather_forecast,
    _transform_forecast,
    _fetch_amap_forecast,
)


class TestTransformForecast:
    """字段映射单元测试"""

    def test_maps_amap_fields_to_snake_case(self):
        raw = {
            "date": "2026-05-11",
            "week": "星期一",
            "dayweather": "晴",
            "nightweather": "多云",
            "daytemp": "28",
            "nighttemp": "18",
            "daywind": "东风",
            "nightwind": "东北风",
            "daypower": "3",
            "nightpower": "2",
        }
        result = _transform_forecast(raw)
        assert result == {
            "date": "2026-05-11",
            "week": "星期一",
            "day_weather": "晴",
            "night_weather": "多云",
            "day_temp": "28",
            "night_temp": "18",
            "day_wind": "东风",
            "night_wind": "东北风",
            "day_power": "3",
            "night_power": "2",
        }

    def test_ignores_unknown_fields(self):
        raw = {"date": "2026-05-11", "week": "星期一", "extra_field": "ignored"}
        result = _transform_forecast(raw)
        assert "extra_field" not in result
        assert result == {"date": "2026-05-11", "week": "星期一"}


class TestGetWeatherForecastErrors:
    """错误处理单元测试 — 不发起真实 HTTP 请求"""

    @pytest.mark.asyncio
    async def test_empty_adcode_returns_error(self):
        result = await get_weather_forecast("")
        assert result == {"error": "city_adcode 不能为空"}

    @pytest.mark.asyncio
    async def test_whitespace_adcode_returns_error(self):
        result = await get_weather_forecast("   ")
        assert result == {"error": "city_adcode 不能为空"}

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self):
        with mock.patch(
            "app.mcp_core.servers.weather_server.settings"
        ) as mock_settings:
            mock_settings.amap_api_key = ""
            result = await get_weather_forecast("110000")
            assert result == {"error": "天气服务未配置，请设置 AMAP_API_KEY"}

    @pytest.mark.asyncio
    async def test_http_error_returns_error(self):
        with mock.patch(
            "app.mcp_core.servers.weather_server._fetch_amap_forecast"
        ) as mock_fetch:
            mock_fetch.side_effect = httpx.ConnectError("连接失败")
            result = await get_weather_forecast("110000")
            assert result["error"].startswith("天气服务请求失败:")

    @pytest.mark.asyncio
    async def test_amap_status_zero_returns_error(self):
        with mock.patch(
            "app.mcp_core.servers.weather_server._fetch_amap_forecast"
        ) as mock_fetch:
            mock_fetch.return_value = {
                "status": "0",
                "info": "INVALID_USER_KEY",
            }
            result = await get_weather_forecast("110000")
            assert result == {"error": "天气查询失败: INVALID_USER_KEY"}

    @pytest.mark.asyncio
    async def test_empty_forecasts_returns_error(self):
        with mock.patch(
            "app.mcp_core.servers.weather_server._fetch_amap_forecast"
        ) as mock_fetch:
            mock_fetch.return_value = {"status": "1", "forecasts": []}
            result = await get_weather_forecast("000000")
            assert result == {"error": "未查询到该城市的天气预报"}


class TestGetWeatherForecastSuccess:
    """成功路径单元测试 — mock _fetch_amap_forecast"""

    @pytest.mark.asyncio
    async def test_returns_structured_forecast(self):
        mock_response = {
            "status": "1",
            "forecasts": [
                {
                    "city": "北京市",
                    "adcode": "110000",
                    "reporttime": "2026-05-11 10:00:00",
                    "casts": [
                        {
                            "date": "2026-05-11",
                            "week": "星期一",
                            "dayweather": "晴",
                            "nightweather": "多云",
                            "daytemp": "28",
                            "nighttemp": "18",
                            "daywind": "东风",
                            "nightwind": "东北风",
                            "daypower": "3",
                            "nightpower": "2",
                        },
                    ],
                }
            ],
        }
        with mock.patch(
            "app.mcp_core.servers.weather_server._fetch_amap_forecast"
        ) as mock_fetch:
            mock_fetch.return_value = mock_response
            result = await get_weather_forecast("110000")

        assert result["city"] == "北京市"
        assert result["adcode"] == "110000"
        assert result["report_time"] == "2026-05-11 10:00:00"
        assert len(result["forecasts"]) == 1
        f = result["forecasts"][0]
        assert f["day_weather"] == "晴"
        assert f["night_weather"] == "多云"
        assert f["day_temp"] == "28"


class TestIntegration:
    """集成测试 — 需要真实 AMAP_API_KEY"""

    @pytest.mark.asyncio
    async def test_forecast_valid_adcode(self):
        """北京 adcode 应返回 7 天预报"""
        result = await get_weather_forecast("110000")
        assert "error" not in result
        assert result["city"] == "北京市"
        assert len(result["forecasts"]) >= 4
        f = result["forecasts"][0]
        assert "day_weather" in f
        assert "night_temp" in f

    @pytest.mark.asyncio
    async def test_forecast_invalid_adcode_returns_error(self):
        """无效 adcode"""
        result = await get_weather_forecast("999999")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """并发查询多个城市，验证无竞态"""
        import asyncio
        results = await asyncio.gather(
            get_weather_forecast("110000"),  # 北京
            get_weather_forecast("310000"),  # 上海
            get_weather_forecast("440100"),  # 广州
        )
        for r in results:
            assert "error" not in r, f"Unexpected error: {r}"
            assert len(r["forecasts"]) >= 4
