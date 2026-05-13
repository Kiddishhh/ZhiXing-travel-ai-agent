"""food_tools.py 单元测试 — HTTP mock，无需真实 API Key"""
import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx


class TestGeocode:
    """_geocode: 地址 → 坐标"""

    @pytest.mark.asyncio
    async def test_returns_location_on_success(self):
        """正常返回坐标字符串"""
        from app.tools.food_tools import _geocode

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "1",
            "geocodes": [{"location": "116.481499,39.990475"}],
        }
        mock_client.get.return_value = mock_resp

        result = await _geocode(mock_client, "北京")
        assert result == "116.481499,39.990475"

    @pytest.mark.asyncio
    async def test_returns_none_when_city_not_found(self):
        """无效城市返回 None"""
        from app.tools.food_tools import _geocode

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "0", "geocodes": []}
        mock_client.get.return_value = mock_resp

        result = await _geocode(mock_client, "不存在的城市xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self):
        """网络超时返回 None"""
        from app.tools.food_tools import _geocode

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("timeout")

        result = await _geocode(mock_client, "北京")
        assert result is None
