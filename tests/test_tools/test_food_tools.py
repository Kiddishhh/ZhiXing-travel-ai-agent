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
        mock_resp.status_code = 200
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
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "0", "geocodes": []}
        mock_client.get.return_value = mock_resp

        result = await _geocode(mock_client, "不存在的城市xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_non_200_status(self):
        """HTTP 非 200 状态返回 None"""
        from app.tools.food_tools import _geocode

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.text = "Bad Gateway"
        mock_client.get.return_value = mock_resp

        result = await _geocode(mock_client, "北京")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self):
        """网络超时返回 None"""
        from app.tools.food_tools import _geocode

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("timeout")

        result = await _geocode(mock_client, "北京")
        assert result is None


class TestSearchPoi:
    """_search_poi: 周边 POI 搜索（v5/place/around）"""

    @pytest.mark.asyncio
    async def test_returns_poi_list_on_success(self):
        """正常返回结构化 POI 列表"""
        from app.tools.food_tools import _search_poi

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "1",
            "pois": [
                {
                    "name": "海底捞火锅",
                    "address": "长安街100号",
                    "type": "餐饮服务;中餐厅;火锅",
                    "business": {"tel": "010-12345678", "opentime": "10:00-22:00"},
                    "photos": [{"url": "https://example.com/photo1.jpg"}],
                    "location": "116.48,39.99",
                },
                {
                    "name": "兰州拉面",
                    "address": "王府井大街20号",
                    "type": "餐饮服务;中餐厅;面馆",
                    "business": None,
                    "photos": [],
                    "location": "116.49,39.98",
                },
            ],
        }
        mock_client.get.return_value = mock_resp

        result = await _search_poi(mock_client, "116.48,39.99", "北京 餐厅")
        assert len(result) == 2
        assert result[0]["name"] == "海底捞火锅"
        assert result[0]["tel"] == "010-12345678"
        assert result[0]["opentime"] == "10:00-22:00"
        assert result[0]["photos"] == ["https://example.com/photo1.jpg"]
        # 第二个 POI business 为空
        assert result[1]["tel"] == ""
        assert result[1]["opentime"] == ""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_pois(self):
        """无 POI 时返回空列表"""
        from app.tools.food_tools import _search_poi

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "1", "pois": []}
        mock_client.get.return_value = mock_resp

        result = await _search_poi(mock_client, "116.48,39.99", "北京 餐厅")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_http_error(self):
        """HTTP 错误返回空列表（降级）"""
        from app.tools.food_tools import _search_poi

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("server error")

        result = await _search_poi(mock_client, "116.48,39.99", "北京 餐厅")
        assert result == []


class TestSearchTavily:
    """_search_tavily: Tavily 美食攻略搜索"""

    @pytest.mark.asyncio
    async def test_returns_structured_data_on_success(self):
        """正常返回结构化搜索结果"""
        from app.tools.food_tools import _search_tavily

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "answer": "西安必吃推荐：肉夹馍、凉皮、羊肉泡馍...",
            "results": [
                {
                    "title": "西安美食攻略",
                    "url": "https://example.com/xian-food",
                    "content": "详细介绍西安回民街美食...",
                }
            ],
        }
        mock_client.post.return_value = mock_resp

        result = await _search_tavily(mock_client, "西安 美食攻略")
        assert result is not None
        assert "西安必吃推荐" in result["answer"]
        assert result["results"][0]["title"] == "西安美食攻略"
        assert len(result["results"][0]["content"]) <= 300

    @pytest.mark.asyncio
    async def test_returns_none_when_no_results(self):
        """无 results 字段时返回 None"""
        from app.tools.food_tools import _search_tavily

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_client.post.return_value = mock_resp

        result = await _search_tavily(mock_client, "xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self):
        """超时返回 None"""
        from app.tools.food_tools import _search_tavily

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timeout")

        result = await _search_tavily(mock_client, "西安 美食")
        assert result is None
