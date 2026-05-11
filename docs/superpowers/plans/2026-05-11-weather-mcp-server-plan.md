# 天气服务 MCP Server 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现基于 fastmcp + 高德天气 API 的 MCP server，暴露 `get_weather_forecast(city_adcode)` 异步工具返回 7 天预报。

**Architecture:** 单文件 MCP server（`weather_server.py`），fastmcp `@mcp.tool()` 注册异步工具 → `httpx.AsyncClient` 请求高德 API → 字段映射转换 → 返回结构化 dict。错误统一以 `{"error": "..."}` 返回。

**Tech Stack:** fastmcp 2.13.1, httpx 0.28.1, pytest-asyncio 1.3.0, 高德天气 API v3

**Spec:** `docs/superpowers/specs/2026-05-11-weather-mcp-server-design.md`

---

### 文件结构

```
app/mcp_core/
├── __init__.py                          # 新建：mcp_core 包
└── servers/
    ├── __init__.py                      # 新建：servers 包
    └── weather_server.py                # 新建：天气 MCP server

tests/test_mcp/
├── __init__.py                          # 新建：test_mcp 包
└── test_weather_server.py              # 新建：天气 server 测试
```

---

### Task 1: 创建目录结构和包文件

**Files:**
- Create: `app/mcp_core/__init__.py`
- Create: `app/mcp_core/servers/__init__.py`
- Create: `tests/test_mcp/__init__.py`

- [ ] **Step 1: 创建包目录和空 `__init__.py`**

```bash
mkdir -p "app/mcp_core/servers" "tests/test_mcp"
touch "app/mcp_core/__init__.py"
touch "app/mcp_core/servers/__init__.py"
touch "tests/test_mcp/__init__.py"
```

- [ ] **Step 2: 提交**

```bash
git add app/mcp_core/ tests/test_mcp/
git commit -m "chore: add mcp_core and test_mcp package scaffolding"
```

---

### Task 2: 编写单元测试（TDD — 先写失败的测试）

**Files:**
- Create: `tests/test_mcp/test_weather_server.py`

- [ ] **Step 1: 编写完整测试文件**

```python
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
        assert len(result["forecasts"]) >= 4  # 至少 4 天预报
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
```

- [ ] **Step 2: 运行测试确认全部失败（函数尚未实现）**

```bash
python -m pytest tests/test_mcp/test_weather_server.py -v
```

预期：全部 FAIL（`ModuleNotFoundError` 或 `ImportError`）

- [ ] **Step 3: 提交**

```bash
git add tests/test_mcp/test_weather_server.py
git commit -m "test: add weather MCP server unit and integration tests"
```

---

### Task 3: 实现 `_fetch_amap_forecast` 和 `_transform_forecast`

**Files:**
- Create: `app/mcp_core/servers/weather_server.py`

- [ ] **Step 1: 编写 server 文件（先写辅助函数部分）**

```python
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
```

- [ ] **Step 2: 运行部分单元测试验证辅助函数可用**

```bash
python -m pytest tests/test_mcp/test_weather_server.py::TestTransformForecast -v
```

预期：2 个 `_transform_forecast` 测试 PASS

- [ ] **Step 3: 提交**

```bash
git add app/mcp_core/servers/weather_server.py
git commit -m "feat: add _fetch_amap_forecast and _transform_forecast helpers"
```

---

### Task 4: 实现 `get_weather_forecast` 工具函数

**Files:**
- Modify: `app/mcp_core/servers/weather_server.py`（追加内容）

- [ ] **Step 1: 在 weather_server.py 末尾追加工具函数和入口**

```python
# ── MCP Server ──────────────────────────────────────────

from fastmcp import FastMCP

mcp = FastMCP("amap-weather")


@mcp.tool()
async def get_weather_forecast(city_adcode: str) -> dict:
    """根据城市 adcode 查询未来 7 天天气预报

    Args:
        city_adcode: 高德行政区划编码（如 110000=北京, 310000=上海）

    Returns:
        {
            "city": "北京市",
            "adcode": "110000",
            "report_time": "2026-05-11 10:00:00",
            "forecasts": [
                {
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
                },
                ...
            ]
        }
    """
    if not city_adcode or not city_adcode.strip():
        return {"error": "city_adcode 不能为空"}

    try:
        data = await _fetch_amap_forecast(city_adcode.strip())
    except httpx.HTTPError as e:
        return {"error": f"天气服务请求失败: {e}"}

    # 处理 _fetch_amap_forecast 返回的错误 dict
    if "error" in data:
        return data

    if data.get("status") != "1":
        return {"error": f"天气查询失败: {data.get('info', '未知错误')}"}

    forecasts = data.get("forecasts", [])
    if not forecasts:
        return {"error": "未查询到该城市的天气预报"}

    city_info = forecasts[0]
    return {
        "city": city_info.get("city"),
        "adcode": city_info.get("adcode"),
        "report_time": city_info.get("reporttime"),
        "forecasts": [_transform_forecast(c) for c in city_info.get("casts", [])],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- [ ] **Step 2: 运行单元测试（错误处理 + 成功路径）**

```bash
python -m pytest tests/test_mcp/test_weather_server.py -v -k "not Integration"
```

预期：全部单元测试 PASS（6 个测试）

- [ ] **Step 3: 运行集成测试（需要 API key）**

```bash
python -m pytest tests/test_mcp/test_weather_server.py::TestIntegration -v
```

预期：3 个集成测试 PASS

- [ ] **Step 4: 运行全部测试确认无回归**

```bash
python -m pytest tests/test_mcp/ -v
```

预期：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add app/mcp_core/servers/weather_server.py
git commit -m "feat: add get_weather_forecast MCP tool with async Amap API integration"
```

---

### Task 5: 语法和导入完整性验证

- [ ] **Step 1: 语法检查**

```bash
python -c "import ast; ast.parse(open('app/mcp_core/servers/weather_server.py', encoding='utf-8').read()); print('Syntax OK')"
```

- [ ] **Step 2: 导入验证**

```bash
python -c "from app.mcp_core.servers.weather_server import mcp, get_weather_forecast, _transform_forecast, _fetch_amap_forecast; print('Import OK')"
```

- [ ] **Step 3: 运行全部已有测试确认无回归**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 4: 提交**

```bash
git commit -m "chore: verify weather MCP server syntax and imports" --allow-empty
```
（如果上一步测试全部通过且无变更，跳过此提交）

---

### 完整文件内容参考

**`app/mcp_core/servers/weather_server.py`** 最终完整内容：

```python
"""
高德天气 MCP Server

暴露 get_weather_forecast(city_adcode) 工具，
根据城市 adcode 查询未来 7 天天气预报。
"""
import httpx
from fastmcp import FastMCP

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


# ── MCP Server ──────────────────────────────────────────

mcp = FastMCP("amap-weather")


@mcp.tool()
async def get_weather_forecast(city_adcode: str) -> dict:
    """根据城市 adcode 查询未来 7 天天气预报

    Args:
        city_adcode: 高德行政区划编码（如 110000=北京, 310000=上海）

    Returns:
        {
            "city": "北京市",
            "adcode": "110000",
            "report_time": "2026-05-11 10:00:00",
            "forecasts": [
                {
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
                },
                ...
            ]
        }
    """
    if not city_adcode or not city_adcode.strip():
        return {"error": "city_adcode 不能为空"}

    try:
        data = await _fetch_amap_forecast(city_adcode.strip())
    except httpx.HTTPError as e:
        return {"error": f"天气服务请求失败: {e}"}

    if "error" in data:
        return data

    if data.get("status") != "1":
        return {"error": f"天气查询失败: {data.get('info', '未知错误')}"}

    forecasts = data.get("forecasts", [])
    if not forecasts:
        return {"error": "未查询到该城市的天气预报"}

    city_info = forecasts[0]
    return {
        "city": city_info.get("city"),
        "adcode": city_info.get("adcode"),
        "report_time": city_info.get("reporttime"),
        "forecasts": [_transform_forecast(c) for c in city_info.get("casts", [])],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
```
