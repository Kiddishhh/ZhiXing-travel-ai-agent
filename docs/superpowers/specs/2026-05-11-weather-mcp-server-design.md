# 天气服务 MCP Server 设计

## 背景

当前 `_weather_agent()`（`app/agents/routers/destination_router.py:151`）为占位实现，需要接入真实天气数据。高德天气 API 已申请 `AMAP_API_KEY`（`app/config.py:59`），`fastmcp==2.13.1` 和 `httpx==0.28.1` 已在依赖中。

## 目标

实现一个 MCP server，暴露 `get_weather_forecast(city_adcode: str)` 异步工具，根据城市 adcode 返回未来 7 天天气预报（含白天/夜间天气、温度、风力）。

## 文件位置

```
app/mcp_core/servers/weather_server.py    # MCP server 实现
tests/test_mcp/                            # 对应测试（新建目录）
```

## 架构

```
┌──────────────────────────────────────┐
│     FastMCP("amap-weather")          │
│                                      │
│  @mcp.tool()                         │
│  async def get_weather_forecast(     │
│      city_adcode: str                │
│  ) → dict                            │
│                                      │
│  内部:                                │
│  ├─ 校验 adcode 格式                  │
│  ├─ 构造高德 API 请求                  │
│  ├─ httpx.AsyncClient GET            │
│  ├─ 解析响应 + 错误分类                │
│  └─ 返回 {"forecasts": [...], ...}   │
└──────────────────────────────────────┘
         │
         ▼  httpx.AsyncClient
  高德天气 API
  GET /v3/weather/weatherInfo
  ?key={AMAP_API_KEY}&city={adcode}&extensions=all
```

## 组件

### 1. MCP Server 入口

```python
from fastmcp import FastMCP

mcp = FastMCP("amap-weather")
```

- 通过 `if __name__ == "__main__": mcp.run(transport="stdio")` 启动
- `transport="stdio"` 适配 MCP 客户端通过子进程调用

### 2. Tool: `get_weather_forecast`（异步）

```python
@mcp.tool()
async def get_weather_forecast(city_adcode: str) -> dict:
    """根据城市 adcode 查询未来 7 天天气预报"""
```

- `async def` 使 fastmcp 在 event loop 中调度，不阻塞其他并发请求
- 入参：`city_adcode` 为高德行政区划编码（如 `110000` 北京）
- 返回：`dict`，fastmcp 自动序列化为 JSON

### 3. 高德 API 调用函数（异步）

```python
async def _fetch_amap_forecast(adcode: str) -> dict:
    """调用高德天气 API，返回原始响应 dict"""
```

- 从 `app.config.settings` 获取 `amap_api_key`
- 使用 `httpx.AsyncClient` 异步请求，支持连接复用
- 设置 10s 超时

## 数据流

```
调用方 (MCP client)
  → await get_weather_forecast("110000")
    → 校验 adcode 非空
    → await _fetch_amap_forecast("110000")
      → async with httpx.AsyncClient() as client:
          await client.get(高德 API)
      → 检查 HTTP 状态码
      → 检查高德返回 status=="1"
      → 提取 forecasts[] 数组
    → 转换字段名为下划线风格
    → 返回 {"forecasts": [...], "city": "...", "report_time": "..."}
```

## 高德 API 响应映射

高德返回字段 → 输出字段（下划线命名）：

| 高德字段 | 输出字段 | 说明 |
|---------|---------|------|
| `date` | `date` | 日期 |
| `week` | `week` | 星期几 |
| `dayweather` | `day_weather` | 白天天气 |
| `nightweather` | `night_weather` | 夜间天气 |
| `daytemp` | `day_temp` | 白天温度(℃) |
| `nighttemp` | `night_temp` | 夜间温度(℃) |
| `daywind` | `day_wind` | 白天风力风向 |
| `nightwind` | `night_wind` | 夜间风力风向 |
| `daypower` | `day_power` | 白天风力等级 |
| `nightpower` | `night_power` | 夜间风力等级 |

## 错误处理

| 场景 | 返回 |
|------|------|
| `city_adcode` 为空字符串 | `{"error": "city_adcode 不能为空"}` |
| `amap_api_key` 未配置 | `{"error": "天气服务未配置，请设置 AMAP_API_KEY"}` |
| HTTP 请求失败（网络/超时） | `{"error": "天气服务请求失败: {详情}"}` |
| 高德返回 `status=="0"` | `{"error": "天气查询失败: {info}"}` |
| 高德返回空 `forecasts` | `{"error": "未查询到该城市的天气预报"}` |

所有错误都通过 `{"error": "..."}` 统一返回，不抛异常，保证 MCP 调用方拿到结构化 JSON。

## 依赖

- `fastmcp` — MCP server 框架（已有）
- `httpx` — HTTP 异步客户端（已有，`httpx.AsyncClient`）
- `app.config.settings` — 获取 `amap_api_key`（已有）

无新增依赖。

## 测试策略

测试文件：`tests/test_mcp/test_weather_server.py`

| 测试用例 | 类型 | 说明 |
|---------|------|------|
| `test_forecast_valid_adcode` | 集成 | 用真实 adcode 异步调用，验证返回 forecasts 数组 |
| `test_forecast_empty_adcode` | 单元 | 空字符串入参，验证返回 error |
| `test_forecast_invalid_adcode` | 集成 | 无效 adcode，验证高德返回错误信息 |
| `test_forecast_key_missing` | 单元 | 模拟 API key 为空，验证错误返回 |
| `test_concurrent_requests` | 集成 | 多个 adcode 并发查询，验证无竞态、连接复用 |

测试使用 `pytest-asyncio` 标记 `@pytest.mark.asyncio`；单元测试通过 `unittest.mock.AsyncMock` 隔离外部依赖。

## 与现有系统集成

实现后，`_weather_agent()` 从占位实现改为异步调用 `get_weather_forecast`：

```python
# destination_router.py 后续改动
async def _weather_agent(query: str) -> str:
    # 1. 从 query 中提取目的地
    # 2. 查找对应 adcode
    # 3. await mcp_client.call_tool("get_weather_forecast", {"city_adcode": adcode})
    # 4. 格式化天气信息为 Markdown
```

这部分改动不在本次范围内，weather MCP server 本身是独立可用的服务。
