# food_tools.py 直连 API 优化设计

**日期**: 2026-05-13  
**状态**: 设计完成  
**影响范围**: `app/tools/food_tools.py`（单文件）

---

## 1. 背景

当前 `food_tools.py` 的 `query_food` 通过 MCP 客户端间接调用 Amap 和 Tavily API。每次调用需启动全部 6 个 MCP 服务、加载 37 个工具，再通过字符串匹配筛选 3 个目标工具。`query_food` 是纯工具函数而非 agent，不需要 MCP 的动态工具选择能力，这一层中间层带来三个问题：

- **启动开销大**：查餐饮却要拉起 weather/12306/VariFlight/Aigohotel 全部连接
- **参数受限**：`maps_around_search` 不支持 `types`（POI 分类码）、`show_fields` 等高德 v5 API 参数
- **代码脆弱**：substring 匹配工具名，MCP 服务变更即失效

## 2. 设计方案

### 2.1 架构变更

移除 `food_tools.py` 对 `MCPClientManager` 的依赖，改为 `httpx` 直接调三个 HTTP 端点：

```
优化前: get_mcp_client() → 6 MCP 服务器 → 37 工具 → substring 匹配 3 个 → 调用
优化后: httpx.AsyncClient → 3 个 HTTP 直连请求
```

依赖变化：

```diff
- from app.mcp_core.client import get_mcp_client
+ import httpx
+ from app.config import settings
```

### 2.2 数据流

```
query_food(destination, food_type, query)
  │
  ├─ Step 1  地理编码  →  destination → 坐标 (lng,lat)
  ├─ Step 2  POI搜索   →  坐标 + types=050000 → POI列表
  ├─ Step 3  Tavily    →  美食攻略关键词 → answer + results
  └─ Step 4  整合输出  →  格式化 Markdown 返回
```

### 2.3 Step 1: 地理编码

```
GET https://restapi.amap.com/v3/geocode/geo
  ?address={destination}
  &key={AMAP_API_KEY}

响应解析: geocodes[0].location → "116.481499,39.990475"
失败: 终止流程，返回明确错误提示
```

### 2.4 Step 2: POI 周边搜索

```
GET https://restapi.amap.com/v5/place/around
  ?location={lng,lat}
  &keywords={around_keyword}
  &types=050000
  &show_fields=business,photos
  &radius=2000
  &page_size=10
  &key={AMAP_API_KEY}
```

**关键词推导**：

| food_type | keywords | types |
|-----------|----------|-------|
| `"restaurant"` | `query` 或 `"{destination} 餐厅"` | 050000 |
| `"local_snack"` | `query` 或 `"{destination} 小吃"` | 050000 |
| `None`（全部） | `query` 或 `"{destination} 美食"` | 050000 |

`types=050000` 是餐饮服务大类，覆盖中餐/外国餐/快餐/小吃/咖啡厅等全部子类型。`show_fields=business,photos` 额外获取营业时间、电话、照片 URL。

**POI 字段提取**（从返回的 `pois[N]` 中取）：

| 源字段 | 输出 |
|--------|------|
| `name` | 餐厅名称 |
| `address` | 详细地址 |
| `type` | POI 类型（如"中餐厅"） |
| `business.tel` | 电话 |
| `business.opentime` | 营业时间 |
| `photos[0].url` | 首张照片 |
| `location` | 坐标 |

### 2.5 Step 3: Tavily 美食攻略

```
POST https://api.tavily.com/search
Body: {
  "api_key": "{TAVILY_API_KEY}",
  "query": "{search_query}",
  "search_depth": "advanced",
  "max_results": 5,
  "include_answer": true
}
```

**搜索词推导**：

| food_type | search_query |
|-----------|-------------|
| `"restaurant"` | `"{destination} 必吃餐厅推荐 特色菜"` |
| `"local_snack"` | `"{destination} 本地小吃 特色美食攻略"` |
| `None` | `"{destination} 美食攻略 必吃推荐"` |

注意：`search_depth` 从 `"basic"` 提升为 `"advanced"`，获取更深入的攻略摘要。

### 2.6 错误处理（独立容错）

| 场景 | 处理 |
|------|------|
| 地理编码失败 | **终止**，返回 `"未找到 {destination} 的坐标，请确认城市名称"` |
| POI 搜索失败 / Key 缺失 | **降级**，只输出 Tavily 攻略，标注 `⚠️ 地图餐饮数据暂不可用` |
| Tavily 失败 / Key 缺失 | **降级**，只输出 Amap POI 结果，标注 `⚠️ 美食攻略数据暂不可用` |
| 全部失败 | 返回 `"⚠️ 餐饮查询服务暂不可用，请稍后重试"` |
| HTTP 超时 | 单源 15s 超时，走降级逻辑 |

### 2.7 输出格式

```markdown
## 🍜 {destination} 餐饮推荐

### 🗺️ 周边餐厅
| 名称 | 地址 | 类型 | 电话 |
|------|------|------|------|
| XX餐厅 | XX路1号 | 中餐厅 | 010-xxx |

### 📝 美食攻略
{来自 Tavily 的 answer 摘要 + 精选链接}

### 💡 推荐建议
{结合 POI 搜索结果和攻略内容的 2-3 句总结，由 LLM 在 step_config prompt 中自然生成}
```

注意：`query_food` 的返回值是原始结构化数据，最终的用户面向文本由 step 5 的 LLM 在拿到 ToolMessage 后生成。"推荐建议"部分建议在 step_config 的 prompt 中提示 LLM 结合结果做总结。

### 2.8 代码结构（food_tools.py 内部）

```python
# ── 常量 ──
AMAP_GEO_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_AROUND_URL = "https://restapi.amap.com/v5/place/around"
TAVILY_URL = "https://api.tavily.com/search"
POI_TYPE_FOOD = "050000"
SEARCH_RADIUS = "2000"
REQUEST_TIMEOUT = 15.0

# ── 内部辅助函数 ──
async def _geocode(client, address: str) -> str | None
async def _search_poi(client, location, keyword) -> list[dict]
async def _search_tavily(client, query) -> dict | None
def _format_poi_results(pois: list[dict]) -> str
def _format_tavily_result(data: dict) -> str

# ── 公开工具 ──
@tool
async def query_food(destination, food_type=None, query=None) -> str
```

所有 `_` 前缀的辅助函数在 `query_food` 内部复用同一个 `httpx.AsyncClient` 实例。

### 2.9 关键参数默认值

| 参数 | 值 | 理由 |
|------|-----|------|
| `types` | `050000` | 餐饮服务全类 |
| `radius` | `2000` | 2km 半径，覆盖城市核心餐饮区 |
| `page_size` | `10` | 返回 10 条，兼顾信息量和响应体积 |
| `search_depth` | `advanced` | 攻略检索需要更深度摘要 |
| `timeout` | `15.0` | 单请求上限，避免挂起 |

## 3. 不做的事

- 不拆分新文件或新模块 — 改动仅限于 `food_tools.py`
- 不修改 `search_server.py`、Amap MCP、`MCPClientManager` — 它们继续为其他模块服务
- 不在 `food_tools.py` 中添加 agent 或 LLM 调用 — 保持纯工具函数定位
- 不引入新的第三方依赖 — `httpx` 已是项目现有依赖

## 4. 性能预期

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 连接初始化 | 6 个 MCP 服务 | 1 个 httpx client |
| 工具预加载 | 37 个 | 0 |
| 实际请求耗时 | ~3-5s（含 MCP 启动开销） | ~1-2s（纯 HTTP RTT） |

## 5. 测试要点

- 地理编码正常返回坐标
- 地理编码城市不存在返回错误提示
- POI 搜索带 `types=050000` 过滤
- Tavily 部分失败时降级输出 POI 结果
- Amap Key 缺失时跳过 POI 部分
- 全部源失败时返回统一提示
- 输出为合法 Markdown 格式
- `httpx.TimeoutException` 不中断其他源
