# food_tools.py 直连 API 优化 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `food_tools.py` 从 MCP 间接调用改为 `httpx` 直连 Amap v3/v5 + Tavily API，获得完整参数控制能力并消除 MCP 启动开销。

**Architecture:** 在 `food_tools.py` 内新增 5 个 `_`-前缀的内部辅助函数（`_geocode`、`_search_poi`、`_search_tavily`、`_format_poi_results`、`_format_tavily_result`），共用同一个 `httpx.AsyncClient` 实例。`query_food` 主函数重写为串联三步调用 + 独立容错 + Markdown 输出。不拆新文件，不修改其他模块。

**Tech Stack:** httpx（已有依赖）, pytest-asyncio, unittest.mock.AsyncMock

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `app/tools/food_tools.py` | 修改 | 移除 MCP 依赖，新增 5 个辅助函数 + 重写 `query_food` |
| `tests/test_tools/test_food_tools.py` | 新建 | 所有辅助函数 + `query_food` 的单元测试（HTTP mock） |

---

### Task 1: `_geocode` 辅助函数（TDD）

**Files:**
- Create: `tests/test_tools/test_food_tools.py`
- Modify: `app/tools/food_tools.py`

- [ ] **Step 1: 创建测试目录和测试文件**

```bash
mkdir tests\test_tools
```

- [ ] **Step 2: 编写 `_geocode` 的 3 个测试用例**

在 `tests/test_tools/test_food_tools.py` 写入：

```python
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
```

- [ ] **Step 3: 运行测试 — 预期 FAIL（`_geocode` 不存在）**

```bash
python -m pytest tests/test_tools/test_food_tools.py::TestGeocode -v
```

- [ ] **Step 4: 在 `food_tools.py` 顶部添加新导入和 `_geocode` 实现**

在 `app/tools/food_tools.py` 的 docstring 之后、**现有 `from app.mcp_core.client import get_mcp_client` 行之前**，插入新导入和常量：

```python
import httpx
from app.config import settings

# ── API 端点常量 ──
AMAP_GEO_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_AROUND_URL = "https://restapi.amap.com/v5/place/around"
TAVILY_URL = "https://api.tavily.com/search"
POI_TYPE_FOOD = "050000"
SEARCH_RADIUS = "2000"
REQUEST_TIMEOUT = 15.0
```

然后在文件末尾、**旧 `query_food` 函数之前**，插入 `_geocode` 函数：

```python
# ── 辅助函数 ──

async def _geocode(client: httpx.AsyncClient, address: str) -> str | None:
    """地理编码：结构化地址 → 经纬度坐标，失败返回 None"""
    try:
        resp = await client.get(AMAP_GEO_URL, params={
            "address": address,
            "key": settings.amap_api_key,
        })
        data = resp.json()
        if data.get("status") == "1" and data.get("geocodes"):
            return data["geocodes"][0]["location"]
        return None
    except Exception as e:
        app_logger.warning(f"地理编码失败: {e}")
        return None
```

**注意**：保留旧的 `from app.mcp_core.client import get_mcp_client` 和旧 `query_food` 函数体不动，后续 Task 5 移除。

- [ ] **Step 5: 运行 `_geocode` 测试 — 预期 PASS**

```bash
python -m pytest tests/test_tools/test_food_tools.py::TestGeocode -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_tools/test_food_tools.py app/tools/food_tools.py
git commit -m "feat: add _geocode helper with direct Amap v3 API call"
```

---

### Task 2: `_search_poi` 辅助函数（TDD）

**Files:**
- Modify: `tests/test_tools/test_food_tools.py`
- Modify: `app/tools/food_tools.py`

- [ ] **Step 1: 编写 `_search_poi` 的 3 个测试用例**

追加到 `tests/test_tools/test_food_tools.py`：

```python
class TestSearchPoi:
    """_search_poi: 周边 POI 搜索（v5/place/around）"""

    @pytest.mark.asyncio
    async def test_returns_poi_list_on_success(self):
        """正常返回结构化 POI 列表"""
        from app.tools.food_tools import _search_poi

        mock_client = AsyncMock()
        mock_resp = MagicMock()
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
        mock_resp.json.return_value = {"status": "1", "pois": []}
        mock_client.get.return_value = mock_resp

        result = await _search_poi(mock_client, "116.48,39.99", "北京 餐厅")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self):
        """网络错误返回空列表（降级）"""
        from app.tools.food_tools import _search_poi

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("server error")

        result = await _search_poi(mock_client, "116.48,39.99", "北京 餐厅")
        assert result == []
```

- [ ] **Step 2: 运行测试 — 预期 FAIL（`_search_poi` 不存在）**

```bash
python -m pytest tests/test_tools/test_food_tools.py::TestSearchPoi -v
```

- [ ] **Step 3: 在 `food_tools.py` 添加 `_search_poi` 实现**

在 `_geocode` 函数后面追加：

```python
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
    except Exception as e:
        app_logger.warning(f"POI搜索失败: {e}")
        return []
```

- [ ] **Step 4: 运行 `_search_poi` 测试 — 预期 PASS**

```bash
python -m pytest tests/test_tools/test_food_tools.py::TestSearchPoi -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_tools/test_food_tools.py app/tools/food_tools.py
git commit -m "feat: add _search_poi helper with Amap v5/place/around + types=050000"
```

---

### Task 3: `_search_tavily` 辅助函数（TDD）

**Files:**
- Modify: `tests/test_tools/test_food_tools.py`
- Modify: `app/tools/food_tools.py`

- [ ] **Step 1: 编写 `_search_tavily` 的 3 个测试用例**

追加到 `tests/test_tools/test_food_tools.py`：

```python
class TestSearchTavily:
    """_search_tavily: Tavily 美食攻略搜索"""

    @pytest.mark.asyncio
    async def test_returns_structured_data_on_success(self):
        """正常返回结构化搜索结果"""
        from app.tools.food_tools import _search_tavily

        mock_client = AsyncMock()
        mock_resp = MagicMock()
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
        """无结果时返回 None"""
        from app.tools.food_tools import _search_tavily

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}  # 缺少 results 字段
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
```

- [ ] **Step 2: 运行测试 — 预期 FAIL**

```bash
python -m pytest tests/test_tools/test_food_tools.py::TestSearchTavily -v
```

- [ ] **Step 3: 在 `food_tools.py` 添加 `_search_tavily`**

在 `_search_poi` 后面追加：

```python
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
        data = resp.json()
        if data.get("results") is not None:
            return {
                "answer": data.get("answer", ""),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": (r.get("content", "") or "")[:300],
                    }
                    for r in data["results"]
                ],
            }
        return None
    except Exception as e:
        app_logger.warning(f"Tavily搜索失败: {e}")
        return None
```

- [ ] **Step 4: 运行 `_search_tavily` 测试 — 预期 PASS**

```bash
python -m pytest tests/test_tools/test_food_tools.py::TestSearchTavily -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_tools/test_food_tools.py app/tools/food_tools.py
git commit -m "feat: add _search_tavily helper with advanced search depth"
```

---

### Task 4: 格式化辅助函数（TDD）

**Files:**
- Modify: `tests/test_tools/test_food_tools.py`
- Modify: `app/tools/food_tools.py`

- [ ] **Step 1: 编写 `_format_*` 的测试用例**

追加到 `tests/test_tools/test_food_tools.py`：

```python
class TestFormatPoiResults:
    """_format_poi_results: POI 列表 → Markdown 表格"""

    def test_formats_pois_as_markdown_table(self):
        from app.tools.food_tools import _format_poi_results

        pois = [
            {
                "name": "海底捞",
                "address": "长安街100号",
                "type": "餐饮服务;中餐厅;火锅",
                "tel": "010-12345678",
                "opentime": "10:00-22:00",
                "photos": [],
                "location": "116.48,39.99",
            },
        ]
        result = _format_poi_results(pois)
        assert "### 🗺️ 周边餐厅" in result
        assert "海底捞" in result
        assert "长安街100号" in result
        assert "火锅" in result         # 取最后一级分类
        assert "中餐厅" not in result   # 不出现全路径
        assert "010-12345678" in result

    def test_returns_empty_string_for_empty_list(self):
        from app.tools.food_tools import _format_poi_results

        assert _format_poi_results([]) == ""


class TestFormatTavilyResult:
    """_format_tavily_result: Tavily 响应 → Markdown"""

    def test_formats_with_answer_and_links(self):
        from app.tools.food_tools import _format_tavily_result

        data = {
            "answer": "西安必吃推荐：肉夹馍、凉皮...",
            "results": [
                {"title": "攻略一", "url": "https://a.com", "content": "xxx"},
                {"title": "攻略二", "url": "https://b.com", "content": "yyy"},
            ],
        }
        result = _format_tavily_result(data)
        assert "### 📝 美食攻略" in result
        assert "西安必吃推荐" in result
        assert "[攻略一](https://a.com)" in result
        assert "[攻略二](https://b.com)" in result

    def test_skips_result_without_url(self):
        from app.tools.food_tools import _format_tavily_result

        data = {
            "answer": "",
            "results": [{"title": "无链接", "url": "", "content": "xxx"}],
        }
        result = _format_tavily_result(data)
        assert "无链接" not in result  # 无 url 的跳过

    def test_formats_without_answer(self):
        from app.tools.food_tools import _format_tavily_result

        data = {
            "answer": "",
            "results": [{"title": "攻略", "url": "https://a.com", "content": "yyy"}],
        }
        result = _format_tavily_result(data)
        assert "[攻略](https://a.com)" in result
        # answer 为空时不展示
        assert "**参考链接**" in result

    def test_returns_empty_string_for_none(self):
        from app.tools.food_tools import _format_tavily_result

        assert _format_tavily_result(None) == ""
```

- [ ] **Step 2: 运行测试 — 预期 FAIL（格式化函数不存在）**

```bash
python -m pytest tests/test_tools/test_food_tools.py::TestFormatPoiResults tests/test_tools/test_food_tools.py::TestFormatTavilyResult -v
```

- [ ] **Step 3: 在 `food_tools.py` 添加格式化函数**

在 `_search_tavily` 后面、旧 `query_food` 前面追加：

```python
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
```

- [ ] **Step 4: 运行格式化测试 — 预期 PASS**

```bash
python -m pytest tests/test_tools/test_food_tools.py::TestFormatPoiResults tests/test_tools/test_food_tools.py::TestFormatTavilyResult -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_tools/test_food_tools.py app/tools/food_tools.py
git commit -m "feat: add _format_poi_results and _format_tavily_result helpers"
```

---

### Task 5: 重写 `query_food` 主函数

**Files:**
- Modify: `app/tools/food_tools.py`（替换旧 `query_food`）
- Modify: `tests/test_tools/test_food_tools.py`（集成测试）

- [ ] **Step 1: 编写 `query_food` 集成测试**

追加到 `tests/test_tools/test_food_tools.py`：

```python
class TestQueryFoodIntegration:
    """query_food 集成测试 — mock 全部 HTTP 响应"""

    @pytest.mark.asyncio
    async def test_all_sources_work(self):
        """三个数据源均正常 → 输出完整 Markdown"""
        from app.tools.food_tools import query_food

        # 构造连锁 mock：geocode → poi → tavily
        with patch("app.tools.food_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()

            async def get_side_effect(url, **kwargs):
                resp = MagicMock()
                if "geocode/geo" in url:
                    resp.json.return_value = {
                        "status": "1",
                        "geocodes": [{"location": "108.940,34.260"}],
                    }
                elif "place/around" in url:
                    resp.json.return_value = {
                        "status": "1",
                        "pois": [
                            {
                                "name": "回民街小吃",
                                "address": "西安市莲湖区回民街",
                                "type": "餐饮服务;地方小吃",
                                "business": {"tel": "", "opentime": "08:00-23:00"},
                                "photos": [],
                                "location": "108.94,34.26",
                            }
                        ],
                    }
                return resp

            async def post_side_effect(url, **kwargs):
                resp = MagicMock()
                resp.json.return_value = {
                    "answer": "西安美食推荐...",
                    "results": [
                        {"title": "攻略", "url": "https://x.com", "content": "..."}
                    ],
                }
                return resp

            mock_client.get = AsyncMock(side_effect=get_side_effect)
            mock_client.post = AsyncMock(side_effect=post_side_effect)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await query_food.ainvoke({
                "destination": "西安",
                "food_type": "restaurant",
            })

        assert "## 🍜 西安 餐饮推荐" in result
        assert "回民街小吃" in result
        assert "### 📝 美食攻略" in str(result)

    @pytest.mark.asyncio
    async def test_geocode_fails_returns_error(self):
        """地理编码失败 → 终止，返回错误"""
        from app.tools.food_tools import query_food

        with patch("app.tools.food_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"status": "0", "geocodes": []}
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await query_food.ainvoke({
                "destination": "不存在的城市xyz",
            })

        assert "无法获取目的地坐标" in str(result)
        # 没有任何 POI 或攻略输出
        assert "🗺️" not in str(result)

    @pytest.mark.asyncio
    async def test_tavily_fails_degraded_to_poi_only(self):
        """Tavily 失败 → 降级只输出 POI 结果"""
        from app.tools.food_tools import query_food

        with patch("app.tools.food_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()

            # geocode 成功
            async def get_side(url, **kw):
                resp = MagicMock()
                if "geocode" in url:
                    resp.json.return_value = {
                        "status": "1",
                        "geocodes": [{"location": "116.48,39.99"}],
                    }
                elif "around" in url:
                    resp.json.return_value = {
                        "status": "1",
                        "pois": [
                            {"name": "测试餐厅", "address": "某地", "type": "中餐厅",
                             "business": {}, "photos": [], "location": "1,1"}
                        ],
                    }
                return resp
            mock_client.get = AsyncMock(side_effect=get_side)
            # tavily 失败
            mock_client.post.side_effect = httpx.TimeoutException("timeout")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await query_food.ainvoke({"destination": "北京"})

        assert "测试餐厅" in str(result)
        assert "⚠️ 美食攻略数据暂不可用" in str(result)

    @pytest.mark.asyncio
    async def test_all_sources_fail(self):
        """全部数据源失败 → 返回统一提示"""
        from app.tools.food_tools import query_food

        with patch("app.tools.food_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            # geocode 失败
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"status": "0", "geocodes": []}
            mock_client.get.return_value = mock_resp
            mock_client.post.side_effect = httpx.TimeoutException("timeout")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await query_food.ainvoke({"destination": "nowhere"})

        assert "暂不可用" in str(result)
```

- [ ] **Step 2: 运行集成测试 — 预期 FAIL（旧 `query_food` 用 MCP）**

```bash
python -m pytest tests/test_tools/test_food_tools.py::TestQueryFoodIntegration -v
```

- [ ] **Step 3: 替换 `query_food` 函数体**

在 `app/tools/food_tools.py` 中，**完全替换**旧的 `query_food` 函数和 `@tool` 装饰器：

```python
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

    # 确定搜索关键词
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
        return "⚠️ 餐饮查询服务暂不可用，请稍后重试。"

    output = f"## 🍜 {destination} 餐饮推荐\n"
    if warnings:
        output += "\n".join(warnings) + "\n\n"
    output += "\n\n".join(results)
    return output
```

同时**删除**文件中的旧导入行：
```python
from app.mcp_core.client import get_mcp_client  # ← 删除这行
```

- [ ] **Step 4: 运行全部集成测试 — 预期 PASS**

```bash
python -m pytest tests/test_tools/test_food_tools.py::TestQueryFoodIntegration -v
```

- [ ] **Step 5: 运行全部单元测试 — 预期 PASS**

```bash
python -m pytest tests/test_tools/test_food_tools.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_tools/test_food_tools.py app/tools/food_tools.py
git commit -m "feat: rewrite query_food with direct API calls, remove MCP dependency"
```

---

### Task 6: 最终验证

**Files:** 无新增，运行全量测试确认无回归。

- [ ] **Step 1: 检查语法**

```bash
python -c "import ast; ast.parse(open('app/tools/food_tools.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 2: 运行 `food_tools` 全部单元测试**

```bash
python -m pytest tests/test_tools/test_food_tools.py -v
```

- [ ] **Step 3: 确认 `food_tools` 模块可正常导入**

```bash
python -c "from app.tools.food_tools import query_food; print('import OK')"
```

- [ ] **Step 4: 检查 `step_config.py` 导入仍正常**

```bash
python -c "from app.agents.handoffs.step_config import get_step_config; print('step_config OK')"
```

- [ ] **Step 5: 运行既有相关测试（如果有 API Key 则跑集成测试）**

```bash
python -m pytest tests/test_mcp/test_food.py -v 2>&1 || echo "需要真实 API Key，跳过"
```

> 注：`test_food.py` 是旧版集成测试，依赖真实 API Key。如果 `.env` 缺少 Key，测试会失败（预期行为）。可以先用 `pytest -k` 排除它们。

- [ ] **Step 6: Commit**

```bash
git add app/tools/food_tools.py
git commit -m "chore: final cleanup, verify imports and syntax"
```
