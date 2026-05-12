# Tools 补全实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补全 4 个已清空的工具文件（accommodation/food/budget/order），统一为每步骤单一工具接口，更新 step_config 和 tools/__init__.py。

**Architecture:** 轻量直接调用模式——工具函数直接调用 MCP 工具或从 state 计算，不创建中间 Agent。住宿调用 aigohotel-mcp，餐饮组合 Amap + Tavily，预算和订单纯 state 计算。

**Tech Stack:** Python >= 3.11, LangChain tools, MCP (aigohotel/amap/search), LangGraph ToolRuntime

**Spec:** `docs/superpowers/specs/2026-05-12-tools-completion-design.md`

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `app/tools/accommodation_tools.py` | 重写 | `query_accommodation` — 调用 aigohotel-mcp searchHotels |
| `app/tools/food_tools.py` | 重写 | `query_food` — 调用 Amap geo+around_search + Tavily search |
| `app/tools/budget_tools.py` | 重写 | `calculate_budget` — 从 state 读取数据计算 BudgetBreakdown |
| `app/tools/order_tools.py` | 重写 | `create_order` — 从 state 汇总生成 Markdown 订单 |
| `app/tools/__init__.py` | 修改 | 更新 import + TOOL_REGISTRY，清理失效条目 |
| `app/agents/handoffs/step_config.py` | 修改 | 更新步骤 4/5/7/8 的 prompt 和 tools 列表 |
| `tests/test_mcp/test_accommodation.py` | 新建 | 住宿工具集成测试 |
| `tests/test_mcp/test_food.py` | 新建 | 餐饮工具集成测试 |
| `tests/test_mcp/test_budget_and_order.py` | 新建 | 预算+订单纯计算测试 |

---

### Task 1: 重写 `accommodation_tools.py`

**Files:**
- Rewrite: `app/tools/accommodation_tools.py`
- Create: `tests/test_mcp/test_accommodation.py`

- [ ] **Step 1: 编写住宿工具测试**

```python
"""住宿工具集成测试"""
import pytest
from app.tools.accommodation_tools import query_accommodation


@pytest.mark.asyncio
async def test_query_accommodation_hotel():
    """测试查询酒店"""
    result = await query_accommodation.ainvoke({
        "destination": "北京",
        "check_in_date": "2026-06-01",
        "stay_nights": 2,
        "accommodation_type": "hotel",
    })
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)


@pytest.mark.asyncio
async def test_query_accommodation_all_types():
    """测试查询所有住宿类型（不指定 type）"""
    result = await query_accommodation.ainvoke({
        "destination": "上海",
        "check_in_date": "2026-07-15",
        "stay_nights": 3,
    })
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_mcp/test_accommodation.py -v
```

Expected: FAIL — `query_accommodation` not defined in empty file.

- [ ] **Step 3: 实现 `query_accommodation`**

```python
"""
住宿查询工具
调用 aigohotel-mcp 查询酒店/民宿
"""
from langchain_core.tools import tool
from app.mcp_core.client import get_mcp_client
from app.utils.logger import app_logger


@tool
async def query_accommodation(
    destination: str,
    check_in_date: str,
    stay_nights: int,
    accommodation_type: str = None,
    budget_min: float = 0,
    budget_max: float = 99999,
) -> str:
    """
    查询住宿选项（酒店/民宿/青旅）

    参数说明:
    - destination: 目的地城市，如 "北京"
    - check_in_date: 入住日期，格式 YYYY-MM-DD
    - stay_nights: 入住天数
    - accommodation_type: 住宿类型（可选）。可选值: hotel, hostel。不传则查询全部
    - budget_min: 最低预算（可选，默认 0）
    - budget_max: 最高预算（可选，默认 99999）

    返回:
    - 格式化的住宿选项信息（名称、价格、评分、预订链接）
    """
    app_logger.info(f"🏨 查询住宿: {destination}, {check_in_date}, {stay_nights}晚, type={accommodation_type}")

    manager = await get_mcp_client()
    all_tools = await manager.get_tools()

    # 筛选 aigohotel 工具
    hotel_tools = [
        t for t in all_tools
        if any(kw in t.name.lower() for kw in ['searchhotels', 'gethoteldetail'])
    ]

    search_tool = None
    for t in hotel_tools:
        if 'searchhotels' in t.name.lower():
            search_tool = t
            break

    if search_tool is None:
        return "⚠️ 住宿查询服务暂不可用，请稍后重试。"

    # 构建查询参数
    search_params = {
        "place": destination,
        "checkInDate": check_in_date,
        "stayNights": stay_nights,
    }

    # 映射住宿类型
    if accommodation_type == "hotel":
        search_params["starRating"] = "3,4,5"
    elif accommodation_type == "hostel":
        search_params["tags"] = "hostel,youth_hostel"

    app_logger.info(f"调用 searchHotels: {search_params}")

    try:
        result = await search_tool.ainvoke(search_params)
        result_str = str(result) if not isinstance(result, str) else result

        if not result_str or result_str.strip() == "":
            return f"未找到 {destination} 的{'酒店' if accommodation_type == 'hotel' else '住宿'}选项，请调整日期或预算后重试。"

        return result_str
    except Exception as e:
        app_logger.error(f"住宿查询失败: {e}")
        return f"住宿查询出错: {str(e)}"
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_mcp/test_accommodation.py -v
```

Expected: PASS（测试依赖 MCP 连接，如无网络则跳过）

- [ ] **Step 5: 提交**

```bash
git add app/tools/accommodation_tools.py tests/test_mcp/test_accommodation.py
git commit -m "feat: implement query_accommodation tool with aigohotel-mcp"
```

---

### Task 2: 重写 `food_tools.py`

**Files:**
- Rewrite: `app/tools/food_tools.py`
- Create: `tests/test_mcp/test_food.py`

- [ ] **Step 1: 编写餐饮工具测试**

```python
"""餐饮工具集成测试"""
import pytest
from app.tools.food_tools import query_food


@pytest.mark.asyncio
async def test_query_food_restaurant():
    """测试查询餐厅"""
    result = await query_food.ainvoke({
        "destination": "西安",
        "food_type": "restaurant",
    })
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)


@pytest.mark.asyncio
async def test_query_food_local_snack():
    """测试查询本地小吃"""
    result = await query_food.ainvoke({
        "destination": "成都",
        "food_type": "local_snack",
    })
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)


@pytest.mark.asyncio
async def test_query_food_all():
    """测试查询所有餐饮类型"""
    result = await query_food.ainvoke({
        "destination": "重庆",
    })
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_mcp/test_food.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 `query_food`**

```python
"""
餐饮查询工具
调用 Amap MCP（周边搜索）+ Tavily MCP（美食攻略）
"""
from langchain_core.tools import tool
from app.mcp_core.client import get_mcp_client
from app.utils.logger import app_logger


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

    # 筛选 Amap geocoding 工具
    geo_tool = None
    around_tool = None
    search_tool = None

    for t in all_tools:
        name = t.name.lower()
        if 'maps_geo' in name:
            geo_tool = t
        elif 'maps_around_search' in name:
            around_tool = t
        elif 'search' in name and 'tavily' in name.lower():
            search_tool = t

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
            # 先 geocoding 获取坐标
            geo_result = await geo_tool.ainvoke({"address": destination})
            geo_str = str(geo_result) if not isinstance(geo_result, str) else geo_result
            app_logger.info(f"Geocoding 结果: {geo_str[:200]}")

            # 周边搜索
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_mcp/test_food.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/tools/food_tools.py tests/test_mcp/test_food.py
git commit -m "feat: implement query_food tool with Amap + Tavily MCP"
```

---

### Task 3: 重写 `budget_tools.py` 和 `order_tools.py`

**Files:**
- Rewrite: `app/tools/budget_tools.py`
- Rewrite: `app/tools/order_tools.py`
- Create: `tests/test_mcp/test_budget_and_order.py`

- [ ] **Step 1: 编写预算和订单测试**

```python
"""预算计算和订单生成测试"""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import ToolMessage
from app.tools.budget_tools import calculate_budget
from app.tools.order_tools import create_order


def make_mock_runtime(state_overrides=None):
    """构造模拟的 ToolRuntime"""
    state = {
        "current_step": "budget_summarization",
        "user_requirement": {
            "departure_city": "北京",
            "destination": "西安",
            "departure_date": "2026-06-01",
            "travel_days": 3,
            "adult_count": 2,
            "children_count": 0,
            "budget_min": 2000,
            "budget_max": 5000,
            "budget_level": "comfort",
            "travel_styles": ["culture"],
            "special_needs": None,
        },
        "selected_destination": "西安",
        "selected_transport": "train",
        "selected_accommodation_types": ["star_hotel"],
        "selected_food_types": ["specialty", "local"],
        "transport_options": [
            {"transport_type": "train", "details": "G651", "departure_time": "08:00", "arrival_time": "12:30", "duration": "4h30m", "price": 550.0}
        ],
        "accommodation_options": [
            {"name": "西安钟楼酒店", "type": "star_hotel", "location": "钟楼附近", "price_per_night": 400.0, "rating": 4.5, "amenities": ["WiFi", "早餐"]}
        ],
        "food_options": [
            {"type": "specialty", "recommendations": ["羊肉泡馍", "肉夹馍"], "estimated_daily_cost": 150.0}
        ],
        "itinerary": [
            {"day_number": 1, "date": "2026-06-01", "activities": ["兵马俑"], "meals": ["午餐: 回民街"], "accommodation": "西安钟楼酒店", "transport": "步行"},
            {"day_number": 2, "date": "2026-06-02", "activities": ["大雁塔"], "meals": ["晚餐: 饺子宴"], "accommodation": "西安钟楼酒店", "transport": "公交"},
            {"day_number": 3, "date": "2026-06-03", "activities": ["城墙"], "meals": ["午餐: 凉皮"], "accommodation": "无", "transport": "高铁返回"},
        ],
    }
    if state_overrides:
        state.update(state_overrides)

    mock_runtime = MagicMock()
    mock_runtime.state = state
    mock_runtime.tool_call_id = "test-001"
    return mock_runtime


class TestCalculateBudget:
    """测试预算计算工具"""

    def test_calculate_budget_normal(self):
        """测试正常预算计算"""
        runtime = make_mock_runtime()
        result = calculate_budget.invoke({}, runtime=runtime)

        assert isinstance(result, str)
        assert "交通" in result
        assert "住宿" in result
        assert "餐饮" in result
        assert "总计" in result
        print(result)

    def test_calculate_budget_over_limit(self):
        """测试超预算场景"""
        runtime = make_mock_runtime()
        # 设置极低预算上限
        runtime.state["user_requirement"]["budget_max"] = 500
        result = calculate_budget.invoke({}, runtime=runtime)

        assert isinstance(result, str)
        assert "超支" in result or "超出" in result
        print(result)

    def test_calculate_budget_missing_data(self):
        """测试缺少数据时的处理"""
        runtime = make_mock_runtime({"transport_options": [], "accommodation_options": [], "food_options": []})
        result = calculate_budget.invoke({}, runtime=runtime)

        assert isinstance(result, str)
        print(result)


class TestCreateOrder:
    """测试订单生成工具"""

    def test_create_order(self):
        """测试生成完整订单"""
        runtime = make_mock_runtime()
        result = create_order.invoke({}, runtime=runtime)

        assert isinstance(result, str)
        assert "西安" in result
        assert "2026-06-01" in result
        assert "兵马俑" in result
        print(result)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_mcp/test_budget_and_order.py -v
```

Expected: FAIL（budget_tools.py 和 order_tools.py 为空文件）

- [ ] **Step 3: 实现 `calculate_budget`**

```python
"""
预算计算工具
从 TravelState 读取各步骤数据，计算完整费用明细
"""
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from app.core.state import TravelState, BudgetBreakdown
from app.utils.logger import app_logger


@tool
def calculate_budget(
    runtime: ToolRuntime[None, TravelState] = None,
) -> str:
    """
    计算完整旅行预算明细。

    从 runtime.state 读取交通/住宿/餐饮/行程数据，自动汇总计算:
    - 交通费
    - 住宿费 (price_per_night × 住宿天数)
    - 餐饮费 (estimated_daily_cost × 旅行天数)
    - 景点门票预估 (每成人每天 100 元)
    - 杂费 (总额的 10%)

    返回格式化的预算文本，供 LLM 展示给用户。
    如果超出用户预算上限，会标注超支警告。
    """
    state = runtime.state
    req = state.get("user_requirement", {})
    budget_limit = req.get("budget_max", 0) or 0
    travel_days = req.get("travel_days", 1)
    adult_count = req.get("adult_count", 1)
    children_count = req.get("children_count", 0)

    app_logger.info(
        f"[{runtime.tool_call_id}] 计算预算: "
        f"{travel_days}天, {adult_count}成人 + {children_count}儿童, 限额 {budget_limit}"
    )

    # ── 1. 交通费 ──
    transport_total = 0.0
    transport_detail = []
    transport_options = state.get("transport_options", []) or []
    for t in transport_options:
        price = t.get("price", 0)
        transport_total += price
        transport_detail.append(f"  {t.get('details', '未知交通')}: ¥{price}")

    # ── 2. 住宿费 ──
    accommodation_total = 0.0
    accommodation_detail = []
    accommodation_options = state.get("accommodation_options", []) or []
    nights = max(travel_days - 1, 1)  # 住 travel_days - 1 晚
    for a in accommodation_options:
        price_per_night = a.get("price_per_night", 0)
        acc_total = price_per_night * nights
        accommodation_total += acc_total
        accommodation_detail.append(
            f"  {a.get('name', '未知住宿')}: ¥{price_per_night}/晚 × {nights}晚 = ¥{acc_total}"
        )

    # ── 3. 餐饮费 ──
    food_total = 0.0
    food_detail = []
    food_options = state.get("food_options", []) or []
    for f in food_options:
        daily = f.get("estimated_daily_cost", 0)
        f_total = daily * travel_days
        food_total += f_total
        food_detail.append(
            f"  {f.get('type', '未知餐饮')}: ¥{daily}/天 × {travel_days}天 = ¥{f_total}"
        )

    # ── 4. 景点门票 ──
    attractions_total = adult_count * travel_days * 100
    if children_count > 0:
        attractions_total += children_count * travel_days * 50  # 儿童半价

    # ── 5. 小计与杂费 ──
    subtotal = transport_total + accommodation_total + food_total + attractions_total
    misc = round(subtotal * 0.1, 2)
    grand_total = subtotal + misc

    # ── 6. 格式化输出 ──
    lines = [
        "## 💰 旅行预算明细",
        "",
        f"**出发地**: {req.get('departure_city', '未知')}",
        f"**目的地**: {state.get('selected_destination', '未知')}",
        f"**出行天数**: {travel_days}天",
        f"**人数**: {adult_count}成人 + {children_count}儿童",
        f"**住宿天数**: {nights}晚",
        "",
        "### 交通费",
        *transport_detail,
        f"> 交通小计: ¥{transport_total}",
        "",
        "### 住宿费",
        *accommodation_detail,
        f"> 住宿小计: ¥{accommodation_total}",
        "",
        "### 餐饮费",
        *food_detail,
        f"> 餐饮小计: ¥{food_total}",
        "",
        "### 景点门票",
        f"  预估门票: ¥{attractions_total} (成人{adult_count}人 × {travel_days}天 × ¥100)",
        "",
        "### 杂费 (10%)",
        f"  杂费: ¥{misc}",
        "",
        "---",
        f"**总计: ¥{grand_total}**",
    ]

    # 超支警告
    if budget_limit and grand_total > budget_limit:
        over = grand_total - budget_limit
        lines.append(f"")
        lines.append(f"⚠️ **超支警告**: 总计 ¥{grand_total} 超出预算上限 ¥{budget_limit}，超出 ¥{over}！")
        lines.append(f"建议回退调整交通/住宿/餐饮选择。")

    app_logger.info(
        f"[{runtime.tool_call_id}] 预算计算完成: 总计 ¥{grand_total}, "
        f"限额 ¥{budget_limit}"
    )

    return "\n".join(lines)
```

- [ ] **Step 4: 实现 `create_order`**

```python
"""
订单生成工具
从 TravelState 汇总所有数据，生成最终订单摘要
"""
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from app.core.state import TravelState
from app.utils.logger import app_logger


@tool
def create_order(
    runtime: ToolRuntime[None, TravelState] = None,
) -> str:
    """
    生成最终旅行订单摘要。

    从 runtime.state 汇总所有旅行信息（需求、目的地、交通、住宿、餐饮、行程、预算），
    格式化为 Markdown 订单供用户最终确认。

    注意: 此工具只生成摘要文本，不结束流程。
    流程结束由 generate_order_tool 负责（goto="__end__"）。
    """
    state = runtime.state
    req = state.get("user_requirement", {})
    destination = state.get("selected_destination", "未知")
    transport = state.get("selected_transport", "未知")
    accommodation_types = state.get("selected_accommodation_types", [])
    food_types = state.get("selected_food_types", [])
    itinerary = state.get("itinerary", []) or []
    budget = state.get("budget", {})
    budget_total = budget.get("total", 0)

    app_logger.info(f"[{runtime.tool_call_id}] 生成订单摘要: {destination}")

    lines = [
        "# 🎉 旅行订单确认",
        "",
        "## 📋 基本信息",
        f"- **出发地**: {req.get('departure_city', '未知')}",
        f"- **目的地**: {destination}",
        f"- **出发日期**: {req.get('departure_date', '未知')}",
        f"- **出行天数**: {req.get('travel_days', 0)}天",
        f"- **人数**: {req.get('adult_count', 0)}成人 + {req.get('children_count', 0)}儿童",
        f"- **预算上限**: ¥{req.get('budget_max', '不限')}",
        f"- **旅行风格**: {', '.join(req.get('travel_styles', []))}",
        "",
        "## ✈️ 交通方式",
        f"- 已选: {transport}",
        *(f"  {t.get('details', '')}: ¥{t.get('price', 0)}" for t in (state.get('transport_options', []) or [])),
        "",
        "## 🏨 住宿",
        f"- 类型: {', '.join(accommodation_types)}",
        *(f"  {a.get('name', '')}: ¥{a.get('price_per_night', 0)}/晚" for a in (state.get('accommodation_options', []) or [])),
        "",
        "## 🍜 餐饮",
        f"- 偏好: {', '.join(food_types)}",
        "",
        "## 📅 每日行程",
    ]

    for day in itinerary:
        day_num = day.get("day_number", "?")
        date = day.get("date", "")
        activities = day.get("activities", [])
        meals = day.get("meals", [])
        acc = day.get("accommodation", "无")

        lines.append(f"### 第 {day_num} 天 ({date})")
        lines.append(f"- 住宿: {acc}")
        for act in activities:
            lines.append(f"  - {act}")
        for meal in meals:
            lines.append(f"  - {meal}")
        lines.append("")

    lines.extend([
        "## 💰 费用汇总",
        f"- 总计: **¥{budget_total}**",
        "",
        "---",
        "> 请确认以上信息，确认后调用 `generate_order_tool` 完成下单。",
    ])

    app_logger.info(f"[{runtime.tool_call_id}] 订单摘要生成完成")

    return "\n".join(lines)
```

- [ ] **Step 5: 运行测试确认通过**

```bash
python -m pytest tests/test_mcp/test_budget_and_order.py -v
```

Expected: PASS（纯计算测试，无需 MCP 连接）

- [ ] **Step 6: 提交**

```bash
git add app/tools/budget_tools.py app/tools/order_tools.py tests/test_mcp/test_budget_and_order.py
git commit -m "feat: implement calculate_budget and create_order tools"
```

---

### Task 4: 更新 `app/tools/__init__.py`

**Files:**
- Modify: `app/tools/__init__.py`

- [ ] **Step 1: 更新导入和注册**

将 `app/tools/__init__.py` 替换为以下内容：

```python
"""
工具注册中心

TOOL_REGISTRY 是全局工具注册表，所有工具在此注册后可由 step_config 按名引用。
"""
from typing import Callable

from .router_query import query_destination_info
from .state_transition import (
    record_requirement_tool,
    select_destination_tool,
    select_transport_tool,
    select_accommodation_tool,
    select_food_tool,
    generate_itinerary_tool,
    summarize_budget_tool,
    generate_order_tool,
    go_back_to_step,
    go_back_to_requirement,
    go_back_to_destination,
    go_back_to_transport,
    go_back_to_accommodation,
    go_back_to_food,
    go_back_to_itinerary,
    go_back_to_budget,
    check_current_progress,
)
from .transport_tools import query_transport_options
from .accommodation_tools import query_accommodation
from .food_tools import query_food
from .budget_tools import calculate_budget
from .order_tools import create_order

# ── 全局工具注册表 ──

TOOL_REGISTRY: dict[str, Callable] = {}


def register_tool(name: str, func: Callable) -> None:
    """注册工具到全局注册表"""
    TOOL_REGISTRY[name] = func


# ── 注册所有工具 ──

register_tool("query_destination_info", query_destination_info)
register_tool("record_requirement_tool", record_requirement_tool)
register_tool("select_destination_tool", select_destination_tool)
register_tool("select_transport_tool", select_transport_tool)
register_tool("select_accommodation_tool", select_accommodation_tool)
register_tool("select_food_tool", select_food_tool)
register_tool("generate_itinerary_tool", generate_itinerary_tool)
register_tool("summarize_budget_tool", summarize_budget_tool)
register_tool("generate_order_tool", generate_order_tool)
register_tool("go_back_to_step", go_back_to_step)
register_tool("go_back_to_requirement", go_back_to_requirement)
register_tool("go_back_to_destination", go_back_to_destination)
register_tool("go_back_to_transport", go_back_to_transport)
register_tool("go_back_to_accommodation", go_back_to_accommodation)
register_tool("go_back_to_food", go_back_to_food)
register_tool("go_back_to_itinerary", go_back_to_itinerary)
register_tool("go_back_to_budget", go_back_to_budget)
register_tool("check_current_progress", check_current_progress)

# ── 业务查询工具 ──
register_tool("query_transport_options", query_transport_options)
register_tool("query_accommodation", query_accommodation)
register_tool("query_food", query_food)
register_tool("calculate_budget", calculate_budget)
register_tool("create_order", create_order)

__all__ = [
    "TOOL_REGISTRY",
    "register_tool",
    "query_destination_info",
]
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/tools/__init__.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add app/tools/__init__.py
git commit -m "refactor: update TOOL_REGISTRY with new unified tools"
```

---

### Task 5: 更新 `app/agents/handoffs/step_config.py`

**Files:**
- Modify: `app/agents/handoffs/step_config.py`

- [ ] **Step 1: 更新 import 区域**

将第 26-30 行的旧 import：

```python
from app.tools.router_query import query_destination_info
from app.tools.transport_tools import query_transport_options
from app.tools.accommodation_tools import query_hotels, query_hostels
from app.tools.food_tools import query_restaurants, query_local_food
from app.tools.budget_tools import calculate_budget
```

替换为：

```python
from app.tools.router_query import query_destination_info
from app.tools.transport_tools import query_transport_options
from app.tools.accommodation_tools import query_accommodation
from app.tools.food_tools import query_food
from app.tools.budget_tools import calculate_budget
from app.tools.order_tools import create_order
```

- [ ] **Step 2: 更新步骤 4 prompt 和 tools 列表**

将步骤 4（`accommodation_planning`）的 prompt 中：

```
**可用的住宿查询工具**:
- `query_hotels` — 酒店查询
- `query_hostels` — 民宿查询
```

替换为：

```
**可用的住宿查询工具**:
- `query_accommodation` — 统一住宿查询（支持 hotel/hostel 类型筛选）
```

将 tools 列表中的 `query_hotels, query_hostels,` 替换为 `query_accommodation,`

- [ ] **Step 3: 更新步骤 5 prompt 和 tools 列表**

将步骤 5（`food_planning`）的 prompt 中：

```
**可用的餐饮查询工具**:
- `query_restaurants` — 餐厅查询
- `query_local_food` — 本地小吃查询
```

替换为：

```
**可用的餐饮查询工具**:
- `query_food` — 统一餐饮查询（支持 restaurant/local_snack 类型筛选，融合周边搜索和美食攻略）
```

将 tools 列表中的 `query_restaurants, query_local_food,` 替换为 `query_food,`

- [ ] **Step 4: 更新步骤 7 prompt 和 tools 列表**

步骤 7（`budget_summarization`）的 prompt 保持不变（已有 `calculate_budget` 和 `summarize_budget_tool`），只确认 tools 列表包含 `calculate_budget, summarize_budget_tool,`

- [ ] **Step 5: 更新步骤 8 prompt 和 tools 列表**

将步骤 8（`order_generation`）的 prompt 中任务描述更新，在 "展示完整旅行计划摘要" 之前加入对 `create_order` 的引用：

```
**任务**:
1. 调用 `create_order` 生成最终订单摘要
2. 展示完整旅行计划摘要给用户确认
3. 用户确认后 → 调用 `generate_order_tool` (自动结束流程)
```

将 tools 列表更新为包含 `create_order,`：

```python
"tools": [
    create_order,
    generate_order_tool,
    go_back_to_budget, go_back_to_itinerary, go_back_to_food,
    go_back_to_accommodation, go_back_to_transport,
    go_back_to_destination, go_back_to_requirement,
    go_back_to_step,
    check_current_progress,
],
```

- [ ] **Step 6: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/agents/handoffs/step_config.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 7: 提交**

```bash
git add app/agents/handoffs/step_config.py
git commit -m "refactor: update step_config for unified tool interface"
```

---

### Task 6: 最终验证

**Files:**
- None (验证 only)

- [ ] **Step 1: 检查所有 Python 文件语法**

```bash
python -c "import ast, sys, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.py') if 'venv' not in str(p) and '__pycache__' not in str(p)]; print('All files OK')"
```

Expected: `All files OK`

- [ ] **Step 2: 运行纯计算测试（无需 MCP）**

```bash
python -m pytest tests/test_mcp/test_budget_and_order.py -v
```

Expected: PASS

- [ ] **Step 3: 验证 import 链无循环引用**

```bash
python -c "from app.tools import TOOL_REGISTRY; print(f'{len(TOOL_REGISTRY)} tools registered'); [print(f'  {k}') for k in sorted(TOOL_REGISTRY.keys())]"
```

Expected: 输出所有注册工具（约 22 个）

- [ ] **Step 4: 提交最终状态**

```bash
git status
git diff --stat
```
