# Tools 补全设计

**日期**: 2026-05-12
**状态**: 已批准

## 背景

4 个工具文件被清空（0 字节），导致运行时 `ImportError`：

- `app/tools/accommodation_tools.py` — 住宿查询（步骤 4）
- `app/tools/food_tools.py` — 餐饮查询（步骤 5）
- `app/tools/budget_tools.py` — 预算计算（步骤 7）
- `app/tools/order_tools.py` — 订单生成（步骤 8）

需要按照已有 `transport_tools.py` 的代码风格补齐实现，并同步更新 `step_config.py` 和 `tools/__init__.py`。

## 设计决策

### 架构模式：轻量直接调用

和交通层的 Coordinator + Subagent 模式不同，住宿/餐饮/预算/订单采用轻量模式——工具函数直接调用 MCP 工具或从 state 计算，不创建中间 Agent。

理由：住宿和餐饮都是单一数据源查询，不需要多轮推理。

### 工具接口简化

每步骤用**一个统一工具**替代原有的多个分散工具，减少 LLM 选择负担：

| 步骤 | 旧工具 | 新工具 |
|------|--------|--------|
| 住宿 | `query_hotels`, `query_hostels` | `query_accommodation` |
| 餐饮 | `query_restaurants`, `query_local_food` | `query_food` |
| 预算 | `calculate_budget`（空壳） | `calculate_budget`（保留重写） |
| 订单 | 无 | `create_order` |

## 详细设计

### 1. `accommodation_tools.py` — 住宿查询

**单一工具** `query_accommodation`：

```python
@tool
async def query_accommodation(
    destination: str,
    check_in_date: str,
    stay_nights: int,
    accommodation_type: str = None,  # "hotel" / "hostel" / None=全部
    budget_min: float = 0,
    budget_max: float = 99999,
) -> str:
```

**数据源**：`aigohotel-mcp`（HTTP，密钥已配置）

**流程**：
1. 通过 `get_mcp_client()` 获取 MCP 工具
2. 筛选 `searchHotels` 工具
3. 根据 `accommodation_type` 映射过滤参数（hotel → 星级酒店过滤，hostel → 青旅/民宿过滤）
4. 调用 MCP 工具获取结果
5. 格式化返回（名称、价格、评分、链接）

**依赖**：`app.mcp_core.client.get_mcp_client`

### 2. `food_tools.py` — 餐饮查询

**单一工具** `query_food`：

```python
@tool
async def query_food(
    destination: str,
    food_type: str = None,  # "restaurant" / "local_snack" / None=全部
    query: str = None,
) -> str:
```

**数据源**：Amap MCP（`maps_geo` + `maps_around_search`）+ Tavily 搜索 MCP（`search_travel_info`）

**流程**：
1. 调用 Amap `maps_geo` 获取目的地坐标
2. 调用 Amap `maps_around_search` 搜周边餐饮（关键词由 `food_type` 决定："餐厅" vs "小吃/本地美食"）
3. 调用 Tavily `search_travel_info` 获取美食攻略/文化背景
4. 合并 Amap 结构化结果 + Tavily 攻略文字，格式化返回

**依赖**：`app.mcp_core.client.get_mcp_client`

### 3. `budget_tools.py` — 预算计算

**保留** `calculate_budget` 为 `@tool`：

```python
@tool
def calculate_budget(
    runtime: ToolRuntime[None, TravelState] = None,
) -> str:
```

**数据源**：无外部 MCP，纯从 `runtime.state` 计算

**流程**：
1. 从 state 读取：`transport_options`, `accommodation_options`, `food_options`, `user_requirement`, `itinerary`
2. 汇总计算：
   - 交通费：从 `transport_options` 提取
   - 住宿费：`price_per_night × (travel_days - 1)`
   - 餐饮费：`estimated_daily_cost × travel_days`
   - 景点门票：按天数和人数预估
   - 杂费：总额的 10%
3. 对比 `budget_max`，超支时标注警告
4. 返回 `BudgetBreakdown` 格式化文本

**依赖**：`app.core.state.TravelState`, `BudgetBreakdown`

### 4. `order_tools.py` — 订单生成

**新增** `create_order`：

```python
@tool
def create_order(
    runtime: ToolRuntime[None, TravelState] = None,
) -> str:
```

**数据源**：无外部 MCP，纯从 `runtime.state` 汇总

**流程**：
1. 从 state 汇总所有旅行信息
2. 格式化为 Markdown 订单摘要（包含目的地、日期、交通、住宿、餐饮、每日行程、预算明细）
3. 返回摘要文本，LLM 展示后调用 `generate_order_tool` 结束流程（`goto="__end__"`）

**依赖**：`app.core.state.TravelState`

## 文件变更清单

### 新建 / 重写
- `app/tools/accommodation_tools.py` — 实现 `query_accommodation`
- `app/tools/food_tools.py` — 实现 `query_food`
- `app/tools/budget_tools.py` — 重写 `calculate_budget`
- `app/tools/order_tools.py` — 实现 `create_order`

### 修改
- `app/tools/__init__.py`：
  - 删除 4 个空文件的 import（`query_hotels`, `query_hostels`, `query_restaurants`, `query_local_food`, `calculate_budget`, `create_order`）
  - 删除已废弃的 transport 旧注册（`query_driving_route`, `query_flight`, `query_train`）
  - 新增 import + register：`query_accommodation`, `query_food`, `calculate_budget`, `create_order`

- `app/agents/handoffs/step_config.py`：
  - 更新 import：`query_accommodation` 替代 `query_hotels`/`query_hostels`，`query_food` 替代 `query_restaurants`/`query_local_food`，新增 `create_order`
  - 更新步骤 4 prompt：引用 `query_accommodation` 单一工具
  - 更新步骤 5 prompt：引用 `query_food` 单一工具
  - 更新步骤 7 prompt：明确 LLM 先调 `calculate_budget` 再调 `summarize_budget_tool`
  - 更新步骤 8 prompt：新增 `create_order` 工具描述
  - 更新各步骤 `tools` 列表

## 测试计划

- `tests/test_mcp/test_accommodation.py` — 测试 `query_accommodation` 调用 aigohotel MCP
- `tests/test_mcp/test_food.py` — 测试 `query_food` 调用 Amap + Tavily MCP
- `tests/test_mcp/test_budget_and_order.py` — 测试 `calculate_budget` 和 `create_order` 纯计算逻辑
