# Handoffs 主流程架构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 Handoffs 主流程架构骨架——State 定义、工具注册机制、流程控制工具、步骤配置、中间件、图构建。

**Architecture:** 单 agent 节点 + StepConfigMiddleware 动态注入 prompt/tools + Command 跳转。8 个步骤按 `current_step` 顺序流转，通用 `go_back_to_step` 工具实现回退。

**Tech Stack:** LangGraph 1.0.5, LangChain Community (ChatTongyi qwen-max), LangChain Core (@tool, Command)

---

### Task 1: 创建目录结构和 `__init__.py` 占位文件

**Files:**
- Create: `app/agents/handoffs/steps/__init__.py`
- Create: `app/agents/handoffs/__init__.py` (覆盖空占位)

- [ ] **Step 1: 创建 steps 包 `__init__.py`**

```bash
mkdir -p "D:\AI agent\知行智能旅游规划助手\app\agents\handoffs\steps"
```

- [ ] **Step 2: 写入 steps 包初始化文件**

写入 `app/agents/handoffs/steps/__init__.py`：

```python
"""Handoffs 步骤配置模块（占位——各步骤配置在 step_config.py 中集中定义）"""
```

- [ ] **Step 3: 写入 handoffs 包初始化文件**

写入 `app/agents/handoffs/__init__.py`：

```python
"""Handoffs 主流程——单 Agent + 步骤动态切换"""

from app.agents.handoffs.graph import create_travel_planner

__all__ = ["create_travel_planner"]
```

- [ ] **Step 4: 提交**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
git add app/agents/handoffs/steps/__init__.py app/agents/handoffs/__init__.py
git commit -m "chore: create handoffs package structure with placeholder init files"
```

---

### Task 2: 实现 `app/core/state.py` — TravelState + 枚举 + 辅助数据

**Files:**
- Create: `app/core/state.py`

- [ ] **Step 1: 写入完整的 state.py**

```python
"""
旅行规划系统 State 定义

包含枚举类型、子结构 TypedDict、主 TravelState、初始化函数和回退辅助数据。
"""
from typing import Literal, TypedDict, Optional, NotRequired, List

from langgraph.graph import MessagesState

# ── 枚举 ──

PlanningStep = Literal[
    "requirement_collection",
    "destination_recommendation",
    "transport_planning",
    "accommodation_planning",
    "food_planning",
    "itinerary_generation",
    "budget_summarization",
    "order_generation",
]

TravelStyle = Literal["relaxation", "culture", "adventure", "food"]
BudgetLevel = Literal["economy", "comfort", "luxury"]
TransportType = Literal["flight", "train", "driving"]
AccommodationType = Literal["star_hotel", "economy_hotel", "hostel", "youth_hostel"]
FoodType = Literal["specialty", "chain", "local"]

# ── 子结构 ──

class UserRequirement(TypedDict):
    departure_city: str
    destination: str
    departure_date: str
    travel_days: int
    adult_count: int
    children_count: int
    budget_min: Optional[float]
    budget_max: Optional[float]
    budget_level: BudgetLevel
    travel_styles: List[TravelStyle]
    special_needs: Optional[str]


class DestinationInfo(TypedDict):
    name: str
    description: str
    weather_info: Optional[str]
    attractions: List[str]
    estimated_cost: Optional[float]


class TransportInfo(TypedDict):
    transport_type: TransportType
    details: str
    departure_time: str
    arrival_time: str
    duration: str
    price: float


class AccommodationInfo(TypedDict):
    name: str
    type: AccommodationType
    location: str
    price_per_night: float
    rating: Optional[float]
    amenities: List[str]


class FoodInfo(TypedDict):
    type: FoodType
    recommendations: List[str]
    estimated_daily_cost: float


class ItineraryDay(TypedDict):
    day_number: int
    activities: List[str]
    meals: List[str]
    accommodation: str


class BudgetBreakdown(TypedDict):
    transport: float
    accommodation: float
    food: float
    attractions: float
    misc: float
    total: float


# ── 步骤顺序 ──

STEP_ORDER: List[PlanningStep] = [
    "requirement_collection",
    "destination_recommendation",
    "transport_planning",
    "accommodation_planning",
    "food_planning",
    "itinerary_generation",
    "budget_summarization",
    "order_generation",
]

# ── 回退辅助数据 ──

ALLOWED_BACK_STEPS: set[PlanningStep] = {
    "requirement_collection",
    "destination_recommendation",
    "transport_planning",
    "accommodation_planning",
    "food_planning",
    "itinerary_generation",
    "budget_summarization",
}

STEP_CLEANUP_MAP: dict[PlanningStep, list[str]] = {
    "requirement_collection": [
        "selected_destination", "selected_transport",
        "selected_accommodation_types", "selected_food_types",
        "destination_options", "transport_options",
        "accommodation_options", "food_options",
        "itinerary", "budget", "report", "order_id",
    ],
    "destination_recommendation": [
        "selected_transport", "selected_accommodation_types",
        "selected_food_types", "transport_options",
        "accommodation_options", "food_options",
        "itinerary", "budget", "report", "order_id",
    ],
    "transport_planning": [
        "selected_accommodation_types", "selected_food_types",
        "accommodation_options", "food_options",
        "itinerary", "budget", "report", "order_id",
    ],
    "accommodation_planning": [
        "selected_food_types", "food_options",
        "itinerary", "budget", "report", "order_id",
    ],
    "food_planning": [
        "itinerary", "budget", "report", "order_id",
    ],
    "itinerary_generation": [
        "budget", "report", "order_id",
    ],
    "budget_summarization": [
        "report", "order_id",
    ],
}

# ── 主 State ──

class TravelState(MessagesState):
    """旅行规划系统主状态，继承 MessagesState 自动获得 messages 字段"""
    current_step: NotRequired[PlanningStep]
    user_requirement: NotRequired[UserRequirement]
    selected_destination: NotRequired[str]
    selected_transport: NotRequired[TransportType]
    selected_accommodation_types: NotRequired[List[AccommodationType]]
    selected_food_types: NotRequired[List[FoodType]]
    destination_options: NotRequired[List[DestinationInfo]]
    transport_options: NotRequired[List[TransportInfo]]
    accommodation_options: NotRequired[List[AccommodationInfo]]
    food_options: NotRequired[List[FoodInfo]]
    itinerary: NotRequired[List[ItineraryDay]]
    budget: NotRequired[BudgetBreakdown]
    report: NotRequired[str]
    order_id: NotRequired[str]
    approval_pending: NotRequired[bool]
    approval_reason: NotRequired[str]
    user_id: NotRequired[str]
    session_id: NotRequired[str]
    created_at: NotRequired[float]
    updated_at: NotRequired[float]


def create_initial_state(user_id: str, session_id: str) -> dict:
    """创建初始状态，返回符合 TravelState 的 dict"""
    import time
    return {
        "messages": [],
        "current_step": "requirement_collection",
        "destination_options": [],
        "transport_options": [],
        "accommodation_options": [],
        "food_options": [],
        "approval_pending": False,
        "user_id": user_id,
        "session_id": session_id,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/core/state.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add app/core/state.py
git commit -m "feat: add TravelState with enums, sub-structures, and back-step helpers"
```

---

### Task 3: 实现 `app/tools/state_transition.py` — 流程控制工具

**Files:**
- Create: `app/tools/state_transition.py` (覆盖当前空占位)

- [ ] **Step 1: 写入完整的 state_transition.py**

```python
"""
流程控制工具集

包含: 8 个前进工具 + 1 个通用回退工具 + 1 个进度查询工具。

关键设计:
- 前进工具: 接收用户确认的数据, 记录到 State 并推进 current_step
- go_back_to_step: 通用回退, LLM 根据 prompt 自主选择目标步骤
  工具内部验证步骤有效性, 拦截非法回退, 清除回退目标之后的数据
- check_current_progress: 纯信息查询, 不修改状态
"""
import time

from langchain_core.tools import tool
from langgraph.types import Command

from app.core.state import (
    PlanningStep, ALLOWED_BACK_STEPS, STEP_CLEANUP_MAP,
    TransportType, AccommodationType, FoodType,
    UserRequirement, BudgetBreakdown,
)
from app.utils.logger import app_logger

# ── 前进工具 (8 个) ──

@tool
def record_requirement_tool(user_requirement: UserRequirement) -> Command:
    """
    记录用户旅行需求并推进到目的地推荐步骤。

    参数:
    - user_requirement: 完整的用户需求对象
      包含: departure_city, destination, departure_date, travel_days,
            adult_count, children_count, budget_min, budget_max,
            budget_level, travel_styles, special_needs
    """
    app_logger.info(f"需求记录完成: {user_requirement.get('destination')}")
    return Command(update={
        "user_requirement": user_requirement,
        "current_step": "destination_recommendation",
        "updated_at": time.time(),
    }, goto="agent")


@tool
def select_destination_tool(destination: str) -> Command:
    """
    用户确认目的地后调用。记录选择并推进到交通规划。

    参数:
    - destination: 用户选择的目的地名称, 如 "西安"
    """
    app_logger.info(f"目的地确认: {destination}")
    return Command(update={
        "selected_destination": destination,
        "current_step": "transport_planning",
        "updated_at": time.time(),
    }, goto="agent")


@tool
def select_transport_tool(transport_type: TransportType) -> Command:
    """
    用户确认交通方式后调用。记录选择并推进到住宿规划。

    参数:
    - transport_type: 选择的交通方式 ("flight" / "train" / "driving")
    """
    app_logger.info(f"交通确认: {transport_type}")
    return Command(update={
        "selected_transport": transport_type,
        "current_step": "accommodation_planning",
        "updated_at": time.time(),
    }, goto="agent")


@tool
def select_accommodation_tool(accommodation_types: list[AccommodationType]) -> Command:
    """
    用户确认住宿类型后调用。记录选择并推进到餐饮规划。

    参数:
    - accommodation_types: 选择的住宿类型列表
      如 ["star_hotel", "hostel"]
    """
    app_logger.info(f"住宿确认: {accommodation_types}")
    return Command(update={
        "selected_accommodation_types": accommodation_types,
        "current_step": "food_planning",
        "updated_at": time.time(),
    }, goto="agent")


@tool
def select_food_tool(food_types: list[FoodType]) -> Command:
    """
    用户确认餐饮类型后调用。记录选择并推进到行程生成。

    参数:
    - food_types: 选择的餐饮类型列表
      如 ["specialty", "local"]
    """
    app_logger.info(f"餐饮确认: {food_types}")
    return Command(update={
        "selected_food_types": food_types,
        "current_step": "itinerary_generation",
        "updated_at": time.time(),
    }, goto="agent")


@tool
def generate_itinerary_tool(itinerary: list[dict]) -> Command:
    """
    行程生成完成后调用。记录行程并推进到预算汇总。

    参数:
    - itinerary: 每日行程列表, 每项包含:
      day_number, activities, meals, accommodation
    """
    app_logger.info(f"行程生成完成: {len(itinerary)} 天")
    return Command(update={
        "itinerary": itinerary,
        "current_step": "budget_summarization",
        "updated_at": time.time(),
    }, goto="agent")


@tool
def summarize_budget_tool(budget: BudgetBreakdown) -> Command:
    """
    预算计算完成后调用。记录预算并推进到订单生成。

    参数:
    - budget: 预算明细对象
      包含: transport, accommodation, food, attractions, misc, total
    """
    app_logger.info(f"预算汇总完成: 总计 {budget.get('total')} 元")
    return Command(update={
        "budget": budget,
        "current_step": "order_generation",
        "updated_at": time.time(),
    }, goto="agent")


@tool
def generate_order_tool() -> Command:
    """
    用户确认下单后调用。生成订单并结束旅行规划流程。

    生成模拟订单号, 将 report 标记为 final, 图运行结束。
    """
    import uuid
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    app_logger.info(f"订单生成: {order_id}")
    return Command(update={
        "order_id": order_id,
        "report": "旅行规划已完成, 订单已生成。感谢使用知行智能旅游规划助手!",
        "updated_at": time.time(),
    }, goto="__end__")

# ── 通用回退工具 (1 个) ──

@tool
def go_back_to_step(target_step: PlanningStep) -> Command:
    """
    回退到指定的流程步骤。

    此工具将:
    1. 验证 target_step 是否为有效的 PlanningStep
    2. 拦截非法回退 (如回退到 order_generation)
    3. 清除目标步骤之后产生的所有数据
    4. 更新 current_step 到目标步骤
    5. 记录回退日志

    参数:
    - target_step: 要回退到的目标步骤名称
      可选值: "requirement_collection", "destination_recommendation",
              "transport_planning", "accommodation_planning",
              "food_planning", "itinerary_generation",
              "budget_summarization"
      不可回退到 "order_generation" (最终步骤)。
    """
    # ── 1. 验证目标步骤是否有效 ──
    valid_steps = {
        "requirement_collection", "destination_recommendation",
        "transport_planning", "accommodation_planning",
        "food_planning", "itinerary_generation",
        "budget_summarization", "order_generation",
    }
    assert target_step in valid_steps, (
        f"无效步骤: '{target_step}'。有效步骤: {sorted(valid_steps)}"
    )

    # ── 2. 拦截回退到最终步骤 ──
    assert target_step != "order_generation", (
        "禁止回退到 order_generation (最终步骤不可作为回退目标)"
    )

    # ── 3. 日志记录回退操作 ──
    app_logger.warning(f"回退操作: → {target_step}")

    # ── 4. 清除回退目标之后的数据 ──
    cleanup_fields = STEP_CLEANUP_MAP.get(target_step, [])
    update = {"current_step": target_step, "updated_at": time.time()}
    for field in cleanup_fields:
        update[field] = None
    app_logger.info(f"已清除字段: {cleanup_fields}")

    # ── 5. 返回 Command ──
    return Command(update=update, goto="agent")


# ── 辅助工具 ──

@tool
def check_current_progress() -> str:
    """
    查看当前旅行规划的完成进度。

    返回当前步骤名称、已完成的步骤、剩余步骤等进度信息。
    不修改任何状态——LLM 根据 State 中的 current_step 回答。
    """
    return "请根据当前对话状态中的 current_step 字段告知用户规划进度"
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/tools/state_transition.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add app/tools/state_transition.py
git commit -m "feat: add flow control tools — 8 forward + go_back_to_step + check_progress"
```

---

### Task 4: 改造 `app/tools/__init__.py` — TOOL_REGISTRY 注册机制

**Files:**
- Modify: `app/tools/__init__.py`

- [ ] **Step 1: 写入新的 tools/__init__.py**

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
    check_current_progress,
)

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
register_tool("check_current_progress", check_current_progress)

__all__ = [
    "TOOL_REGISTRY",
    "register_tool",
    "query_destination_info",
]
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/tools/__init__.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 验证导入可用**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "from app.tools import TOOL_REGISTRY; print(f'已注册 {len(TOOL_REGISTRY)} 个工具:', list(TOOL_REGISTRY.keys()))"
```

Expected: `已注册 11 个工具: ['query_destination_info', 'record_requirement_tool', 'select_destination_tool', 'select_transport_tool', 'select_accommodation_tool', 'select_food_tool', 'generate_itinerary_tool', 'summarize_budget_tool', 'generate_order_tool', 'go_back_to_step', 'check_current_progress']`

- [ ] **Step 4: 提交**

```bash
git add app/tools/__init__.py
git commit -m "feat: add TOOL_REGISTRY with register_tool and 11 registered tools"
```

---

### Task 5: 创建业务占位工具文件

**Files:**
- Create: `app/tools/transport_tools.py`
- Create: `app/tools/accommodation_tools.py`
- Create: `app/tools/food_tools.py`
- Create: `app/tools/budget_tools.py`
- Create: `app/tools/order_tools.py`
- Modify: `app/tools/__init__.py` (追加注册)

- [ ] **Step 1: 写入 transport_tools.py**

```python
"""
交通规划工具集

后续对接 API 时的注意事项:
- 高德地图 API: https://lbs.amap.com/api/webservice/summary
- 12306/航司 API 需额外申请
- 所有工具返回统一结构方便前端渲染
"""
from langchain_core.tools import tool


# TODO: 接入高德地图驾车路径规划 API
# 接口: GET https://restapi.amap.com/v3/direction/driving
# 参数: origin(lng,lat), destination(lng,lat), strategy(0-5)
# 返回: 路线距离(m)、预估时间(s)、费用(元)
# 前置: 需要先调用地理编码接口获取经纬度坐标
# 工具签名: query_driving_route(origin: str, destination: str) -> str
# 注册名: "query_driving_route"
@tool
async def query_driving_route(origin: str, destination: str) -> str:
    """自驾路线查询 (占位)"""
    return f"自驾路线查询功能待实现 (出发: {origin}, 到达: {destination})"


# TODO: 接入航班查询 API
# 接口: 飞猪/携程开放平台 或 航空数据聚合接口
# 参数: departure_city, destination, date (YYYY-MM-DD)
# 返回: 航班号、出发/到达时间、时长、票价
# 工具签名: query_flight(departure_city: str, destination: str, date: str) -> str
# 注册名: "query_flight"
@tool
async def query_flight(departure_city: str, destination: str, date: str) -> str:
    """航班查询 (占位)"""
    return f"航班查询功能待实现 ({departure_city} → {destination}, {date})"


# TODO: 接入高铁/火车查询 API
# 接口: 12306 官方 API 或第三方聚合接口
# 参数: departure_city, destination, date (YYYY-MM-DD)
# 返回: 车次、出发/到达时间、时长、票价
# 工具签名: query_train(departure_city: str, destination: str, date: str) -> str
# 注册名: "query_train"
@tool
async def query_train(departure_city: str, destination: str, date: str) -> str:
    """高铁/火车查询 (占位)"""
    return f"火车查询功能待实现 ({departure_city} → {destination}, {date})"
```

- [ ] **Step 2: 写入 accommodation_tools.py**

```python
"""
住宿规划工具集

后续对接 API 时的注意事项:
- 携程/飞猪开放平台 API
- 或高德地图 POI 搜索 (酒店/民宿)
- 需要获取目的地经纬度坐标
- 结果包含: 名称、类型、位置、价格、评分、设施
"""
from langchain_core.tools import tool


# TODO: 接入酒店查询 API
# 接口: 携程/飞猪开放平台 或 高德 POI 搜索 (keywords=酒店)
# 参数: destination(目的地), check_in(入住日期), check_out(离店日期),
#       price_min, price_max, rating_min
# 返回: 酒店列表 (名称、星级、价格、评分、位置)
# 工具签名: query_hotels(destination: str, check_in: str, check_out: str) -> str
# 注册名: "query_hotels"
@tool
async def query_hotels(destination: str, check_in: str, check_out: str) -> str:
    """酒店查询 (占位)"""
    return f"酒店查询功能待实现 (目的地: {destination}, {check_in} ~ {check_out})"


# TODO: 接入民宿查询 API
# 接口: 途家/爱彼迎开放平台 或 高德 POI 搜索 (keywords=民宿)
# 参数: destination, check_in, check_out, price_range
# 返回: 民宿列表 (名称、价格、评分、位置、设施)
# 工具签名: query_hostels(destination: str, check_in: str, check_out: str) -> str
# 注册名: "query_hostels"
@tool
async def query_hostels(destination: str, check_in: str, check_out: str) -> str:
    """民宿查询 (占位)"""
    return f"民宿查询功能待实现 (目的地: {destination}, {check_in} ~ {check_out})"
```

- [ ] **Step 3: 写入 food_tools.py**

```python
"""
餐饮规划工具集

后续对接 API 时的注意事项:
- 大众点评/美团开放平台 API
- 或高德地图 POI 搜索 (keywords=美食)
- 需要获取目的地经纬度坐标
- 结果按美食类型过滤: 特色美食 / 连锁快餐 / 本地小吃
"""
from langchain_core.tools import tool


# TODO: 接入餐厅查询 API
# 接口: 大众点评/美团开放平台 或 高德 POI 搜索 (keywords=美食)
# 参数: destination(目的地), food_type(美食类型), count(返回数量)
# 返回: 餐厅列表 (名称、类型、人均消费、评分、位置)
# 工具签名: query_restaurants(destination: str, food_type: str = "") -> str
# 注册名: "query_restaurants"
@tool
async def query_restaurants(destination: str, food_type: str = "") -> str:
    """餐厅查询 (占位)"""
    type_hint = f" ({food_type})" if food_type else ""
    return f"餐厅查询功能待实现 (目的地: {destination}{type_hint})"


# TODO: 接入本地小吃查询 API
# 接口: 大众点评/美团 或 高德 POI 搜索 (keywords=小吃)
# 参数: destination, count
# 返回: 本地小吃列表 (名称、价格区间、推荐指数)
# 工具签名: query_local_food(destination: str) -> str
# 注册名: "query_local_food"
@tool
async def query_local_food(destination: str) -> str:
    """本地小吃查询 (占位)"""
    return f"本地小吃查询功能待实现 (目的地: {destination})"
```

- [ ] **Step 4: 写入 budget_tools.py**

```python
"""
预算计算工具集

后续实现时的注意事项:
- 汇总交通 + 住宿 + 餐饮 + 景点门票 + 杂费
- 与用户预算限额对比, 超支时给出警告
- 返回 BudgetBreakdown 结构
"""
from langchain_core.tools import tool


# TODO: 实现预算计算逻辑
# 输入: transport_cost, accommodation_cost, food_cost, attractions_cost, misc_cost
# 逻辑: 逐项累加 → 比较 budget_max → 超支警告
# 返回: BudgetBreakdown (transport, accommodation, food, attractions, misc, total)
# 工具签名: calculate_budget(transport: float, accommodation: float, food: float,
#                            attractions: float, misc: float, budget_limit: float) -> dict
# 注册名: "calculate_budget"
@tool
def calculate_budget(transport: float, accommodation: float, food: float,
                     attractions: float, misc: float, budget_limit: float) -> str:
    """预算计算 (占位)"""
    total = transport + accommodation + food + attractions + misc
    if total > budget_limit:
        return (f"预算计算结果: 总计 {total} 元, 超出预算 {total - budget_limit} 元。"
                f"建议回退调整。")
    return f"预算计算结果: 总计 {total} 元, 在预算 {budget_limit} 元以内。"
```

- [ ] **Step 5: 写入 order_tools.py**

```python
"""
订单生成工具集

后续实现时的注意事项:
- 汇总全部旅行计划生成最终订单
- 订单内容包括: 目的地、交通、住宿、餐饮、行程、预算
- 生成订单号并持久化 (PostgreSQL)
- 可选: 支付接口对接
"""
from langchain_core.tools import tool


# TODO: 实现订单生成逻辑
# 输入: 无 (从 State 中读取全部规划数据)
# 逻辑: 汇总 → 生成订单号 → 持久化到 PostgreSQL
# 返回: 订单号 + 完整订单摘要
# 工具签名: create_order() -> str
# 注册名: "create_order"
@tool
def create_order() -> str:
    """订单生成 (占位)"""
    return "订单生成功能待实现。请调用 generate_order_tool 完成流程。"
```

- [ ] **Step 6: 在 tools/__init__.py 中追加占位工具注册**

在文件末尾追加以下导入和注册：

```python
# ── 注册业务占位工具 ──
from .transport_tools import query_driving_route, query_flight, query_train
from .accommodation_tools import query_hotels, query_hostels
from .food_tools import query_restaurants, query_local_food
from .budget_tools import calculate_budget
from .order_tools import create_order

register_tool("query_driving_route", query_driving_route)
register_tool("query_flight", query_flight)
register_tool("query_train", query_train)
register_tool("query_hotels", query_hotels)
register_tool("query_hostels", query_hostels)
register_tool("query_restaurants", query_restaurants)
register_tool("query_local_food", query_local_food)
register_tool("calculate_budget", calculate_budget)
register_tool("create_order", create_order)
```

- [ ] **Step 7: 语法检查全部占位文件**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "
import ast
for f in ['transport_tools', 'accommodation_tools', 'food_tools', 'budget_tools', 'order_tools']:
    path = f'app/tools/{f}.py'
    ast.parse(open(path, encoding='utf-8').read())
    print(f'{path}: OK')
"
```

Expected: 5 行 OK

- [ ] **Step 8: 验证 TOOL_REGISTRY 总量**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "from app.tools import TOOL_REGISTRY; print(f'共 {len(TOOL_REGISTRY)} 个工具'); [print(f'  {k}') for k in TOOL_REGISTRY]"
```

Expected: `共 20 个工具`

- [ ] **Step 9: 提交**

```bash
git add app/tools/transport_tools.py app/tools/accommodation_tools.py app/tools/food_tools.py app/tools/budget_tools.py app/tools/order_tools.py app/tools/__init__.py
git commit -m "feat: add business tool stubs with detailed API integration comments"
```

---

### Task 6: 实现 `app/agents/handoffs/step_config.py` — 步骤配置

**Files:**
- Create: `app/agents/handoffs/step_config.py`

- [ ] **Step 1: 写入完整的 step_config.py**

```python
"""
步骤配置中心

集中管理 8 个步骤的 prompt、工具列表和前置依赖。
get_step_config() 返回的字典由 AgentMiddleware 在每次 LLM 调用前查询。
"""
from app.tools.state_transition import (
    record_requirement_tool,
    select_destination_tool,
    select_transport_tool,
    select_accommodation_tool,
    select_food_tool,
    generate_itinerary_tool,
    summarize_budget_tool,
    generate_order_tool,
    go_back_to_step,
    check_current_progress,
)
from app.tools.router_query import query_destination_info
from app.tools.transport_tools import query_flight, query_train, query_driving_route
from app.tools.accommodation_tools import query_hotels, query_hostels
from app.tools.food_tools import query_restaurants, query_local_food
from app.tools.budget_tools import calculate_budget


async def get_step_config() -> dict:
    """
    获取步骤配置字典。

    每步结构:
    - prompt: system prompt (使用 {field_name} 占位符, 由 middleware 渲染)
    - tools: 该步可用的工具函数列表
    - requires: 前置 State 字段 (未就绪时 middleware 报错拦截)
    """
    return {
        # ========== 步骤 1: 需求收集 ==========
        "requirement_collection": {
            "prompt": """你是专业的旅行规划顾问, 负责收集用户的旅行需求。

**当前阶段**: 需求收集 (第 1 步, 共 8 步)

**需要收集的信息**:
- 🏠 出发地点
- 📅 出发日期
- 🗓️ 出行天数
- 👥 人数 (成人/儿童)
- 💰 预算范围 (元/人)
- 🎨 旅行风格: relaxation/culture/adventure/food (可多选)
- 📝 特殊需求 (可选)

**操作指南**:
- 一次只问 1-2 个问题, 保持对话自然
- 信息完整后 → 调用 `record_requirement_tool` 进入下一步
- 这是第一步, 无回退选项
""",
            "tools": [
                record_requirement_tool,
                check_current_progress,
            ],
            "requires": []
        },

        # ========== 步骤 2: 目的地推荐 ==========
        "destination_recommendation": {
            "prompt": """你是目的地推荐专家。

**当前阶段**: 目的地推荐 (第 2 步, 共 8 步)

**用户需求**:
- 出发日期: {user_requirement}
- 预算: {user_requirement}
- 旅行风格: {user_requirement}

**任务**:
1. 调用 `query_destination_info` 获取目的地信息
2. 根据需求推荐 3 个目的地, 说明特色和适合理由
3. 用户确认后 → 调用 `select_destination_tool`

**回退选项**:
- 重新规划整个旅行 → `go_back_to_step("requirement_collection")`
""",
            "tools": [
                query_destination_info,
                select_destination_tool,
                go_back_to_step,
                check_current_progress,
            ],
            "requires": ["user_requirement"]
        },

        # ========== 步骤 3: 交通规划 ==========
        "transport_planning": {
            "prompt": """你是交通规划专家。

**当前阶段**: 交通规划 (第 3 步, 共 8 步)

**已确定信息**:
- 目的地: {selected_destination}

**可用的交通查询工具**:
- `query_flight` — 航班查询
- `query_train` — 高铁/火车查询
- `query_driving_route` — 自驾路线查询

**任务**:
1. 推荐交通方式: ✈️ 航班 / 🚄 高铁 / 🚗 自驾
2. 调用对应工具查询具体信息
3. 用户确认后 → 调用 `select_transport_tool`

**回退选项**:
- 换目的地 → `go_back_to_step("destination_recommendation")`
- 重新规划整个旅行 → `go_back_to_step("requirement_collection")`
""",
            "tools": [
                query_flight, query_train, query_driving_route,
                select_transport_tool,
                go_back_to_step,
                check_current_progress,
            ],
            "requires": ["user_requirement", "selected_destination"]
        },

        # ========== 步骤 4: 住宿规划 ==========
        "accommodation_planning": {
            "prompt": """你是住宿规划专家。

**当前阶段**: 住宿规划 (第 4 步, 共 8 步)

**已确定信息**:
- 目的地: {selected_destination}
- 交通: {selected_transport}

**可用的住宿查询工具**:
- `query_hotels` — 酒店查询
- `query_hostels` — 民宿查询

**任务**:
1. 推荐住宿类型: 🏨 星级酒店 / 🏠 民宿 / 🛏️ 青旅 (可多选)
2. 调用查询工具获取具体选项
3. 用户确认后 → 调用 `select_accommodation_tool`

**回退选项**:
- 换交通 → `go_back_to_step("transport_planning")`
- 换目的地 → `go_back_to_step("destination_recommendation")`
- 重新规划整个旅行 → `go_back_to_step("requirement_collection")`
""",
            "tools": [
                query_hotels, query_hostels,
                select_accommodation_tool,
                go_back_to_step,
                check_current_progress,
            ],
            "requires": ["user_requirement", "selected_destination", "selected_transport"]
        },

        # ========== 步骤 5: 餐饮规划 ==========
        "food_planning": {
            "prompt": """你是餐饮规划专家。

**当前阶段**: 餐饮规划 (第 5 步, 共 8 步)

**已确定信息**:
- 目的地: {selected_destination}
- 住宿类型: {selected_accommodation_types}

**可用的餐饮查询工具**:
- `query_restaurants` — 餐厅查询
- `query_local_food` — 本地小吃查询

**任务**:
1. 推荐餐饮类型: 🍜 特色美食 / 🍔 连锁快餐 / 🍘 本地小吃 (可多选)
2. 调用查询工具获取具体选项
3. 用户确认后 → 调用 `select_food_tool`

**回退选项**:
- 换住宿 → `go_back_to_step("accommodation_planning")`
- 换交通 → `go_back_to_step("transport_planning")`
- 换目的地 → `go_back_to_step("destination_recommendation")`
- 重新规划整个旅行 → `go_back_to_step("requirement_collection")`
""",
            "tools": [
                query_restaurants, query_local_food,
                select_food_tool,
                go_back_to_step,
                check_current_progress,
            ],
            "requires": ["user_requirement", "selected_destination", "selected_transport", "selected_accommodation_types"]
        },

        # ========== 步骤 6: 行程生成 ==========
        "itinerary_generation": {
            "prompt": """你是行程规划专家。

**当前阶段**: 行程生成 (第 6 步, 共 8 步)

**已收集信息**:
- 目的地: {selected_destination}
- 交通: {selected_transport}
- 住宿: {selected_accommodation_types}
- 餐饮: {selected_food_types}

**任务**:
1. 根据用户需求生成每日详细行程
2. 包含景点、餐饮、住宿安排
3. 用户确认后 → 调用 `generate_itinerary_tool`

**回退选项**:
- 改餐饮 → `go_back_to_step("food_planning")`
- 改住宿 → `go_back_to_step("accommodation_planning")`
- 改交通 → `go_back_to_step("transport_planning")`
- 换目的地 → `go_back_to_step("destination_recommendation")`
- 重新规划整个旅行 → `go_back_to_step("requirement_collection")`
""",
            "tools": [
                generate_itinerary_tool,
                go_back_to_step,
                check_current_progress,
            ],
            "requires": ["user_requirement", "selected_destination", "selected_transport", "selected_accommodation_types", "selected_food_types"]
        },

        # ========== 步骤 7: 预算汇总 ==========
        "budget_summarization": {
            "prompt": """你是预算分析专家。

**当前阶段**: 预算汇总 (第 7 步, 共 8 步)

**任务**:
1. 调用 `calculate_budget` 计算费用明细
2. 展示: 交通 + 住宿 + 餐饮 + 门票 + 杂费
3. 如超预算, 建议回退调整
4. 用户确认后 → 调用 `summarize_budget_tool`

**回退选项**:
- 改行程 → `go_back_to_step("itinerary_generation")`
- 改餐饮 → `go_back_to_step("food_planning")`
- 改住宿 → `go_back_to_step("accommodation_planning")`
- 改交通 → `go_back_to_step("transport_planning")`
- 换目的地 → `go_back_to_step("destination_recommendation")`
- 重新规划整个旅行 → `go_back_to_step("requirement_collection")`
- 回到任意步骤 → `go_back_to_step("<step_name>")`
""",
            "tools": [
                calculate_budget,
                summarize_budget_tool,
                go_back_to_step,
                check_current_progress,
            ],
            "requires": ["user_requirement", "itinerary"]
        },

        # ========== 步骤 8: 订单生成 ==========
        "order_generation": {
            "prompt": """你是订单处理专家。

**当前阶段**: 订单生成 (第 8 步, 共 8 步)

**任务**:
1. 确认用户准备下单
2. 展示完整旅行计划摘要
3. 用户确认后 → 调用 `generate_order_tool` (自动结束流程)

**回退选项** (最后修改机会):
- 看预算 → `go_back_to_step("budget_summarization")`
- 改行程 → `go_back_to_step("itinerary_generation")`
- 改餐饮 → `go_back_to_step("food_planning")`
- 改住宿 → `go_back_to_step("accommodation_planning")`
- 改交通 → `go_back_to_step("transport_planning")`
- 换目的地 → `go_back_to_step("destination_recommendation")`
- 重新规划整个旅行 → `go_back_to_step("requirement_collection")`
- 回到任意步骤 → `go_back_to_step("<step_name>")`
""",
            "tools": [
                generate_order_tool,
                go_back_to_step,
                check_current_progress,
            ],
            "requires": ["user_requirement", "itinerary", "budget"]
        },
    }
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/agents/handoffs/step_config.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add app/agents/handoffs/step_config.py
git commit -m "feat: add step_config with 8-step prompts, tools, and requires"
```

---

### Task 7: 实现 `app/core/middleware.py` — AgentMiddleware 中间件

**Files:**
- Create: `app/core/middleware.py`

- [ ] **Step 1: 写入完整的 middleware.py**

```python
"""
步骤配置中间件

AgentMiddleware 实现 awrap_model_call 钩子, 在每次 LLM 调用前:
1. 读取 current_step
2. 查 step_config 获取对应 prompt + tools
3. 验证前置依赖
4. 注入配置到模型请求
"""
from typing import Callable, Any

from langgraph.config import ModelRequest, ModelResponse
from app.core.state import TravelState
from app.utils.logger import app_logger


class AgentMiddleware:
    """步骤配置中间件 - 根据 current_step 动态配置 Agent"""

    def __init__(self, step_config: dict):
        self._step_config = step_config

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """
        根据 current_step 动态注入 prompt 和 tools
        """
        state: TravelState = request.state
        current_step = state.get("current_step", "requirement_collection")

        app_logger.info(f"当前步骤: {current_step}")

        if current_step not in self._step_config:
            app_logger.error(f"未知步骤: {current_step}")
            raise ValueError(f"未知步骤: {current_step}")

        step_config = self._step_config[current_step]

        # ── 验证前置依赖 ──
        for required_field in step_config["requires"]:
            val = state.get(required_field)
            if val is None:
                error_msg = (
                    f"步骤 {current_step} 需要 '{required_field}' 字段, "
                    f"但当前未设置"
                )
                app_logger.error(f"前置依赖缺失: {error_msg}")
                raise ValueError(error_msg)
            app_logger.debug(f"前置依赖满足: {required_field}")

        # ── 注入 prompt + tools ──
        try:
            system_prompt = step_config["prompt"].format(**state)
        except KeyError as e:
            app_logger.warning(f"prompt 占位符无法渲染: {e}, 使用原始模板")
            system_prompt = step_config["prompt"]

        modified_request = request.override(
            system_prompt=system_prompt,
            tools=step_config["tools"]
        )

        app_logger.info(
            f"已注入步骤配置: {len(step_config['tools'])} 个工具"
        )
        return await handler(modified_request)


async def create_step_config_middleware() -> AgentMiddleware:
    """
    工厂函数: 创建预加载配置的 AgentMiddleware 实例
    """
    from app.agents.handoffs.step_config import get_step_config

    step_config = await get_step_config()
    app_logger.info("AgentMiddleware 创建完成")
    return AgentMiddleware(step_config)
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/core/middleware.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add app/core/middleware.py
git commit -m "feat: add AgentMiddleware with awrap_model_call for dynamic step config injection"
```

---

### Task 8: 实现 `app/agents/handoffs/graph.py` — 主图构建

**Files:**
- Create: `app/agents/handoffs/graph.py`

- [ ] **Step 1: 写入完整的 graph.py**

```python
"""
Handoffs 主流程 Graph 构建

单 agent 节点 + Command 跳转。
每次 LLM 调用前, AgentMiddleware 根据 current_step 注入对应 prompt + tools。
工具返回 Command 直接跳回 agent 节点 (或 __end__ 终止)。
"""
from langgraph.graph import StateGraph, START
from langchain_community.chat_models import ChatTongyi
from app.core.state import TravelState
from app.core.middleware import create_step_config_middleware
from app.config import settings
from app.utils.logger import app_logger


async def create_travel_planner() -> StateGraph:
    """
    构建 handoffs 主流程 Graph。

    图结构:
        START → agent → END
                  ↑   │
                  │   │ LLM 调用工具 → Command(goto="agent" / "__end__")
                  └───┘

    返回编译后的图 (await graph.ainvoke(initial_state) 即可运行)
    """
    middleware = await create_step_config_middleware()

    llm = ChatTongyi(
        model="qwen-max",
        api_key=settings.dashscope_api_key,
    )

    builder = StateGraph(TravelState)
    builder.add_node(
        "agent",
        _make_agent_node(llm),
        middleware=[middleware],
    )
    builder.add_edge(START, "agent")

    app_logger.info("Handoffs 主流程 Graph 构建完成")
    return builder.compile()


def _make_agent_node(llm: ChatTongyi):
    """创建 agent 调用节点 (闭包捕获 llm 实例)"""

    async def agent_node(state: TravelState) -> dict:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    return agent_node
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/agents/handoffs/graph.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add app/agents/handoffs/graph.py
git commit -m "feat: add create_travel_planner graph with single-agent + Command routing"
```

---

### Task 9: 全项目语法检查 + 集成验证

**Files:**
- 无新建，验证所有文件

- [ ] **Step 1: 全项目语法检查**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast, sys, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('app').rglob('*.py')]; print('全项目语法检查通过')"
```

Expected: `全项目语法检查通过`

- [ ] **Step 2: 验证核心导入链**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "
from app.core.state import TravelState, PlanningStep, create_initial_state, ALLOWED_BACK_STEPS, STEP_CLEANUP_MAP
from app.tools import TOOL_REGISTRY, query_destination_info
print(f'State 定义正常, {len(PlanningStep.__args__)} 个步骤')
print(f'TOOL_REGISTRY: {len(TOOL_REGISTRY)} 个工具')
print(f'ALLOWED_BACK_STEPS: {len(ALLOWED_BACK_STEPS)} 个可回退步骤')
print(f'STEP_CLEANUP_MAP: {len(STEP_CLEANUP_MAP)} 个清理配置')
state = create_initial_state('u1', 's1')
print(f'初始 State current_step: {state[\"current_step\"]}')
print('核心导入链验证通过')
"
```

Expected: 5 行输出，最后一行 `核心导入链验证通过`

- [ ] **Step 3: 验证 async 工厂函数可正常创建**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "
import asyncio
async def test():
    # 测试 step_config 加载
    from app.agents.handoffs.step_config import get_step_config
    config = await get_step_config()
    print(f'步骤配置加载成功: {len(config)} 个步骤')
    for name, step in config.items():
        print(f'  {name}: {len(step[\"tools\"])} 工具, requires={step[\"requires\"]}')
    # 测试 middleware 创建
    from app.core.middleware import AgentMiddleware
    mw = AgentMiddleware(config)
    print(f'中间件创建成功 (step_config 条目: {len(mw._step_config)})')
asyncio.run(test())
"
```

Expected: 8 个步骤 + 中间件创建成功

- [ ] **Step 4: 提交**

```bash
git add .
git commit -m "chore: full project syntax check passed, all imports verified"
```

---

### Task 10: 最终 Git 状态确认

- [ ] **Step 1: 检查未提交文件**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
git status
```

Expected: `nothing to commit, working tree clean`

- [ ] **Step 2: 查看 commit 历史**

```bash
git log --oneline -10
```

Expected: 包含本次所有 feat/chore 提交
