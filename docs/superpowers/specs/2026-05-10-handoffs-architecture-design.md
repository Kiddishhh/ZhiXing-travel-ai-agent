# Handoffs 主流程架构设计

## 背景

当前项目已有目的地 Router（`app/agents/routers/destination_router.py`）使用 Send 并行分发模式实现目的地查询。需要在此基础上构建完整的旅行规划 handoffs 主流程，覆盖从需求收集到订单生成的 8 个步骤。

核心需求：
- 每个步骤有独立的 system prompt 和工具集
- 通过 `current_step` 字段控制流程推进和回退
- Agent 调用特殊工具（`select_xxx` / `go_back_to_step`）触发状态转换
- 中间件根据 `current_step` 动态注入 prompt 和 tools
- 复用现有目的地 Router

## 架构概览

```
┌──────────┐    ┌──────────────────┐    ┌──────────────────┐
│  State   │───▶│   Middleware     │───▶│  Step Config     │
│current_  │    │AgentMiddleware   │    │get_step_config() │
│  step    │    │awrap_model_call  │    │prompt + tools    │
└──────────┘    └──────────────────┘    └──────────────────┘
                       │
                       ▼
               ┌──────────────┐
               │  Graph       │
               │  START→agent │
               │  Command 跳转│
               └──────┬───────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │  Tool   │ │  Tool   │ │  Tool   │
    │handoff  │ │router   │ │transport│
    │_tools   │ │_query   │ │_tools   │
    └─────────┘ └─────────┘ └─────────┘
```

模式：单 Agent + 动态切换（方案 C）。一个 agent 节点，通过中间件根据 `current_step` 动态注入 prompt 和 tools。工具执行后通过 `Command` 直接返回 agent 节点，图结构极简。

## 目录结构

```
app/
├── agents/
│   ├── routers/                    # Router 模式（已有，不动）
│   │   ├── __init__.py
│   │   └── destination_router.py
│   ├── handoffs/                   # Handoffs 主流程
│   │   ├── __init__.py             # 导出 create_travel_planner()
│   │   ├── graph.py                # 主图构建
│   │   ├── step_config.py          # get_step_config() 步骤配置
│   │   └── steps/                  # 每步 prompt 配置
│   │       ├── __init__.py
│   │       ├── requirement_collection.py
│   │       ├── destination_recommendation.py
│   │       ├── transport_planning.py
│   │       ├── accommodation_planning.py
│   │       ├── food_planning.py
│   │       ├── itinerary_generation.py
│   │       ├── budget_summarization.py
│   │       └── order_generation.py
│   └── subagents/                  # 保留，未来独立子 agent
├── core/                           # 基础设施层
│   ├── ChromaDB/                   # 不动
│   ├── state.py                    # TravelState + 全部枚举
│   └── middleware.py               # AgentMiddleware 类
├── api/                            # 不动
├── rag/                            # 不动
├── tools/                          # 工具层（按领域拆分）
│   ├── __init__.py                 # TOOL_REGISTRY 全局注册表
│   ├── router_query.py             # 已有，目的地 RAG 检索
│   ├── state_transition.py         # 流程控制工具（前进 + 通用回退 + 进度查询）
│   ├── transport_tools.py          # 交通工具（占位 + 注释）
│   ├── accommodation_tools.py      # 住宿工具（占位 + 注释）
│   ├── food_tools.py               # 餐饮工具（占位 + 注释）
│   ├── budget_tools.py             # 预算工具（占位 + 注释）
│   └── order_tools.py              # 订单工具（占位 + 注释）
├── schemas/                        # 删除，类型移入 core/state.py
├── models/                         # 保留给未来 ORM
├── mcp_core/                       # 保留给未来 MCP
└── utils/                          # 不动
```

### 变更清单

| 操作 | 路径 | 说明 |
|------|------|------|
| 新建 | `app/core/state.py` | TravelState + 枚举 |
| 新建 | `app/core/middleware.py` | AgentMiddleware 类 |
| 新建 | `app/agents/handoffs/graph.py` | 主图构建 |
| 新建 | `app/agents/handoffs/step_config.py` | get_step_config() |
| 新建 | `app/agents/handoffs/steps/*.py` | 8 个步骤配置 |
| 新建 | `app/tools/state_transition.py` | 流程控制工具（填充当前空占位） |
| 新建 | `app/tools/transport_tools.py` | 交通工具（占位 + 详细注释） |
| 新建 | `app/tools/accommodation_tools.py` | 住宿工具（占位 + 详细注释） |
| 新建 | `app/tools/food_tools.py` | 餐饮工具（占位 + 详细注释） |
| 新建 | `app/tools/budget_tools.py` | 预算工具（占位 + 详细注释） |
| 新建 | `app/tools/order_tools.py` | 订单工具（占位 + 详细注释） |
| 修改 | `app/tools/__init__.py` | TOOL_REGISTRY |
| 修改 | `app/agents/handoffs/__init__.py` | 导出入口 |
| 不变 | `app/agents/routers/destination_router.py` | 完全不动 |
| 不变 | `app/rag/` | 完全不动 |
| 不变 | `app/core/ChromaDB/` | 完全不动 |

## State 设计（`app/core/state.py`）

```python
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

# ── 步骤顺序定义（用于回退时的数据清理） ──

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

# 允许回退的步骤（排除 order_generation，它是终点不可回退到）
ALLOWED_BACK_STEPS: set[PlanningStep] = {
    "requirement_collection",
    "destination_recommendation",
    "transport_planning",
    "accommodation_planning",
    "food_planning",
    "itinerary_generation",
    "budget_summarization",
}

# 每个步骤对应的清除字段（回退到该步骤时，清除它之后步骤产生的数据）
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

## 中间件设计（`app/core/middleware.py`）

```python
class AgentMiddleware:
    """步骤配置中间件 - 根据 current_step 动态配置 Agent"""

    def __init__(self, step_config: dict):
        self._step_config = step_config

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        state: TravelState = request.state
        current_step = state.get("current_step", "requirement_collection")
        step_config = self._step_config[current_step]

        # 验证前置依赖
        for required_field in step_config["requires"]:
            if state.get(required_field) is None:
                raise ValueError(
                    f"步骤 {current_step} 需要 {required_field}，但当前未设置"
                )

        # 注入 prompt（渲染 state 字段占位符） + tools
        system_prompt = step_config["prompt"].format(**state)
        modified_request = request.override(
            system_prompt=system_prompt,
            tools=step_config["tools"]
        )
        return await handler(modified_request)


async def create_step_config_middleware() -> AgentMiddleware:
    from app.agents.handoffs.step_config import get_step_config
    step_config = await get_step_config()
    return AgentMiddleware(step_config)
```

## Step 配置设计（`app/agents/handoffs/step_config.py`）

集中管理 8 个步骤。每个步骤定义：
- `prompt`: 该步的 system prompt
- `tools`: 该步可用工具列表（含前进工具、通用回退工具、步骤专属业务工具）
- `requires`: 前置 State 字段

**回退工具策略**：不再为每个步骤创建独立回退工具。只提供一个通用 `go_back_to_step(target_step)` 工具，LLM 根据提示词中的回退选项自主选择目标步骤。工具内部进行步骤验证和无效回退拦截。

步骤概览：

| 步骤 | 前进工具 | 业务工具 | 回退工具 | requires |
|------|---------|---------|---------|----------|
| requirement_collection | `record_requirement_tool` | 无 | 无（首步不可回退） | `[]` |
| destination_recommendation | `select_destination_tool` | `query_destination_info` | `go_back_to_step` | `[user_requirement]` |
| transport_planning | `select_transport_tool` | `query_flight`, `query_train`, `query_driving_route` | `go_back_to_step` | `[user_requirement, selected_destination]` |
| accommodation_planning | `select_accommodation_tool` | `query_hotels`, `query_hostels` | `go_back_to_step` | `[user_requirement, selected_destination, selected_transport]` |
| food_planning | `select_food_tool` | `query_restaurants`, `query_local_food` | `go_back_to_step` | `[... selected_accommodation_types]` |
| itinerary_generation | `generate_itinerary_tool` | 无 | `go_back_to_step` | `[... selected_food_types]` |
| budget_summarization | `summarize_budget_tool` | `calculate_budget` | `go_back_to_step` | `[user_requirement, itinerary]` |
| order_generation | `generate_order_tool` | 无 | `go_back_to_step` | `[user_requirement, itinerary, budget]` |

## Graph 设计（`app/agents/handoffs/graph.py`）

图结构极简，单 agent 节点 + Command 跳转：

```
START → agent_node → END
          ↑    │
          │    │ LLM 调用工具 → 工具返回 Command(goto="agent" 或 "__end__")
          └────┘
```

```python
from langgraph.graph import StateGraph, START
from langchain_community.chat_models import ChatTongyi
from app.core.state import TravelState
from app.core.middleware import create_step_config_middleware
from app.config import settings


async def create_travel_planner() -> StateGraph:
    """构建 handoffs 主流程——单 agent 节点 + Command 跳转"""

    middleware = await create_step_config_middleware()

    llm = ChatTongyi(
        model="qwen-max",
        api_key=settings.dashscope_api_key,
    )

    builder = StateGraph(TravelState)

    # agent_node：标准 LLM 调用节点，middleware 在每次调用前注入 step prompt + tools
    builder.add_node(
        "agent",
        _make_agent_node(llm),
        middleware=[middleware],
    )
    builder.add_edge(START, "agent")

    return builder.compile()


def _make_agent_node(llm):
    """创建 agent 调用节点（闭包捕获 llm 实例）"""
    async def agent_node(state: TravelState) -> dict:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}
    return agent_node
```

终点逻辑由 `generate_order_tool` 控制：返回 `Command(goto="__end__")` 终止图运行。

## Tools 注册机制（`app/tools/__init__.py`）

```python
TOOL_REGISTRY: dict[str, callable] = {}

def register_tool(name: str, func: callable) -> None:
    """注册工具到全局注册表"""
    TOOL_REGISTRY[name] = func
```

## 流程控制工具（`app/tools/state_transition.py`）

### 前进工具（8 个）

全部返回 `Command(update={...}, goto="agent")`。

| 工具名 | 当前步骤 | 更新 current_step → | 附带更新字段 |
|--------|---------|--------------------|-------------|
| `record_requirement_tool` | requirement_collection | destination_recommendation | `user_requirement` |
| `select_destination_tool` | destination_recommendation | transport_planning | `selected_destination` |
| `select_transport_tool` | transport_planning | accommodation_planning | `selected_transport` |
| `select_accommodation_tool` | accommodation_planning | food_planning | `selected_accommodation_types` |
| `select_food_tool` | food_planning | itinerary_generation | `selected_food_types` |
| `generate_itinerary_tool` | itinerary_generation | budget_summarization | `itinerary` |
| `summarize_budget_tool` | budget_summarization | order_generation | `budget` |
| `generate_order_tool` | order_generation | — | `order_id`, `report` |

`generate_order_tool` 返回 `Command(goto="__end__")`，其余返回 `Command(goto="agent")`。

### 通用回退工具（1 个）

```python
@tool
def go_back_to_step(target_step: PlanningStep) -> Command:
    """
    回退到指定的流程步骤。

    此工具将：
    1. 验证 target_step 是否为有效的 PlanningStep
    2. 拦截非法回退（如回退到 order_generation）
    3. 清除目标步骤之后产生的所有数据
    4. 更新 current_step 到目标步骤
    5. 记录回退日志

    参数:
    - target_step: 要回退到的目标步骤名称
      可选值: "requirement_collection", "destination_recommendation",
              "transport_planning", "accommodation_planning",
              "food_planning", "itinerary_generation",
              "budget_summarization"

    不可回退到 "order_generation"（最终步骤）。
    """
    import time
    from app.core.state import PlanningStep, ALLOWED_BACK_STEPS, STEP_CLEANUP_MAP
    from app.utils.logger import app_logger

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
        "禁止回退到 order_generation（最终步骤不可作为回退目标）"
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
```

### 辅助工具

```python
@tool
def check_current_progress() -> str:
    """
    查看当前旅行规划的完成进度。

    返回当前步骤名称、已完成的步骤、剩余步骤等进度信息。
    不修改任何状态。
    """
    # 纯信息查询，由 LLM 在 prompt 指导下根据 State 回答
    return "请根据当前对话状态告知用户规划进度"
```

### 前进工具实现示例

```python
@tool
def select_destination_tool(destination: str) -> Command:
    """
    用户确认目的地后调用。记录选择并推进到交通规划。

    参数:
    - destination: 用户选择的目的地名称，如"西安"
    """
    import time
    from app.utils.logger import app_logger

    app_logger.info(f"目的地确认: {destination}")

    return Command(update={
        "selected_destination": destination,
        "current_step": "transport_planning",
        "updated_at": time.time(),
    }, goto="agent")
```

其余 7 个前进工具结构一致，仅更新字段不同。

## 业务工具文件

### `transport_tools.py`（占位 + 注释）

```python
"""
交通规划工具集

后续对接 API 时的注意事项:
- 高德地图 API: https://lbs.amap.com/api/webservice/summary
- 12306/航司 API 需额外申请
- 所有工具返回统一结构方便前端渲染
"""

# TODO: 接入高德地图驾车路径规划 API
# 接口: GET https://restapi.amap.com/v3/direction/driving
# 参数: origin(lng,lat), destination(lng,lat), strategy(0-5)
# 返回: 路线距离(m)、预估时间(s)、费用(元)
# 前置: 需要先调用地理编码接口获取经纬度坐标
# 工具签名: query_driving_route(origin: str, destination: str) -> str
# 注册名: "query_driving_route"
@tool
async def query_driving_route(origin: str, destination: str) -> str:
    """自驾路线查询（占位）"""
    return f"自驾路线查询功能待实现 (出发: {origin}, 到达: {destination})"


# TODO: 接入高德地图公交路径规划 API
# 接口: GET https://restapi.amap.com/v3/direction/transit/integrated
# 参数: origin(lng,lat), destination(lng,lat), city(城市码), cityd(目的地城市码)
# 返回: 公交/地铁线路、换乘方案、预估时间、票价
# 工具签名: query_flight(departure_city: str, destination: str, date: str) -> str
# 注册名: "query_flight"
@tool
async def query_flight(departure_city: str, destination: str, date: str) -> str:
    """航班查询（占位）"""
    return f"航班查询功能待实现 ({departure_city} → {destination}, {date})"


# TODO: 接入高铁/火车查询 API
# 接口: 12306 官方 API 或第三方聚合接口
# 参数: origin, destination, date
# 返回: 车次、出发/到达时间、时长、票价
# 工具签名: query_train(departure_city: str, destination: str, date: str) -> str
# 注册名: "query_train"
@tool
async def query_train(departure_city: str, destination: str, date: str) -> str:
    """高铁/火车查询（占位）"""
    return f"火车查询功能待实现 ({departure_city} → {destination}, {date})"
```

其余业务工具文件（`accommodation_tools.py`、`food_tools.py`、`budget_tools.py`、`order_tools.py`）结构一致，各自提供 1-3 个占位工具 + 详细的 API 接入注释。

## 集成现有 Router

`destination_recommendation` 步骤通过 `query_destination_info` 工具复用现有 Router：

```
agent_node (destination_recommendation)
  → LLM 调用 query_destination_info("西安", "景点推荐")
    → router_query.py
      → destination_router.create_destination_router()
        → classifier_node → explore_agent + weather_agent
        → compile_report → 返回 Markdown
  → agent_node 展示结果
  → 用户确认 → select_destination_tool → current_step 推进
```

现有代码完全不动：`query_destination_info` 已注册到 TOOL_REGISTRY，`destination_recommendation` 步骤的 tools 列表直接引用。

## 流程示例

### 正常推进

```
用户: 我想去西安玩3天，预算5000

agent (requirement_collection):
  收集: 出发地？出发日期？几人？风格偏好？
  → record_requirement_tool(user_requirement={...})
  → Command(update={current_step: "destination_recommendation"}, goto="agent")

agent (destination_recommendation):
  调用 query_destination_info("西安")
  → 返回西安景点攻略 Markdown
  agent: "西安很不错！大雁塔、兵马俑..."
  用户: "就西安吧"
  → select_destination_tool("西安")
  → Command(update={current_step: "transport_planning", selected_destination: "西安"}, goto="agent")

...依次推进直到 order_generation

agent (order_generation):
  用户: "确认下单"
  → generate_order_tool(...)
  → Command(update={order_id: "ORD-xxx", report: "..."}, goto="__end__")
  → 图运行结束
```

### 通用回退

```
agent (budget_summarization):
  展示预算: 总共 6800 元，超预算
  用户: "住宿太贵了，换个便宜的"

  LLM 根据 prompt 中的回退选项自主决定 → go_back_to_step("accommodation_planning")
    1. 验证 "accommodation_planning" ∈ ALLOWED_BACK_STEPS ✓
    2. 日志: "回退操作: → accommodation_planning"
    3. 清除: selected_food_types, food_options, itinerary, budget, report, order_id
    4. Command(update={current_step: "accommodation_planning", ...}, goto="agent")

agent (accommodation_planning):
  重新以新 prompt + tools 开始住宿规划...
```

## 不涉及

- 不实现各业务工具的具体逻辑（transport、accommodation、food、budget、order 均为占位）
- 不修改 `destination_router.py`、`router_query.py`、`rag/`、`core/ChromaDB/`
- 不涉及 API 层（`app/api/`）
- 不涉及测试（测试在后续实现计划中规划）
