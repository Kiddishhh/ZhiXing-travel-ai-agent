# 交通规划 Subagent State 层实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建交通规划 subagent 的 State 类型层，用三个详细 TypedDict（FlightOption/TrainOption/DrivingRoute）替换通用 TransportInfo。

**Architecture:** 在 `app/agents/subagents/transport_state.py` 新建类型文件；修改 `app/core/state.py` 用三个独立 `Annotated[list, add]` 字段替换 `transport_options`；同步更新 `state_transition.py` 中的工具签名和引用。

**Tech Stack:** Python 3.11+, LangGraph TypedDict State, `typing.NotRequired` / `typing.Annotated`, `operator.add`

---

### Task 1: 新建 `app/agents/subagents/transport_state.py`

**Files:**
- Create: `app/agents/subagents/transport_state.py`

- [ ] **Step 1: 写入完整的 transport_state.py**

```python
"""
交通规划 Subagent State 定义

包含三种交通方式的详细 TypedDict 和 TransportState（子图内部状态）。
"""
from operator import add
from typing import Annotated, NotRequired, TypedDict, List

from app.core.state import TransportType


class FlightOption(TypedDict):
    """单个航班选项"""
    flight_number: str        # 航班号
    airline: str              # 航空公司
    departure_airport: str    # 出发机场
    arrival_airport: str      # 到达机场
    departure_time: str       # 起飞时间
    arrival_time: str         # 降落时间
    duration: str             # 飞行时长
    price: float              # 价格
    cabin_class: str          # 舱位等级
    available_seats: int      # 剩余座位


class TrainOption(TypedDict):
    """单个车次选项"""
    train_number: str                 # 车次号
    departure_station: str            # 出发站
    arrival_station: str              # 到达站
    departure_time: str               # 发车时间
    arrival_time: str                 # 到站时间
    duration: str                     # 运行时长
    seat_types: List[str]             # 可选座位类型
    prices: dict[str, float]          # 各座位类型价格
    available: bool                   # 是否有票


class DrivingRoute(TypedDict):
    """自驾路线"""
    route_name: str          # 路线名称
    distance: str            # 总距离
    duration: str            # 预计时长
    toll_fee: float          # 过路费
    fuel_cost: float         # 油费估算
    steps: List[str]         # 导航步骤
    waypoints: List[str]     # 途经城市


class TransportState(TypedDict):
    """交通规划子图内部状态"""

    # 用户选择
    selected_transport: NotRequired[TransportType]

    # 查询参数
    origin_city: NotRequired[str]
    destination_city: NotRequired[str]
    departure_date: NotRequired[str]
    passenger_count: NotRequired[int]

    # 查询结果（累加）
    flight_options: Annotated[list[FlightOption], add]
    train_options: Annotated[list[TrainOption], add]
    driving_routes: Annotated[list[DrivingRoute], add]

    # 最终选择
    selected_flight: NotRequired[FlightOption]
    selected_train: NotRequired[TrainOption]
    selected_route: NotRequired[DrivingRoute]
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/agents/subagents/transport_state.py', encoding='utf-8').read()); print('OK')"
```

期望输出: `OK`

- [ ] **Step 3: 提交**

```bash
git add app/agents/subagents/transport_state.py
git commit -m "feat: add transport subagent state types (FlightOption, TrainOption, DrivingRoute, TransportState)"
```

---

### Task 2: 修改 `app/core/state.py` — 替换 TransportInfo 为三个详细类型

**Files:**
- Modify: `app/core/state.py`

- [ ] **Step 1: 修改导入区 — 新增从 subagents 导入、移除未使用的 List 导入无需处理**

在 `from operator import add` 之前插入新导入行。定位第 6 行 `from operator import add`，在其后追加：

```python
# 旧（第 6-8 行）:
from operator import add
from typing import Annotated, Literal, TypedDict, Optional, NotRequired, List

# 新:
from operator import add
from typing import Annotated, Literal, TypedDict, Optional, NotRequired, List

from app.agents.subagents.transport_state import FlightOption, TrainOption, DrivingRoute
```

注：需要确认 `transport_state.py` 创建后再执行此步骤。

- [ ] **Step 2: 删除 TransportInfo 类（第 54-60 行）**

删除以下内容：
```python
class TransportInfo(TypedDict):
    transport_type: TransportType
    details: str
    departure_time: str
    arrival_time: str
    duration: str
    price: float
```

- [ ] **Step 3: 更新 STEP_CLEANUP_MAP — requirement_collection 步骤（第 119-127 行）**

替换 `requirement_collection` 条目中的 `"transport_options"` 为三个新字段名：

```python
# 旧:
"requirement_collection": [
    "selected_destination", "selected_transport",
    "selected_accommodation_types", "selected_food_types",
    "destination_options", "transport_options",
    "accommodation_options", "food_options",
    "itinerary", "budget", "report", "order_id",
],

# 新:
"requirement_collection": [
    "selected_destination", "selected_transport",
    "selected_accommodation_types", "selected_food_types",
    "destination_options",
    "flight_options", "train_options", "driving_routes",
    "accommodation_options", "food_options",
    "itinerary", "budget", "report", "order_id",
],
```

- [ ] **Step 4: 更新 STEP_CLEANUP_MAP — destination_recommendation 步骤（第 128-132 行）**

同样替换 `"transport_options"` 为三个新字段名：

```python
# 旧:
"destination_recommendation": [
    "selected_transport", "selected_accommodation_types",
    "selected_food_types", "transport_options",
    "accommodation_options", "food_options",
    "itinerary", "budget", "report", "order_id",
],

# 新:
"destination_recommendation": [
    "selected_transport", "selected_accommodation_types",
    "selected_food_types",
    "flight_options", "train_options", "driving_routes",
    "accommodation_options", "food_options",
    "itinerary", "budget", "report", "order_id",
],
```

- [ ] **Step 5: 更新 STEP_CLEANUP_MAP — transport_planning 步骤（第 133-137 行）**

同样替换 `"transport_options"` 为三个新字段名：

```python
# 旧:
"transport_planning": [
    "selected_accommodation_types", "selected_food_types",
    "accommodation_options", "food_options",
    "itinerary", "budget", "report", "order_id",
],

# 新:
"transport_planning": [
    "selected_accommodation_types", "selected_food_types",
    "flight_options", "train_options", "driving_routes",
    "accommodation_options", "food_options",
    "itinerary", "budget", "report", "order_id",
],
```

- [ ] **Step 6: 替换 TravelState 中的 transport_options 字段（第 175 行附近）**

```python
# 旧:
transport_options: NotRequired[List[TransportInfo]]         # 交通选项

# 新:
# ── 交通选项（三种交通方式的详细结果，累加）
flight_options: NotRequired[Annotated[list[FlightOption], add]]       # 航班选项
train_options: NotRequired[Annotated[list[TrainOption], add]]         # 高铁选项
driving_routes: NotRequired[Annotated[list[DrivingRoute], add]]       # 自驾路线
```

- [ ] **Step 7: 更新 create_initial_state 函数（第 203 行附近）**

```python
# 旧:
"transport_options": [],

# 新:
"flight_options": [],
"train_options": [],
"driving_routes": [],
```

- [ ] **Step 8: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/core/state.py', encoding='utf-8').read()); print('OK')"
```

期望输出: `OK`

- [ ] **Step 9: 提交**

```bash
git add app/core/state.py
git commit -m "refactor: replace TransportInfo with FlightOption/TrainOption/DrivingRoute in TravelState"
```

---

### Task 3: 修改 `app/tools/state_transition.py` — 适配新类型

**Files:**
- Modify: `app/tools/state_transition.py`

- [ ] **Step 1: 更新导入（第 18-25 行）**

```python
# 旧:
from app.core.state import (
    TravelState,
    PlanningStep, ALLOWED_BACK_STEPS, STEP_CLEANUP_MAP,
    TransportType, AccommodationType, FoodType,
    TransportInfo, AccommodationInfo, FoodInfo,
    ItineraryDay,
    UserRequirement, BudgetBreakdown, BudgetLevel,
)

# 新:
from app.core.state import (
    TravelState,
    PlanningStep, ALLOWED_BACK_STEPS, STEP_CLEANUP_MAP,
    TransportType, AccommodationType, FoodType,
    AccommodationInfo, FoodInfo,
    ItineraryDay,
    UserRequirement, BudgetBreakdown, BudgetLevel,
)
from app.agents.subagents.transport_state import FlightOption, TrainOption, DrivingRoute
```

- [ ] **Step 2: 更新 select_transport_tool 签名和实现（第 124-155 行）**

`transport_options` 参数从 `list[TransportInfo]` 改为三个可选列表参数。LLM 调用时按需传入：

```python
# 旧:
@tool
def select_transport_tool(
    transport_type: TransportType,
    transport_options: list[TransportInfo],
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """
    用户确认交通方式后调用。记录选择和交通方案，推进到住宿规划。

    工具内验证 transport_type 是否为合法枚举值。

    参数 (LLM 提供):
    - transport_type: 选择的交通方式 ("flight" / "train" / "driving")
    - transport_options: LLM 查询到的交通方案列表 (含价格), 供预算汇总使用
    """
    assert transport_type in _VALID_TRANSPORT, (
        f"无效交通方式: '{transport_type}'。有效值: {sorted(_VALID_TRANSPORT)}"
    )
    app_logger.info(
        f"[{runtime.tool_call_id}] 交通确认: {transport_type}, {len(transport_options)} 个方案"
    )
    return Command(update={
        "selected_transport": transport_type,
        "transport_options": transport_options,
        "current_step": "accommodation_planning",
        "updated_at": time.time(),
        "messages": [
            ToolMessage(
                content=f"交通方式已确认: {transport_type}, {len(transport_options)} 个方案",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    }, goto="agent")

# 新:
@tool
def select_transport_tool(
    transport_type: TransportType,
    flight_options: list[FlightOption] | None = None,
    train_options: list[TrainOption] | None = None,
    driving_routes: list[DrivingRoute] | None = None,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """
    用户确认交通方式后调用。记录选择和交通方案，推进到住宿规划。

    工具内验证 transport_type 是否为合法枚举值。

    参数 (LLM 提供):
    - transport_type: 选择的交通方式 ("flight" / "train" / "driving")
    - flight_options: 航班方案列表 (可选)
    - train_options: 高铁方案列表 (可选)
    - driving_routes: 自驾路线列表 (可选)
    """
    assert transport_type in _VALID_TRANSPORT, (
        f"无效交通方式: '{transport_type}'。有效值: {sorted(_VALID_TRANSPORT)}"
    )
    total = (len(flight_options or []) + len(train_options or []) +
             len(driving_routes or []))
    app_logger.info(
        f"[{runtime.tool_call_id}] 交通确认: {transport_type}, "
        f"航班{len(flight_options or [])}个/高铁{len(train_options or [])}个/自驾{len(driving_routes or [])}条"
    )
    return Command(update={
        "selected_transport": transport_type,
        "flight_options": flight_options or [],
        "train_options": train_options or [],
        "driving_routes": driving_routes or [],
        "current_step": "accommodation_planning",
        "updated_at": time.time(),
        "messages": [
            ToolMessage(
                content=f"交通方式已确认: {transport_type}, 共 {total} 个方案",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    }, goto="agent")
```

- [ ] **Step 3: 更新 generate_itinerary_tool 中对 transport_options 的引用（第 284 行）**

```python
# 旧:
- 交通方案: {state.get('transport_options', [])}

# 新:
- 航班: {state.get('flight_options', [])}
- 高铁: {state.get('train_options', [])}
- 自驾: {state.get('driving_routes', [])}
```

- [ ] **Step 4: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/tools/state_transition.py', encoding='utf-8').read()); print('OK')"
```

期望输出: `OK`

- [ ] **Step 5: 提交**

```bash
git add app/tools/state_transition.py
git commit -m "refactor: update select_transport_tool to use detailed transport types"
```

---

### Task 4: 运行全量验证

**Files:**
- 无新文件

- [ ] **Step 1: 全项目语法检查**

```bash
python -c "import ast, sys, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.py')]; print('All files OK')"
```

期望输出: `All files OK`

- [ ] **Step 2: 运行现有 Agent 测试**

```bash
python -m pytest tests/test_agents/ -v
```

期望: 3 个现有测试全部通过（无 transport_options 直接引用）。

- [ ] **Step 3: 验证新类型可导入**

```bash
python -c "from app.agents.subagents.transport_state import FlightOption, TrainOption, DrivingRoute, TransportState; print('Import OK')"
```

期望输出: `Import OK`

- [ ] **Step 4: 提交（如有修复）**

```bash
git add -A && git commit -m "chore: fix issues found during verification" || echo "No fixes needed"
```

---

### 变更摘要

| 文件 | 操作 | 说明 |
|---|---|---|
| `app/agents/subagents/transport_state.py` | 新建 | FlightOption, TrainOption, DrivingRoute, TransportState |
| `app/core/state.py` | 修改 | 删除 TransportInfo，新增三个详细字段，更新 STEP_CLEANUP_MAP 和 create_initial_state |
| `app/tools/state_transition.py` | 修改 | 更新导入、select_transport_tool 签名、generate_itinerary_tool 引用 |
