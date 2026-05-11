# 交通规划 Subagent State 层设计

## 背景

现有 `TransportInfo`（`app/core/state.py`）只有 5 个通用字段，无法精确表达航班、高铁、自驾三种交通方式的差异信息。需要在 `app/agents/subagents/` 下构建交通规划子代理，首先设计其 State 类型层。

## 设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 文件位置 | `app/agents/subagents/transport_state.py` | 遵循 Router 模式，subagent 内部状态独立于全局 core/state.py |
| 旧类型处理 | 替换 `TransportInfo` | 详细类型让前端和工具调用有明确的字段契约 |
| TravelState 字段 | 三个独立字段 + add reducer | 类型安全，并行 Send 累加，前端渲染无需判断类型 |
| Subagent 架构 | ReAct Agent | LLM 自主决定调用哪些交通查询工具 |

## 类型定义

### FlightOption

```python
class FlightOption(TypedDict):
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
```

### TrainOption

```python
class TrainOption(TypedDict):
    train_number: str                 # 车次号
    departure_station: str            # 出发站
    arrival_station: str              # 到达站
    departure_time: str               # 发车时间
    arrival_time: str                 # 到站时间
    duration: str                     # 运行时长
    seat_types: list[str]             # 可选座位类型
    prices: dict[str, float]          # 各座位类型价格
    available: bool                   # 是否有票
```

### DrivingRoute

```python
class DrivingRoute(TypedDict):
    route_name: str          # 路线名称
    distance: str            # 总距离
    duration: str            # 预计时长
    toll_fee: float          # 过路费
    fuel_cost: float         # 油费估算
    steps: list[str]         # 导航步骤
    waypoints: list[str]     # 途经城市
```

### TransportState（子图内部状态）

```python
class TransportState(TypedDict):
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

## TravelState 变更

`app/core/state.py` 中：

1. **删除**: `TransportInfo` TypedDict（被三个详细类型替换）
2. **删除**: `transport_options: NotRequired[List[TransportInfo]]`
3. **新增**:
```python
flight_options: NotRequired[Annotated[list[FlightOption], add]]
train_options: NotRequired[Annotated[list[TrainOption], add]]
driving_routes: NotRequired[Annotated[list[DrivingRoute], add]]
```
4. 导入三个新类型，`from app.agents.subagents.transport_state import FlightOption, TrainOption, DrivingRoute`

## 受影响文件

| 文件 | 变更 |
|---|---|
| `app/agents/subagents/transport_state.py` | **新建** — 三个子结构 + TransportState |
| `app/core/state.py` | **修改** — 删除 TransportInfo，替换 transport_options 为三个独立字段，新增导入 |
| `app/tools/state_transition.py` | **修改** — `select_transport_tool` 签名从 `list[TransportInfo]` 改为适配新类型 |
| `app/tools/transport_tools.py` | **修改** — 工具返回类型从 `str` 改为结构化 TypedDict |
| `docs/superpowers/specs/` | **新建** — 本文档 |

## 未决项（下一阶段确定）

- ReAct Agent 模式下 `TransportState` 是否需要继承 `MessagesState` 以支持对话消息
- `select_transport_tool` 的新签名（如何承载三种类型的选择）
- `STEP_CLEANUP_MAP` 中 `transport_planning` 对应的清理字段更新
