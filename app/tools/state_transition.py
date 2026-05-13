"""
流程控制工具集

包含: 8 个前进工具 + 1 个通用回退工具 + 7 个快捷回退工具 + 1 个进度查询工具。

runtime 参数 (ToolRuntime) 由 ToolNode 自动注入, 无需 LLM 提供:
- runtime.state: 当前 TravelState (只读), 用于验证和读取上下文
- runtime.tool_call_id: 本次工具调用的唯一 ID
"""
import time

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command

from app.core.state import (
    TravelState,
    PlanningStep, ALLOWED_BACK_STEPS, STEP_CLEANUP_MAP,
    TransportType, AccommodationType, FoodType,
    TransportInfo, AccommodationInfo, FoodInfo,
    ItineraryDay,
    UserRequirement, BudgetBreakdown, BudgetLevel,
)
from app.config import settings
from app.utils.logger import app_logger

# ── 枚举值集合 (用于运行时验证) ──

_VALID_TRANSPORT = {"flight", "train", "driving"}
_VALID_ACCOMMODATION = {"star_hotel", "economy_hotel", "hostel", "youth_hostel"}
_VALID_FOOD = {"specialty", "chain", "local"}

# ── 前进工具 (8 个) ──

@tool
def record_requirement_tool(
    user_requirement: UserRequirement,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """
    记录用户旅行需求并推进到目的地推荐步骤。

    工具内自动计算:
    - 平均预算: (budget_min + budget_max) / 2
    - 预算等级: economy (<3000) / comfort (3000-8000) / luxury (>8000)
    结果回写到 user_requirement 中。

    参数 (LLM 提供):
    - user_requirement: 用户需求对象
      包含: departure_city, destination, departure_date, travel_days,
            adult_count, children_count, budget_min, budget_max,
            budget_level, travel_styles, special_needs
    """
    # 计算平均预算
    budget_min = user_requirement.get("budget_min", 0) or 0
    budget_max = user_requirement.get("budget_max", 0) or 0
    if budget_min > 0 and budget_max > 0:
        avg_budget = (budget_min + budget_max) / 2
    elif budget_max > 0:
        avg_budget = budget_max
    else:
        avg_budget = budget_min or 3000

    # 推断预算等级
    if avg_budget < 3000:
        budget_level: BudgetLevel = "economy"
    elif avg_budget <= 8000:
        budget_level = "comfort"
    else:
        budget_level = "luxury"

    # 确保预算字段完整
    user_requirement["budget_min"] = budget_min
    user_requirement["budget_max"] = budget_max
    user_requirement["budget_level"] = budget_level

    app_logger.info(
        f"[{runtime.tool_call_id}] 需求记录完成: "
        f"目的地={user_requirement.get('destination')}, "
        f"平均预算={avg_budget}, 等级={budget_level}"
    )

    return Command(update={
        "user_requirement": user_requirement,
        "current_step": "destination_recommendation",
        "updated_at": time.time(),
        "messages": [
            ToolMessage(
                content=f"需求已记录: 目的地 {user_requirement.get('destination')}, "
                        f"{user_requirement.get('travel_days')}天, 预算等级 {budget_level}",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })


@tool
def select_destination_tool(
    destination: str,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """
    用户确认目的地后调用。记录选择并推进到交通规划。

    参数 (LLM 提供):
    - destination: 用户选择的目的地名称, 如 "西安"
    """
    app_logger.info(f"[{runtime.tool_call_id}] 目的地确认: {destination}")
    return Command(update={
        "selected_destination": destination,
        "current_step": "transport_planning",
        "updated_at": time.time(),
        "messages": [
            ToolMessage(
                content=f"目的地已确认: {destination}",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })


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
    })


@tool
def select_accommodation_tool(
    accommodation_types: list[AccommodationType],
    accommodation_options: list[AccommodationInfo],
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """
    用户确认住宿类型后调用。记录选择和住宿方案，推进到餐饮规划。

    工具内验证每个 accommodation_type 是否为合法枚举值。

    参数 (LLM 提供):
    - accommodation_types: 选择的住宿类型列表, 如 ["star_hotel", "hostel"]
    - accommodation_options: LLM 查询到的住宿方案列表 (含价格), 供预算汇总使用
    """
    for at in accommodation_types:
        assert at in _VALID_ACCOMMODATION, (
            f"无效住宿类型: '{at}'。有效值: {sorted(_VALID_ACCOMMODATION)}"
        )
    app_logger.info(
        f"[{runtime.tool_call_id}] 住宿确认: {accommodation_types}, {len(accommodation_options)} 个方案"
    )
    return Command(update={
        "selected_accommodation_types": accommodation_types,
        "accommodation_options": accommodation_options,
        "current_step": "food_planning",
        "updated_at": time.time(),
        "messages": [
            ToolMessage(
                content=f"住宿已确认: {', '.join(accommodation_types)}, {len(accommodation_options)} 个方案",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })


@tool
def select_food_tool(
    food_types: list[FoodType],
    food_options: list[FoodInfo],
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """
    用户确认餐饮类型后调用。记录选择和餐饮方案，推进到行程生成。

    工具内验证每个 food_type 是否为合法枚举值。

    参数 (LLM 提供):
    - food_types: 选择的餐饮类型列表, 如 ["specialty", "local"]
    - food_options: LLM 查询到的餐饮方案列表 (含日均花费), 供预算汇总使用
    """
    for ft in food_types:
        assert ft in _VALID_FOOD, (
            f"无效餐饮类型: '{ft}'。有效值: {sorted(_VALID_FOOD)}"
        )
    app_logger.info(
        f"[{runtime.tool_call_id}] 餐饮确认: {food_types}, {len(food_options)} 个方案"
    )
    return Command(update={
        "selected_food_types": food_types,
        "food_options": food_options,
        "current_step": "itinerary_generation",
        "updated_at": time.time(),
        "messages": [
            ToolMessage(
                content=f"餐饮已确认: {', '.join(food_types)}, {len(food_options)} 个方案",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })


@tool
async def generate_itinerary_tool(
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """
    调用 LLM 生成完整每日行程, 记录到 itinerary 并推进到预算汇总。

    此工具无 LLM 提供参数——所有信息通过 runtime.state 读取:
    - 目的地、天数、交通方式、住宿类型、餐饮类型来自 State
    - 验证上述必要信息完整后, 调用 LLM 生成每日计划

    不需要 LLM 在调用时传入任何参数。
    """
    state = runtime.state
    current_step = state.get("current_step", "?")

    # ── 验证必要信息是否完整 ──
    destination = state.get("selected_destination")
    transport = state.get("selected_transport")
    accommodation = state.get("selected_accommodation_types")
    food = state.get("selected_food_types")
    user_req = state.get("user_requirement", {})

    assert destination, "缺少目的地信息 (selected_destination)"
    assert transport, "缺少交通方式 (selected_transport)"
    assert accommodation, "缺少住宿类型 (selected_accommodation_types)"
    assert food, "缺少餐饮类型 (selected_food_types)"
    assert user_req, "缺少用户需求 (user_requirement)"

    travel_days = user_req.get("travel_days", 1)
    departure_date = user_req.get("departure_date", "未指定")
    adult_count = user_req.get("adult_count", 1)
    children_count = user_req.get("children_count", 0)

    app_logger.info(
        f"[{runtime.tool_call_id}] 开始生成行程: "
        f"{destination} {travel_days}天, "
        f"交通={transport}, 住宿={accommodation}, 餐饮={food}"
    )

    # ── 调用 LLM 生成每日行程 ──
    itinerary_prompt = f"""你是专业的旅行行程规划师。请为以下旅行生成每日详细行程。

## 旅行信息
- 目的地: {destination}
- 出发日期: {departure_date}
- 出行天数: {travel_days} 天
- 人数: {adult_count} 成人{f' + {children_count} 儿童' if children_count else ''}
- 交通方式: {transport}
- 住宿类型: {', '.join(accommodation)}
- 餐饮偏好: {', '.join(food)}

## 已有数据参考
- 交通方案: {state.get('transport_options', [])}
- 住宿方案: {state.get('accommodation_options', [])}
- 餐饮方案: {state.get('food_options', [])}

## 输出要求
为每一天生成一个 JSON 对象, 格式如下:
```json
[
  {{
    "day_number": 1,
    "date": "YYYY-MM-DD",
    "activities": ["上午: ...", "下午: ...", "晚上: ..."],
    "meals": ["早餐: ...", "午餐: ...", "晚餐: ..."],
    "accommodation": "住宿名称",
    "transport": "当日交通方式"
  }}
]
```

严格按照 {travel_days} 天生成, 每一天都要有具体的景点、餐饮和住宿安排。
直接返回 JSON 数组, 不要额外说明。"""

    llm = ChatTongyi(
        model=settings.qwen_model_name,
        api_key=settings.dashscope_api_key,
        temperature=0.7,
    )
    response = await llm.ainvoke(itinerary_prompt)

    # 提取 JSON 内容
    import json
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
    itinerary: list[ItineraryDay] = json.loads(content)

    app_logger.info(
        f"[{runtime.tool_call_id}] 行程生成完成: {len(itinerary)} 天"
    )

    return Command(update={
        "itinerary": itinerary,
        "current_step": "budget_summarization",
        "updated_at": time.time(),
        "messages": [
            ToolMessage(
                content=f"行程已生成: {len(itinerary)} 天行程 (目的地: {destination})",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })


@tool
def summarize_budget_tool(
    budget: BudgetBreakdown,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """
    预算汇总完成后调用。记录完整预算明细并推进到订单生成。

    工具内通过 runtime.state 读取人数和天数进行预算验证:
    - 从 user_requirement 提取 adult_count, children_count, travel_days
    - 计算人均每日花费 = total / ((adult + children) * days)
    - 超预算时发出警告日志

    参数 (LLM 提供):
    - budget: 完整的 BudgetBreakdown 对象
      LLM 应根据前面步骤收集的实际数据:
      - transport: 从 transport_options 中提取交通费用
      - accommodation: 从 accommodation_options 中提取 (price_per_night * nights)
      - food: 从 food_options 中提取 (estimated_daily_cost * days)
      - attractions: 景点门票预估
      - misc: 杂费 (通常为总额的 10%)
      - total: 以上各项之和
    """
    state = runtime.state
    req = state.get("user_requirement", {})
    budget_limit = req.get("budget_max", 0) or 0
    travel_days = req.get("travel_days", 1)
    adult_count = req.get("adult_count", 1)
    children_count = req.get("children_count", 0)
    total_people = adult_count + children_count
    total = budget.get("total", 0)

    # 计算人均每日花费
    if total_people > 0 and travel_days > 0 and total > 0:
        per_person_per_day = total / (total_people * travel_days)
    else:
        per_person_per_day = 0

    if budget_limit and total > budget_limit:
        app_logger.warning(
            f"[{runtime.tool_call_id}] 预算超支: "
            f"总计 {total} > 限额 {budget_limit}, "
            f"人均每日 {per_person_per_day:.0f} 元 "
            f"({total_people}人 × {travel_days}天)"
        )
    else:
        app_logger.info(
            f"[{runtime.tool_call_id}] 预算汇总完成: "
            f"总计 {total}, 人均每日 {per_person_per_day:.0f} 元, "
            f"限额 {budget_limit}, {total_people}人 × {travel_days}天"
        )

    return Command(update={
        "budget": budget,
        "current_step": "order_generation",
        "updated_at": time.time(),
        "messages": [
            ToolMessage(
                content=f"预算汇总完成: 总计 {total} 元, "
                        f"人均每日 {per_person_per_day:.0f} 元 ({total_people}人 × {travel_days}天)",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })


@tool
def generate_order_tool(
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """
    用户确认下单后调用。生成订单并结束旅行规划流程。

    通过 runtime.state 读取完整的旅行计划数据, 生成最终报告和订单号。
    返回 goto="__end__" 终止图运行。
    """
    import uuid
    state = runtime.state
    dest = state.get("selected_destination", "目的地")
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    days = state.get("user_requirement", {}).get("travel_days", 0)
    budget_total = state.get("budget", {}).get("total", 0)

    app_logger.info(
        f"[{runtime.tool_call_id}] 订单生成: {order_id} "
        f"(目的地: {dest}, {days}天, 总预算: {budget_total})"
    )

    report_msg = f"旅行规划已完成, 订单 {order_id} 已生成。感谢使用知行智能旅游规划助手!"
    return Command(update={
        "order_id": order_id,
        "report": report_msg,
        "updated_at": time.time(),
        "messages": [
            ToolMessage(
                content=f"订单已生成: {order_id} (目的地: {dest}, {days}天, 预算: {budget_total}元)",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    }, goto="__end__")


# ── 通用回退工具 (1 个) ──

@tool
def go_back_to_step(
    target_step: PlanningStep,
    reason: str,
    clear_subsequent_data: bool = True,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """
    回退到指定的流程步骤。

    此工具将:
    1. 验证 target_step 是否为有效的 PlanningStep
    2. 拦截非法回退 (如回退到 order_generation)
    3. 根据 clear_subsequent_data 决定是否清除后续数据
    4. 更新 current_step 到目标步骤
    5. 记录回退日志 (含 tool_call_id 和回退原因)

    参数 (LLM 提供):
    - target_step: 要回退到的目标步骤名称
      可选值: "requirement_collection", "destination_recommendation",
              "transport_planning", "accommodation_planning",
              "food_planning", "itinerary_generation",
              "budget_summarization"
    - reason: 回退原因, 如 "用户想更换目的地"
    - clear_subsequent_data: 是否清除目标步骤之后的数据 (默认 True)
    """
    state = runtime.state
    current = state.get("current_step", "?")

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

    # ── 3. 日志记录 ──
    app_logger.warning(
        f"[{runtime.tool_call_id}] 回退操作: {current} → {target_step}, "
        f"原因: {reason}, 清除数据: {clear_subsequent_data}"
    )

    # ── 4. 清除数据 (如需要) ──
    update: dict = {"current_step": target_step, "updated_at": time.time()}
    if clear_subsequent_data:
        cleanup_fields = STEP_CLEANUP_MAP.get(target_step, [])
        for field in cleanup_fields:
            update[field] = None
        app_logger.info(f"[{runtime.tool_call_id}] 已清除字段: {cleanup_fields}")

    # ── 5. 追加 ToolMessage ──
    clear_note = "已清除后续数据" if clear_subsequent_data else "保留已有数据"
    update["messages"] = [
        ToolMessage(
            content=f"回退: {current} → {target_step} (原因: {reason}, {clear_note})",
            tool_call_id=runtime.tool_call_id,
        )
    ]

    # ── 6. 返回 Command ──
    return Command(update=update)


# ── 快捷回退工具 (7 个) ──
# LLM 可根据需要直接调用, 无需指定 target_step

def _build_back_command(
    step: PlanningStep, reason: str,
    clear_subsequent_data: bool,
    runtime: ToolRuntime,
) -> Command:
    """复用 go_back_to_step 的核心逻辑"""
    return go_back_to_step.func(
        target_step=step,
        reason=reason,
        clear_subsequent_data=clear_subsequent_data,
        runtime=runtime,
    )


@tool
def go_back_to_requirement(
    reason: str,
    clear_subsequent_data: bool = True,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """回退到需求收集步骤, 重新规划整个旅行"""
    return _build_back_command(
        "requirement_collection", reason, clear_subsequent_data, runtime
    )


@tool
def go_back_to_destination(
    reason: str,
    clear_subsequent_data: bool = True,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """回退到目的地推荐步骤, 重新选择目的地"""
    return _build_back_command(
        "destination_recommendation", reason, clear_subsequent_data, runtime
    )


@tool
def go_back_to_transport(
    reason: str,
    clear_subsequent_data: bool = True,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """回退到交通规划步骤, 重新选择交通方式"""
    return _build_back_command(
        "transport_planning", reason, clear_subsequent_data, runtime
    )


@tool
def go_back_to_accommodation(
    reason: str,
    clear_subsequent_data: bool = True,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """回退到住宿规划步骤, 重新选择住宿"""
    return _build_back_command(
        "accommodation_planning", reason, clear_subsequent_data, runtime
    )


@tool
def go_back_to_food(
    reason: str,
    clear_subsequent_data: bool = True,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """回退到餐饮规划步骤, 重新选择餐饮"""
    return _build_back_command(
        "food_planning", reason, clear_subsequent_data, runtime
    )


@tool
def go_back_to_itinerary(
    reason: str,
    clear_subsequent_data: bool = True,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """回退到行程生成步骤, 重新生成行程"""
    return _build_back_command(
        "itinerary_generation", reason, clear_subsequent_data, runtime
    )


@tool
def go_back_to_budget(
    reason: str,
    clear_subsequent_data: bool = True,
    runtime: ToolRuntime[None, TravelState] = None,
) -> Command:
    """回退到预算汇总步骤, 重新计算预算"""
    return _build_back_command(
        "budget_summarization", reason, clear_subsequent_data, runtime
    )


# ── 辅助工具 ──

@tool
def check_current_progress(
    runtime: ToolRuntime[None, TravelState] = None,
) -> str:
    """
    查看当前旅行规划的完成进度。

    通过 runtime.state 读取当前步骤和已收集的数据, 返回进度摘要。
    不修改任何状态。
    """
    state = runtime.state
    current_step = state.get("current_step", "?")
    has_requirement = state.get("user_requirement") is not None
    has_destination = state.get("selected_destination") is not None
    has_transport = state.get("selected_transport") is not None
    has_accommodation = state.get("selected_accommodation_types") is not None
    has_food = state.get("selected_food_types") is not None
    has_itinerary = state.get("itinerary") is not None
    has_budget = state.get("budget") is not None

    completed = sum([
        has_requirement, has_destination, has_transport,
        has_accommodation, has_food, has_itinerary, has_budget,
    ])

    return (
        f"当前步骤: {current_step}\n"
        f"已完成 {completed}/7 个步骤\n"
        f"需求记录: {'OK' if has_requirement else '--'}\n"
        f"目的地: {'OK' if has_destination else '--'}\n"
        f"交通: {'OK' if has_transport else '--'}\n"
        f"住宿: {'OK' if has_accommodation else '--'}\n"
        f"餐饮: {'OK' if has_food else '--'}\n"
        f"行程: {'OK' if has_itinerary else '--'}\n"
        f"预算: {'OK' if has_budget else '--'}"
    )
