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
    TransportInfo, AccommodationInfo, FoodInfo,
    ItineraryDay,
    UserRequirement, BudgetBreakdown,
)
from app.utils.logger import app_logger

# ── 前进工具 (8 个) ──

@tool
def record_requirement_tool(user_requirement: UserRequirement) -> Command:
    """
    记录用户旅行需求并推进到目的地推荐步骤。

    仅记录需求数据，不计算预算（预算在第 7 步根据实际数据汇总）。

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
def select_transport_tool(
    transport_type: TransportType,
    transport_options: list[TransportInfo],
) -> Command:
    """
    用户确认交通方式后调用。记录选择和交通方案，推进到住宿规划。

    参数:
    - transport_type: 选择的交通方式 ("flight" / "train" / "driving")
    - transport_options: LLM 查询到的交通方案列表（含价格），供预算汇总使用
    """
    app_logger.info(f"交通确认: {transport_type}, {len(transport_options)} 个方案")
    return Command(update={
        "selected_transport": transport_type,
        "transport_options": transport_options,
        "current_step": "accommodation_planning",
        "updated_at": time.time(),
    }, goto="agent")


@tool
def select_accommodation_tool(
    accommodation_types: list[AccommodationType],
    accommodation_options: list[AccommodationInfo],
) -> Command:
    """
    用户确认住宿类型后调用。记录选择和住宿方案，推进到餐饮规划。

    参数:
    - accommodation_types: 选择的住宿类型列表，如 ["star_hotel", "hostel"]
    - accommodation_options: LLM 查询到的住宿方案列表（含价格），供预算汇总使用
    """
    app_logger.info(f"住宿确认: {accommodation_types}, {len(accommodation_options)} 个方案")
    return Command(update={
        "selected_accommodation_types": accommodation_types,
        "accommodation_options": accommodation_options,
        "current_step": "food_planning",
        "updated_at": time.time(),
    }, goto="agent")


@tool
def select_food_tool(
    food_types: list[FoodType],
    food_options: list[FoodInfo],
) -> Command:
    """
    用户确认餐饮类型后调用。记录选择和餐饮方案，推进到行程生成。

    参数:
    - food_types: 选择的餐饮类型列表，如 ["specialty", "local"]
    - food_options: LLM 查询到的餐饮方案列表（含日均花费），供预算汇总使用
    """
    app_logger.info(f"餐饮确认: {food_types}, {len(food_options)} 个方案")
    return Command(update={
        "selected_food_types": food_types,
        "food_options": food_options,
        "current_step": "itinerary_generation",
        "updated_at": time.time(),
    }, goto="agent")


@tool
def generate_itinerary_tool(itinerary: list[ItineraryDay]) -> Command:
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
    预算汇总完成后调用。记录完整预算明细并推进到订单生成。

    LLM 应根据前面步骤收集的实际数据计算预算:
    - transport: 从 transport_options 中提取交通费用
    - accommodation: 从 accommodation_options 中提取住宿费用 (price_per_night * nights)
    - food: 从 food_options 中提取日均餐饮费用
    - attractions: 景点门票预估
    - misc: 杂费预留 (通常为总额的 10%)
    - total: 以上各项之和

    参数:
    - budget: 完整的 BudgetBreakdown 对象
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
