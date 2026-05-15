"""
预算计算工具
从 TravelState 读取各步骤数据，计算完整费用明细
"""
from langchain.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from app.core.state import TravelState
from app.utils.logger import app_logger


@tool
def calculate_budget(
    rooms_needed: int = 1,
    runtime: ToolRuntime[None, TravelState] = None,
) -> str:
    """
    计算完整旅行预算明细。

    从 runtime.state 读取交通/住宿/餐饮/行程数据，自动汇总计算:
    - 交通费
    - 住宿费 (price_per_night × 住宿天数 × 房间数)
    - 餐饮费 (estimated_daily_cost × 旅行天数)
    - 景点门票预估 (每成人每天 100 元，儿童半价)
    - 杂费 (总额的 10%)

    参数:
    - rooms_needed: 所需房间数，默认 1。LLM 根据出行人数和上下文判断

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
    traveler_count = adult_count + children_count
    for t in transport_options:
        price_per_person = t.get("price", 0)
        price_total = price_per_person * traveler_count
        transport_detail.append(
            f"  {t.get('details', '未知交通')}: ¥{price_per_person}/人 × {traveler_count}人 = ¥{price_total}"
        )
    if transport_options:
        totals = [
            t.get("price", 0) * traveler_count
            for t in transport_options if t.get("price")
        ]
        transport_total = round(sum(totals) / len(totals), 2) if totals else 0.0

    # ── 2. 住宿费 ──
    accommodation_total = 0.0
    accommodation_detail = []
    accommodation_options = state.get("accommodation_options", []) or []
    nights = max(travel_days - 1, 1)
    for a in accommodation_options:
        price_per_night = a.get("price_per_night", 0)
        acc_total = price_per_night * nights * rooms_needed
        accommodation_detail.append(
            f"  {a.get('name', '未知住宿')}: ¥{price_per_night}/晚 × {nights}晚 × {rooms_needed}间 = ¥{acc_total}"
        )
    if accommodation_options:
        totals = [
            a.get("price_per_night", 0) * nights * rooms_needed
            for a in accommodation_options if a.get("price_per_night")
        ]
        accommodation_total = round(sum(totals) / len(totals), 2) if totals else 0.0

    # ── 3. 餐饮费 ──
    food_total = 0.0
    food_detail = []
    food_options = state.get("food_options", []) or []
    for f in food_options:
        daily_per_person = f.get("estimated_daily_cost", 0)
        f_total = daily_per_person * travel_days * traveler_count
        food_detail.append(
            f"  {f.get('type', '未知餐饮')}: ¥{daily_per_person}/人/天 × {traveler_count}人 × {travel_days}天 = ¥{f_total}"
        )
    if food_options:
        totals = [
            f.get("estimated_daily_cost", 0) * travel_days * traveler_count
            for f in food_options if f.get("estimated_daily_cost")
        ]
        food_total = round(sum(totals) / len(totals), 2) if totals else 0.0

    # ── 4. 景点门票 ──
    attractions_total = adult_count * travel_days * 100
    if children_count > 0:
        attractions_total += children_count * travel_days * 50

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
        f"> 交通小计 (均价, {traveler_count}人总计): ¥{transport_total}",
        "",
        "### 住宿费",
        *accommodation_detail,
        f"> 住宿小计 (均价): ¥{accommodation_total}",
        "",
        "### 餐饮费",
        *food_detail,
        f"> 餐饮小计 (均价, {traveler_count}人总计): ¥{food_total}",
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

    total_budget_limit = budget_limit * traveler_count
    if budget_limit and grand_total > total_budget_limit:
        over = grand_total - total_budget_limit
        lines.append("")
        lines.append(
            f"⚠️ **超支警告**: 总计 ¥{grand_total} 超出预算上限 ¥{total_budget_limit}"
            f"（人均 ¥{budget_limit} × {traveler_count}人），超出 ¥{over}！"
        )
        lines.append("建议回退调整交通/住宿/餐饮选择。")

    app_logger.info(
        f"[{runtime.tool_call_id}] 预算计算完成: 总计 ¥{grand_total}, "
        f"限额 ¥{budget_limit}"
    )

    return "\n".join(lines)
