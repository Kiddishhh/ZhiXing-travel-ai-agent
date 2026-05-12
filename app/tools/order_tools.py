"""
订单生成工具
从 TravelState 汇总所有数据，生成最终订单摘要
"""
from langchain.tools import tool
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
    ]

    for t in (state.get('transport_options', []) or []):
        lines.append(f"  {t.get('details', '')}: ¥{t.get('price', 0)}")

    lines.extend([
        "",
        "## 🏨 住宿",
        f"- 类型: {', '.join(accommodation_types)}",
    ])

    for a in (state.get('accommodation_options', []) or []):
        lines.append(f"  {a.get('name', '')}: ¥{a.get('price_per_night', 0)}/晚")

    lines.extend([
        "",
        "## 🍜 餐饮",
        f"- 偏好: {', '.join(food_types)}",
        "",
        "## 📅 每日行程",
    ])

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
