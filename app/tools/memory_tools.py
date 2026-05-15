"""
用户长期记忆写入工具

save_user_preference — 用户显式声明偏好时由 Agent 调用
auto_save_from_state — 订单生成完成后从 TravelState 自动提取画像
"""
from langchain.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from app.core.state import TravelState
from app.core.memory_store import get_memory_store_manager
from app.utils.logger import app_logger


PREFERENCE_TYPE_MAP = {
    "transport": "preferred_transport",
    "food": "dietary_preferences",
    "budget": "budget_level",
    "style": "travel_styles",
    "custom": "extensions",
}


@tool
async def save_user_preference(
    preference_type: str,
    value: str,
    runtime: ToolRuntime[None, TravelState] = None,
) -> str:
    """
    保存用户的长期偏好到记忆库。

    参数说明:
    - preference_type: 偏好类型。可选值:
      "transport" — 交通偏好 (如 "高铁")
      "food" — 饮食偏好 (如 "川菜")
      "budget" — 预算偏好 (如 "舒适")
      "style" — 旅行风格 (如 "文化")
      "custom" — 自定义偏好 (如 "需要无障碍设施")
    - value: 偏好内容

    返回确认信息。
    """
    user_id = runtime.state.get("user_id", "unknown") if runtime else "unknown"
    app_logger.info(f"保存用户偏好: user_id={user_id}, type={preference_type}, value={value}")

    col = PREFERENCE_TYPE_MAP.get(preference_type)
    if col is None:
        return f"未知的偏好类型: {preference_type}，可选: {', '.join(PREFERENCE_TYPE_MAP.keys())}"

    try:
        manager = await get_memory_store_manager()

        if preference_type in ("food", "style"):
            # 数组字段：包装为列表后合并
            fields = {col: [value]}
        elif preference_type == "custom":
            # 扩展字段
            fields = {"extensions": {preference_type: value}}
        else:
            # 标量字段
            fields = {col: value}

        await manager.upsert_profile(user_id, fields)
        return f"已保存您的{preference_type}偏好: {value}"
    except Exception as e:
        app_logger.warning(f"保存偏好失败: {e}")
        return f"保存偏好时遇到问题，请稍后重试。"


@tool
async def auto_save_from_state(
    runtime: ToolRuntime[None, TravelState] = None,
) -> str:
    """
    从当前旅行规划状态中自动提取并保存用户偏好画像。

    在订单生成完成后调用，将本次行程的偏好和历史写入长期记忆。
    无需参数，自动从当前状态中提取。

    返回保存摘要。
    """
    if runtime is None:
        return "无法读取旅行状态，画像未保存。"

    state = runtime.state
    user_id = state.get("user_id", "unknown")
    req = state.get("user_requirement", {}) or {}

    app_logger.info(f"自动保存用户画像: user_id={user_id}")

    try:
        # 从 state 提取字段
        fields = {}

        # budget_level
        budget_level = req.get("budget_level")
        if budget_level:
            fields["budget_level"] = budget_level

        # travel_styles
        travel_styles = req.get("travel_styles", []) or []
        if travel_styles:
            fields["travel_styles"] = list(travel_styles)

        # preferred_transport
        selected_transport = state.get("selected_transport")
        if selected_transport:
            fields["preferred_transport"] = selected_transport

        # favorite_destinations
        selected_destination = state.get("selected_destination")
        if selected_destination:
            fields["favorite_destinations"] = [selected_destination]

        # dietary_preferences（从 food_options 推断）
        food_options = state.get("food_options", []) or []
        food_types = [f.get("type", "") for f in food_options if f.get("type")]
        if food_types:
            fields["dietary_preferences"] = food_types

        # 统计字段
        travel_days = req.get("travel_days", 0)
        departure_date = req.get("departure_date")
        if travel_days:
            fields["total_trips"] = 1  # upsert 时会与旧值累加
        if selected_destination:
            fields["last_destination"] = selected_destination
        if departure_date:
            fields["last_travel_date"] = departure_date

        manager = await get_memory_store_manager()
        result = await manager.upsert_profile(user_id, fields)

        if result:
            total_trips = result.get("total_trips", 0)
            last_dest = result.get("last_destination", "未知")
            return (
                f"已更新您的旅行画像：共 {total_trips} 次出行，"
                f"最近目的地 {last_dest}。"
            )
        return "画像保存完成。"
    except Exception as e:
        app_logger.warning(f"自动保存画像失败: {e}")
        return "自动保存画像时遇到问题，不影响订单生成。"
