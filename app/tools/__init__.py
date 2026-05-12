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
    go_back_to_requirement,
    go_back_to_destination,
    go_back_to_transport,
    go_back_to_accommodation,
    go_back_to_food,
    go_back_to_itinerary,
    go_back_to_budget,
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
register_tool("go_back_to_requirement", go_back_to_requirement)
register_tool("go_back_to_destination", go_back_to_destination)
register_tool("go_back_to_transport", go_back_to_transport)
register_tool("go_back_to_accommodation", go_back_to_accommodation)
register_tool("go_back_to_food", go_back_to_food)
register_tool("go_back_to_itinerary", go_back_to_itinerary)
register_tool("go_back_to_budget", go_back_to_budget)
register_tool("check_current_progress", check_current_progress)

# ── 注册业务工具 ──
from .transport_tools import query_transport_options
from .accommodation_tools import query_accommodation
from .food_tools import query_food
from .budget_tools import calculate_budget
from .order_tools import create_order

register_tool("query_transport_options", query_transport_options)
register_tool("query_accommodation", query_accommodation)
register_tool("query_food", query_food)
register_tool("calculate_budget", calculate_budget)
register_tool("create_order", create_order)

__all__ = [
    "TOOL_REGISTRY",
    "register_tool",
    "query_destination_info",
]
