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

__all__ = [
    "TOOL_REGISTRY",
    "register_tool",
    "query_destination_info",
]
