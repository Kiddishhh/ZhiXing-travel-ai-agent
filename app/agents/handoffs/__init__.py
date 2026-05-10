"""Handoffs 主流程——单 Agent + 步骤动态切换"""

# create_travel_planner 在 graph.py 创建后自动可用
# from app.agents.handoffs.graph import create_travel_planner

__all__ = ["create_travel_planner"]


def __getattr__(name):
    if name == "create_travel_planner":
        from app.agents.handoffs.graph import create_travel_planner as _fn
        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
