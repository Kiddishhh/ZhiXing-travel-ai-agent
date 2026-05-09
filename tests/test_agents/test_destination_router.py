"""Destination Router 单元测试"""
from app.agents.routers.destination_router import (
    Classification,
    AgentOutput,
    DestinationRouterState,
    ClassificationResult,
    route_to_agents,
    _explore_agent,
    _weather_agent,
    build_router_graph,
)


def test_classification_result_model():
    """验证 ClassificationResult Pydantic 模型"""
    data = {
        "classifications": [
            {"agent": "explore", "query": "北京景点推荐"},
            {"agent": "weather", "query": "北京天气"},
        ]
    }
    result = ClassificationResult(**data)
    assert len(result.classifications) == 2
    assert result.classifications[0]["agent"] == "explore"
    assert result.classifications[1]["query"] == "北京天气"


def test_agent_output_typeddict():
    """验证 AgentOutput TypedDict 结构"""
    output: AgentOutput = {"agent_name": "explore", "result": "测试结果"}
    assert output["agent_name"] == "explore"
    assert output["result"] == "测试结果"


def test_weather_agent_placeholder():
    """验证天气 Agent 返回占位结果"""
    result = _weather_agent("北京天气如何")
    assert result == "天气功能待实现"


def test_route_to_agents_returns_send_list():
    """验证 route_to_agents 返回 Send 列表"""
    state: DestinationRouterState = {
        "original_query": "北京旅游",
        "destination": "北京",
        "classifications": [
            {"agent": "explore", "query": "北京景点"},
            {"agent": "weather", "query": "北京天气"},
        ],
        "agent_results": [],
        "final_report": "",
    }
    sends = route_to_agents(state)
    assert len(sends) == 2
    for send in sends:
        assert send.node == "agent_node"


def test_build_router_graph():
    """验证图编译成功"""
    graph = build_router_graph()
    assert graph is not None
    nodes = list(graph.nodes.keys())
    assert "classifier_node" in nodes
    assert "agent_node" in nodes
