# Destination Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Router 模式工作流，通过 LLM 分类器将查询并行分发给探索 Agent（RAG 检索）和天气 Agent（占位），汇总结果。

**Architecture:** LangGraph StateGraph 流程：classifier_node → route_to_agents（Send 并行）→ agent_node（单节点分发 explore/weather）。State 使用 TypedDict + Annotated reducer 实现 agent_results 累加。

**Tech Stack:** langgraph (1.0.5), ChatTongyi (qwen-max), ChromaManager, Pydantic 2

---

### Task 1: 创建 destination_router.py

**文件:**
- Create: `app/agents/routers/destination_router.py`

- [ ] **Step 1.1: 创建 destination_router.py**

```python
"""
目的地查询 Router 工作流

使用 LLM 分类器分析查询意图，通过 LangGraph Send 并行分发给
探索 Agent（RAG 检索）和天气 Agent。
"""
from operator import add
from typing import Annotated, Literal, List, TypedDict

from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.config import settings
from app.core.ChromaDB.chroma_client import ChromaManager
from app.utils.logger import app_logger


# ── Agent 枚举 ──────────────────────────────────────────

AgentType = Literal["explore", "weather"]


# ── State 定义 ──────────────────────────────────────────

class Classification(TypedDict):
    """分类结果"""
    agent: AgentType          # 要调用的 Agent
    query: str                # 子查询


class AgentOutput(TypedDict):
    """Agent 输出"""
    agent_name: str
    result: str


class DestinationRouterState(TypedDict):
    """Router 状态"""
    original_query: str                                   # 原始查询
    destination: str                                      # 目的地名称
    classifications: List[Classification]                 # 分类结果
    agent_results: Annotated[List[AgentOutput], add]      # Agent 结果（累加）
    final_report: str                                     # 综合报告


# ── 分类器 ──────────────────────────────────────────────

class ClassificationResult(BaseModel):
    """分类结果（LLM 结构化输出）"""
    classifications: List[Classification] = Field(
        description="要调用的 Agent 列表及其子查询"
    )


_CLASSIFIER_PROMPT = (
    "你是一个旅游规划查询分类器。分析用户关于目的地的查询，判断需要调用哪些 Agent 来获取信息。\n\n"
    "可用 Agent：\n"
    "- explore：景点、攻略、游玩建议相关\n"
    "- weather：天气、气候、季节相关\n\n"
    "规则：\n"
    "1. 为每个需要调用的 Agent 生成一个子查询\n"
    "2. 子查询应包含目的地名称，便于 Agent 检索\n"
    "3. 至少返回一个 Agent（默认为 explore）\n\n"
    "用户查询：{query}\n"
    "目的地：{destination}"
)


def classifier_node(state: DestinationRouterState) -> dict:
    """LLM 分类器节点：分析查询意图，决定调用哪些 Agent"""
    query = state["original_query"]
    destination = state["destination"]

    llm = ChatTongyi(
        model="qwen-max",
        temperature=0.0,
        api_key=settings.dashscope_api_key,
    )
    structured_llm = llm.with_structured_output(ClassificationResult)

    prompt = _CLASSIFIER_PROMPT.format(query=query, destination=destination)
    result: ClassificationResult = structured_llm.invoke(prompt)

    app_logger.info(
        f"分类结果: {[(c['agent'], c['query']) for c in result.classifications]}"
    )
    return {"classifications": result.classifications}


# ── 路由 ────────────────────────────────────────────────

def route_to_agents(state: DestinationRouterState) -> list[Send]:
    """路由函数：根据分类结果并行派发任务给 Agent"""
    sends = []
    for cls in state["classifications"]:
        sends.append(
            Send(
                "agent_node",
                {
                    "classifications": [cls],
                    "original_query": state["original_query"],
                    "destination": state["destination"],
                    "agent_results": [],
                    "final_report": "",
                },
            )
        )
    return sends


# ── Agent 节点 ──────────────────────────────────────────

def agent_node(state: DestinationRouterState) -> dict:
    """Agent 执行节点：根据分类分发到 explore 或 weather"""
    cls = state["classifications"][0]

    if cls["agent"] == "explore":
        result = _explore_agent(cls["query"])
    elif cls["agent"] == "weather":
        result = _weather_agent(cls["query"])
    else:
        result = f"未知 Agent 类型: {cls['agent']}"

    return {"agent_results": [AgentOutput(agent_name=cls["agent"], result=result)]}


def _explore_agent(query: str) -> str:
    """探索 Agent：从 RAG 检索景点攻略"""
    try:
        chroma_manager = ChromaManager()
        docs = chroma_manager.similarity_search_with_score(query, k=5)

        if not docs:
            return f"未找到与「{query}」相关的攻略信息。"

        lines = [f"## 相关攻略 ({len(docs)} 条)\n"]
        for i, (doc, score) in enumerate(docs, 1):
            snippet = doc.page_content[:200].replace("\n", " ")
            source = doc.metadata.get("source", "未知来源")
            lines.append(f"{i}. [{source}] {snippet}...")
        return "\n\n".join(lines)
    except Exception as e:
        app_logger.error(f"探索 Agent 检索失败: {e}")
        return f"攻略检索异常: {e}"


def _weather_agent(query: str) -> str:
    """天气 Agent：占位，待接入天气 API"""
    app_logger.info(f"天气 Agent 收到查询: '{query}'（当前为占位实现）")
    return "天气功能待实现"


# ── 图构建 ──────────────────────────────────────────────

def build_router_graph() -> StateGraph:
    """构建并编译 Router 工作流图"""
    builder = StateGraph(DestinationRouterState)

    builder.add_node("classifier_node", classifier_node)
    builder.add_node("agent_node", agent_node)

    builder.add_edge(START, "classifier_node")
    builder.add_conditional_edges(
        "classifier_node",
        route_to_agents,
    )
    builder.add_edge("agent_node", END)

    return builder.compile()
```

- [ ] **Step 1.2: 验证模块导入**

Run:
```bash
cd "d:/AI agent/知行智能旅游规划助手"
python -c "from app.agents.routers.destination_router import build_router_graph; print('destination_router imported OK')"
```

Expected: `destination_router imported OK`

---

### Task 2: 编写单元测试

**文件:**
- Create: `tests/test_agents/test_destination_router.py`

- [ ] **Step 2.1: 创建测试文件**

```python
"""Destination Router 单元测试"""
from app.agents.routers.destination_router import (
    Classification,
    AgentOutput,
    DestinationRouterState,
    ClassificationResult,
    classifier_node,
    route_to_agents,
    agent_node,
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
    # 验证 Send 目标节点
    for send in sends:
        assert send.node == "agent_node"


def test_build_router_graph():
    """验证图编译成功"""
    graph = build_router_graph()
    assert graph is not None
    # 验证图中包含预期节点
    nodes = list(graph.nodes.keys())
    assert "classifier_node" in nodes
    assert "agent_node" in nodes
```

- [ ] **Step 2.2: 运行测试**

Run:
```bash
cd "d:/AI agent/知行智能旅游规划助手"
python -m pytest tests/test_agents/test_destination_router.py -v
```

Expected: 5 passed

---

### Task 3: 端到端验证

- [ ] **Step 3.1: 验证完整导入链**

Run:
```bash
cd "d:/AI agent/知行智能旅游规划助手"
python -c "
from app.agents.routers.destination_router import build_router_graph
from app.core.ChromaDB.chroma_client import ChromaManager
from app.utils.logger import app_logger
print('All dependencies imported OK')

# 验证图编译
graph = build_router_graph()
print(f'Graph compiled OK, nodes: {list(graph.nodes.keys())}')
"
```

Expected: 无导入错误，图编译成功

- [ ] **Step 3.2: AST 语法检查**

Run:
```bash
cd "d:/AI agent/知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/agents/routers/destination_router.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

---

## 设计决策

1. **TypedDict + Annotated**: 使用 `TypedDict` 而非 `BaseModel` 定义 State，配合 `Annotated[List, add]` 实现 agent_results 自然累加，避免自定义 reducer。

2. **Send 并行**: `route_to_agents` 返回 `Send("agent_node", ...)` 列表，LangGraph 自动并行执行多个 agent_node，结果通过 `add` reducer 合并到 `agent_results`。

3. **单节点分发**: explore 和 weather 共享同一个 `agent_node`，内部通过 `cls["agent"]` 分发到 `_explore_agent` / `_weather_agent` 私有函数，保持图结构简洁。

4. **ChatTongyi 直接调用**: 分类器直接实例化 ChatTongyi，复用项目已有的 LLM 配置模式（与 reranker.py 一致）。

5. **RAG 直连 ChromaDB**: explore_agent 直接使用 ChromaManager.similarity_search_with_score，不依赖 HybridRetriever（避免重新初始化 BM25 索引）。
