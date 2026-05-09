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
    "你是一个旅游规划查询分类器。分析用户关于目的地的查询，判断需要调用哪些 Agent "
    "来获取信息。\n\n"
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
