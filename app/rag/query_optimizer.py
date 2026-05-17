"""
查询优化器

使用 LLM 自主判断用户查询类型，执行 multi-query / HyDE / 查询改写策略。
"""
from dataclasses import dataclass
from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from app.utils.logger import app_logger


StrategyType = Literal["multi_query", "hyde", "rewrite", "none"]


@dataclass
class QueryOptimizeResult:
    """查询优化结果"""
    original_query: str
    strategy: StrategyType
    optimized_queries: list[str]
    hypothetical_doc: Optional[str] = None


SYSTEM_PROMPT = """你是一个旅游信息查询优化助手。分析用户的查询意图，选择最合适的优化策略并执行。

## 策略说明

- **multi_query**: 将用户查询拆分为 2-4 个不同角度的子查询，覆盖同义词、相关概念、不同表述方式。适合宽泛或歧义查询。
- **hyde**: 生成一份假设的旅游指南文档片段（200-500字），即使内容虚构，其用词和结构接近真实文档。适合具体但检索词可能不匹配的知识性查询。
- **rewrite**: 补全模糊指代、修正口语化表述、统一术语。适合表述不完整或带有口语化的查询。
- **none**: 查询已经明确具体，无需优化。适合含具体地名、日期、价格等明确的精确查询。

## 输出格式

返回 JSON，字段含义：
- strategy: 选择的策略名称
- optimized_queries: 优化后的查询列表（multi_query 返回多个，rewrite/none 返回 1 个，hyde 返回原始查询）
- hypothetical_doc: 仅 hyde 策略需要，其他策略为 null

## 注意事项

1. 简单寒暄、问候（如"你好"）使用 none 策略
2. 包含明确地名 + 具体需求的查询（如"北京三天行程"）使用 none 策略
3. 查询中只有模糊意向（如"有什么好玩的地方"）优先使用 multi_query"""


class QueryOptimizer:
    """查询优化器 — LLM 分类 + 策略执行二合一"""

    def __init__(self, model_name: str = "qwen-turbo", temperature: float = 0.0):
        self._structured_llm = (
            ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=settings.dashscope_api_key,
                base_url=settings.qwen_base_url,
                extra_body={"enable_thinking": False},
                max_retries=2,
                request_timeout=30.0,
            ).with_structured_output(QueryOptimizeResult, method="function_calling")
        )

    def optimize(self, query: str) -> QueryOptimizeResult:
        """执行查询优化，LLM 失败时 fallback 到 none"""
        if not query or not query.strip():
            return QueryOptimizeResult(
                original_query=query,
                strategy="none",
                optimized_queries=[query] if query else [],
            )

        app_logger.info(f"查询优化开始: query='{query[:80]}'")

        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"请优化以下用户查询：\n\n{query}"),
            ]
            result = self._structured_llm.invoke(messages)
            # DashScope function_calling 可能返回 dict 而非 QueryOptimizeResult
            if isinstance(result, dict):
                result = QueryOptimizeResult(
                    original_query=query,
                    strategy=result.get("strategy", "none"),
                    optimized_queries=result.get("optimized_queries", [query]),
                    hypothetical_doc=result.get("hypothetical_doc"),
                )
            app_logger.info(f"查询优化完成: strategy={result.strategy}")
            return result
        except Exception as e:
            app_logger.warning(f"查询优化失败，回退到 none 策略: {e}")
            return QueryOptimizeResult(
                original_query=query,
                strategy="none",
                optimized_queries=[query],
            )
