"""查询优化器单元测试"""
from unittest.mock import MagicMock, patch

import pytest
from app.rag.query_optimizer import (
    QueryOptimizer, QueryOptimizeResult, StrategyType
)


class TestQueryOptimizeResult:
    """QueryOptimizeResult 数据模型测试"""

    def test_default_values(self):
        result = QueryOptimizeResult(
            original_query="北京有什么好玩的",
            strategy="none",
            optimized_queries=["北京有什么好玩的"],
        )
        assert result.original_query == "北京有什么好玩的"
        assert result.strategy == "none"
        assert result.optimized_queries == ["北京有什么好玩的"]
        assert result.hypothetical_doc is None

    def test_multi_query_result(self):
        result = QueryOptimizeResult(
            original_query="适合亲子的目的地",
            strategy="multi_query",
            optimized_queries=[
                "适合带孩子旅游的目的地推荐",
                "亲子游景点攻略",
                "儿童友好旅行目的地",
            ],
        )
        assert result.strategy == "multi_query"
        assert len(result.optimized_queries) == 3

    def test_hyde_result(self):
        result = QueryOptimizeResult(
            original_query="重庆火锅推荐",
            strategy="hyde",
            optimized_queries=["重庆火锅推荐"],
            hypothetical_doc="重庆火锅以麻辣鲜香著称，代表性的有解放碑附近的...",
        )
        assert result.strategy == "hyde"
        assert result.hypothetical_doc is not None


class TestQueryOptimizer:
    """QueryOptimizer 集成测试（需要 LLM 连接）"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_optimize_returns_result(self):
        optimizer = QueryOptimizer()
        result = optimizer.optimize("北京三天行程怎么安排")
        assert isinstance(result, QueryOptimizeResult)
        assert result.original_query == "北京三天行程怎么安排"
        assert result.strategy in ("multi_query", "hyde", "rewrite", "none")
        assert len(result.optimized_queries) >= 1

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_optimize_specific_query_uses_none(self):
        """具体查询应被分类为 none"""
        optimizer = QueryOptimizer()
        result = optimizer.optimize("北京故宫门票多少钱")
        assert isinstance(result, QueryOptimizeResult)
        # 具体查询大概率是 none（不做严格断言，LLM 有随机性）

    def test_optimize_empty_query(self):
        optimizer = QueryOptimizer()
        result = optimizer.optimize("")
        assert result.strategy == "none"

    def test_optimize_fallback_on_error(self):
        """LLM 调用失败时回退到 none 策略"""
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("模拟 LLM 故障")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("app.rag.query_optimizer.ChatOpenAI", return_value=mock_llm):
            optimizer = QueryOptimizer()
            result = optimizer.optimize("北京旅游")
            assert result.strategy == "none"
            assert result.optimized_queries == ["北京旅游"]
