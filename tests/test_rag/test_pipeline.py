"""
测试 RAG 管线流程（mock 注入）

运行方式: python -m pytest tests/test_rag/test_pipeline.py -v -s
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.rag.pipeline import RAGPipeline, RAGPipelineResult
from app.rag.query_optimizer import QueryOptimizer, QueryOptimizeResult


class TestRAGPipelineResult:
    """数据模型测试"""

    def test_default_values(self):
        print("\n[测试] RAGPipelineResult 默认值")
        result = RAGPipelineResult(original_query="测试查询")

        print("[注入] original_query='测试查询'")
        assert result.original_query == "测试查询"
        assert result.strategy == ""
        assert len(result.child_docs) == 0
        print("[OK] 默认值全部正确")

    def test_full_result(self):
        print("\n[测试] RAGPipelineResult 完整构造")
        docs = [Document(page_content="测试")]
        result = RAGPipelineResult(
            original_query="北京旅游", strategy="multi_query",
            optimized_queries=["q1", "q2"], child_docs=docs,
            parent_docs=docs, final_docs=docs,
        )

        print("[注入] strategy='multi_query', child=1, parent=1")
        assert result.strategy == "multi_query"
        assert len(result.optimized_queries) == 2
        assert len(result.child_docs) == 1
        print("[OK] 完整结果构造正确")


class TestRAGPipeline:
    """管线流程测试"""

    def test_run_empty_query(self):
        print("\n[测试] 空查询短路")
        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_retriever = MagicMock()
        mock_splitter = MagicMock()
        mock_reranker = MagicMock()

        pipeline = RAGPipeline(
            optimizer=mock_optimizer, retriever=mock_retriever,
            parent_splitter=mock_splitter, reranker=mock_reranker,
        )

        print("[注入] query=''")
        result = pipeline.run("")

        assert isinstance(result, RAGPipelineResult)
        assert result.original_query == ""
        mock_optimizer.optimize.assert_not_called()
        print("[OK] 空查询正确短路, 下游未调用")

    def test_run_no_results(self):
        print("\n[测试] 检索无结果 → 提前终止")
        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="测试", strategy="none", optimized_queries=["测试"],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_splitter = MagicMock()
        mock_reranker = MagicMock()

        pipeline = RAGPipeline(
            optimizer=mock_optimizer, retriever=mock_retriever,
            parent_splitter=mock_splitter, reranker=mock_reranker,
        )

        print("[注入] optimizer → strategy='none'")
        print("[注入] retriever.invoke → [] (空)")
        result = pipeline.run("测试")

        assert result.strategy == "none"
        assert result.final_docs == []
        mock_reranker.rerank.assert_not_called()
        print("[OK] 无结果时终止, rerank 未调用")

    def test_run_full_pipeline(self):
        print("\n[测试] [1/4]优化 → [2/4]检索 → [3/4]扩展 → [4/4]重排序")
        child_docs = [
            Document(page_content="故宫", metadata={"doc_id": "doc_0", "parent_id": "parent_0"}),
        ]
        parent_docs = [
            Document(page_content="故宫是明清皇家宫殿...", metadata={"parent_id": "parent_0"}),
        ]

        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="北京故宫", strategy="none", optimized_queries=["北京故宫"],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = child_docs
        mock_splitter = MagicMock()
        mock_splitter.get_parent_context.return_value = parent_docs
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = parent_docs

        pipeline = RAGPipeline(
            optimizer=mock_optimizer, retriever=mock_retriever,
            parent_splitter=mock_splitter, reranker=mock_reranker,
        )

        print("[注入] optimizer → strategy='none'")
        print("[注入] retriever.invoke → 1 篇子文档 (doc_0)")
        print("[注入] splitter.get_parent_context → 1 篇父文档 (parent_0)")
        print("[注入] reranker.rerank → 1 篇最终")

        result = pipeline.run("北京故宫")

        assert result.strategy == "none"
        assert len(result.child_docs) ==1
        assert len(result.parent_docs) ==1
        assert len(result.final_docs) == 1
        mock_retriever.invoke.assert_called_once()
        mock_splitter.get_parent_context.assert_called_once()
        mock_reranker.rerank.assert_called_once()
        print("[OK] 完整管线 4 阶段全部通过")

    def test_run_expand_fallback(self):
        print("\n[测试] 上下文扩展为空 → fallback 到子文档")
        child_docs = [
            Document(page_content="故宫", metadata={"doc_id": "doc_0", "parent_id": "parent_0"}),
        ]

        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="北京故宫", strategy="none", optimized_queries=["北京故宫"],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = child_docs
        mock_splitter = MagicMock()
        mock_splitter.get_parent_context.return_value = []
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = child_docs

        pipeline = RAGPipeline(
            optimizer=mock_optimizer, retriever=mock_retriever,
            parent_splitter=mock_splitter, reranker=mock_reranker,
        )

        print("[注入] splitter.get_parent_context → [] (空)")
        result = pipeline.run("北京故宫")

        assert result.parent_docs == child_docs
        mock_reranker.rerank.assert_called_once()
        print("[OK] 降级使用子文档, 管道不中断")

    def test_run_hyde_extra_search(self):
        print("\n[测试] HyDE 策略 → 假设文档额外检索")
        child_docs = [
            Document(page_content="故宫", metadata={"doc_id": "doc_0", "parent_id": "parent_0"}),
        ]
        hyde_extra = [
            Document(page_content="故宫历史", metadata={"doc_id": "doc_1", "parent_id": "parent_1"}),
        ]

        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="北京故宫历史", strategy="hyde",
            optimized_queries=["北京故宫历史"],
            hypothetical_doc="故宫是中国明清两代的皇家宫殿...",
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = [child_docs, hyde_extra]
        mock_splitter = MagicMock()
        mock_splitter.get_parent_context.return_value = [
            Document(page_content="p0", metadata={"parent_id": "parent_0"}),
            Document(page_content="p1", metadata={"parent_id": "parent_1"}),
        ]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []

        pipeline = RAGPipeline(
            optimizer=mock_optimizer, retriever=mock_retriever,
            parent_splitter=mock_splitter, reranker=mock_reranker,
        )

        print("[注入] optimizer → strategy='hyde' + hypothetical_doc")
        print("[注入] retriever.invoke 调用 2 次 (原查询 + HyDE)")
        result = pipeline.run("北京故宫历史")

        assert mock_retriever.invoke.call_count == 2
        assert result.strategy == "hyde"
        assert len(result.child_docs) ==2
        print("[OK] HyDE 额外检索合并, 2 篇子文档")

    def test_run_retriever_exception(self):
        print("\n[测试] multi_query 中单查询异常 → 跳过继续")
        child_docs = [
            Document(page_content="故宫", metadata={"doc_id": "doc_0", "parent_id": "parent_0"}),
        ]

        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="测试", strategy="multi_query",
            optimized_queries=["q1", "q2", "q3"],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = [
            RuntimeError("检索超时"), [], child_docs,
        ]
        mock_splitter = MagicMock()
        mock_splitter.get_parent_context.return_value = [
            Document(page_content="p0"),
        ]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = child_docs

        pipeline = RAGPipeline(
            optimizer=mock_optimizer, retriever=mock_retriever,
            parent_splitter=mock_splitter, reranker=mock_reranker,
        )

        print("[注入] retriever.invoke: q1→异常, q2→空, q3→1篇")
        result = pipeline.run("测试")

        assert mock_retriever.invoke.call_count == 3
        assert len(result.child_docs) ==1
        print("[OK] 异常跳过, 管线继续, child_count=1")
