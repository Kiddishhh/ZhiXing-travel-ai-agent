"""
RAG 管道单元测试 — 用 mock 输入模拟完整管线流程

运行方式: python -m pytest tests/test_rag/test_pipeline.py -v -s
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.rag.pipeline import RAGPipeline, RAGPipelineResult
from app.rag.query_optimizer import QueryOptimizer, QueryOptimizeResult


class TestRAGPipelineResult:
    """数据模型测试"""

    def test_default_values(self):
        print("\n=== 测试: RAGPipelineResult 默认值 ===")
        print("[输入] original_query='测试查询'")
        result = RAGPipelineResult(original_query="测试查询")

        print("[断言] original_query == '测试查询'", end=" ")
        assert result.original_query == "测试查询"
        print("[OK]")

        print("[断言] strategy 默认为空", end=" ")
        assert result.strategy == ""
        print("[OK]")

        print("[断言] optimized_queries 默认为空列表", end=" ")
        assert result.optimized_queries == []
        print("[OK]")

        print("[断言] child_count == 0, parent_count == 0", end=" ")
        assert result.child_count == 0
        assert result.parent_count == 0
        print("[OK]")

        print("→ 默认值全部正确\n")

    def test_full_result(self):
        print("\n=== 测试: RAGPipelineResult 完整结果 ===")
        docs = [Document(page_content="测试文档")]

        print("[输入] strategy='multi_query', child_count=3, parent_count=1")
        result = RAGPipelineResult(
            original_query="北京旅游",
            strategy="multi_query",
            optimized_queries=["q1", "q2"],
            child_docs=docs, parent_docs=docs, final_docs=docs,
            child_count=3, parent_count=1,
        )

        print("[断言] strategy == 'multi_query'", end=" ")
        assert result.strategy == "multi_query"
        print("[OK]")

        print("[断言] len(optimized_queries) == 2", end=" ")
        assert len(result.optimized_queries) == 2
        print("[OK]")

        print("[断言] child_count == 3", end=" ")
        assert result.child_count == 3
        print("[OK]")

        print("→ 完整结果构造正确\n")


class TestRAGPipeline:
    """管线流程测试 — mock 注入, 可视流程"""

    def test_run_empty_query(self):
        print("\n=== 测试: 空查询 → 直接返回空结果 ===")
        print("[注入] 创建 mock 组件（全都不应被调用）")

        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_retriever = MagicMock()
        mock_splitter = MagicMock()
        mock_reranker = MagicMock()

        pipeline = RAGPipeline(
            optimizer=mock_optimizer,
            retriever=mock_retriever,
            parent_splitter=mock_splitter,
            reranker=mock_reranker,
        )
        print("[输入] query='' (空字符串)")

        result = pipeline.run("")

        print("[断言] 返回 RAGPipelineResult", end=" ")
        assert isinstance(result, RAGPipelineResult)
        print("[OK]")

        print("[断言] original_query == ''", end=" ")
        assert result.original_query == ""
        print("[OK]")

        print("[断言] optimize() 未被调用（空查询短路）", end=" ")
        mock_optimizer.optimize.assert_not_called()
        print("[OK]")

        print("→ 空查询正确短路\n")

    def test_run_no_results(self):
        print("\n=== 测试: 检索无结果 → 提前终止 ===")
        print("[注入] optimizer.optimize → strategy='none'")
        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="测试", strategy="none", optimized_queries=["测试"],
        )

        print("[注入] retriever.invoke → [] (空, 无匹配)")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        mock_splitter = MagicMock()
        mock_reranker = MagicMock()

        pipeline = RAGPipeline(
            optimizer=mock_optimizer,
            retriever=mock_retriever,
            parent_splitter=mock_splitter,
            reranker=mock_reranker,
        )
        print("[输入] query='测试'")

        result = pipeline.run("测试")

        print("[断言] strategy == 'none'", end=" ")
        assert result.strategy == "none"
        print("[OK]")

        print("[断言] final_docs == []", end=" ")
        assert result.final_docs == []
        print("[OK]")

        print("[断言] get_parent_context / rerank 均未调用", end=" ")
        mock_splitter.get_parent_context.assert_not_called()
        mock_reranker.rerank.assert_not_called()
        print("[OK]")

        print("→ 无结果时正确终止\n")

    def test_run_full_pipeline(self):
        print("\n=== 测试: [1/4]优化 → [2/4]检索 → [3/4]扩展 → [4/4]重排序 ===")
        child_docs = [
            Document(page_content="故宫",
                     metadata={"doc_id": "doc_0", "parent_id": "parent_0"}),
        ]
        parent_docs = [
            Document(page_content="故宫是明清皇家宫殿...",
                     metadata={"parent_id": "parent_0"}),
        ]

        print("[注入] optimizer → strategy='none'")
        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="北京故宫", strategy="none", optimized_queries=["北京故宫"],
        )

        print("[注入] retriever.invoke → 1 篇子文档")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = child_docs

        print("[注入] splitter.get_parent_context → 1 篇父文档")
        mock_splitter = MagicMock()
        mock_splitter.get_parent_context.return_value = parent_docs

        print("[注入] reranker.rerank → 1 篇最终文档")
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = parent_docs

        pipeline = RAGPipeline(
            optimizer=mock_optimizer,
            retriever=mock_retriever,
            parent_splitter=mock_splitter,
            reranker=mock_reranker,
        )
        print("[输入] query='北京故宫'")

        print("  [1/4] optimizer.optimize → strategy=none")
        print("  [2/4] retriever.invoke → 1 篇子文档")
        print("  [3/4] splitter.get_parent_context → 1 篇父文档")
        print("  [4/4] reranker.rerank → 最终排序")
        result = pipeline.run("北京故宫")

        print("[断言] strategy == 'none'", end=" ")
        assert result.strategy == "none"
        print("[OK]")

        print("[断言] child_count == 1, parent_count == 1", end=" ")
        assert result.child_count == 1
        assert result.parent_count == 1
        print("[OK]")

        print("[断言] len(final_docs) == 1", end=" ")
        assert len(result.final_docs) == 1
        print("[OK]")

        print("[断言] retriever.invoke 被调用 1 次", end=" ")
        mock_retriever.invoke.assert_called_once()
        print("[OK]")

        print("[断言] get_parent_context 被调用", end=" ")
        mock_splitter.get_parent_context.assert_called_once()
        print("[OK]")

        print("[断言] reranker.rerank 被调用 1 次", end=" ")
        mock_reranker.rerank.assert_called_once()
        print("[OK]")

        print("→ 完整管线 4 阶段全部通过\n")

    def test_run_expand_fallback(self):
        print("\n=== 测试: get_parent_context 返回空 → fallback 使用 child_docs ===")
        child_docs = [
            Document(page_content="故宫",
                     metadata={"doc_id": "doc_0", "parent_id": "parent_0"}),
        ]

        print("[注入] optimizer → strategy='none'")
        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="北京故宫", strategy="none", optimized_queries=["北京故宫"],
        )

        print("[注入] retriever.invoke → 1 篇子文档")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = child_docs

        print("[注入] splitter.get_parent_context → [] (空, 无映射)")
        mock_splitter = MagicMock()
        mock_splitter.get_parent_context.return_value = []

        print("[注入] reranker.rerank → 返回 child_docs")
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = child_docs

        pipeline = RAGPipeline(
            optimizer=mock_optimizer,
            retriever=mock_retriever,
            parent_splitter=mock_splitter,
            reranker=mock_reranker,
        )
        print("[输入] query='北京故宫'")

        print("  [1/4] optimize → strategy=none")
        print("  [2/4] retrieve → 1 篇子文档")
        print("  [3/4] get_parent_context → 空 → fallback 到 child_docs")
        print("  [4/4] rerank → 用 child_docs 继续")
        result = pipeline.run("北京故宫")

        print("[断言] parent_docs == child_docs (fallback)", end=" ")
        assert result.parent_docs == child_docs
        print("[OK]")

        print("[断言] rerank 仍被调用", end=" ")
        mock_reranker.rerank.assert_called_once()
        print("[OK]")

        print("→ 扩展为空时正确降级\n")

    def test_run_hyde_extra_search(self):
        print("\n=== 测试: HyDE 策略 → 假设文档做额外检索 ===")
        child_docs = [
            Document(page_content="故宫",
                     metadata={"doc_id": "doc_0", "parent_id": "parent_0"}),
        ]
        hyde_extra = [
            Document(page_content="故宫历史",
                     metadata={"doc_id": "doc_1", "parent_id": "parent_1"}),
        ]

        print("[注入] optimizer → strategy='hyde' + hypothetical_doc")
        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="北京故宫历史", strategy="hyde",
            optimized_queries=["北京故宫历史"],
            hypothetical_doc="故宫是中国明清两代的皇家宫殿...",
        )

        print("[注入] retriever.invoke 调用 2 次: ")
        print("  第1次(原查询) → [doc_0]")
        print("  第2次(HyDE假设文档) → [doc_1]")
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = [child_docs, hyde_extra]

        parent_docs = [
            Document(page_content="parent0", metadata={"parent_id": "parent_0"}),
            Document(page_content="parent1", metadata={"parent_id": "parent_1"}),
        ]

        print("[注入] splitter.get_parent_context → 2 篇父文档")
        mock_splitter = MagicMock()
        mock_splitter.get_parent_context.return_value = parent_docs

        print("[注入] reranker.rerank → 最终排序")
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = parent_docs

        pipeline = RAGPipeline(
            optimizer=mock_optimizer,
            retriever=mock_retriever,
            parent_splitter=mock_splitter,
            reranker=mock_reranker,
        )
        print("[输入] query='北京故宫历史'")

        print("  [1/4] optimize → hyde (含假设文档)")
        print("  [2/4] retrieve(原查询) + retrieve(HyDE) → 2 篇合并")
        print("  [3/4] get_parent_context → 2 篇父文档")
        print("  [4/4] rerank → 最终排序")
        result = pipeline.run("北京故宫历史")

        print("[断言] retriever.invoke 被调用 2 次", end=" ")
        assert mock_retriever.invoke.call_count == 2
        print("[OK]")

        print("[断言] strategy == 'hyde'", end=" ")
        assert result.strategy == "hyde"
        print("[OK]")

        print("[断言] child_count == 2", end=" ")
        assert result.child_count == 2
        print("[OK]")

        print("→ HyDE 额外检索正确合并\n")

    def test_run_retriever_exception_skipped(self):
        print("\n=== 测试: multi_query 中单个检索异常 → 跳过继续 ===")
        child_docs = [
            Document(page_content="故宫",
                     metadata={"doc_id": "doc_0", "parent_id": "parent_0"}),
        ]

        print("[注入] optimizer → multi_query, 3 个子查询")
        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="测试", strategy="multi_query",
            optimized_queries=["q1", "q2", "q3"],
        )

        print("[注入] retriever.invoke 调用 3 次:")
        print("  invoke('q1') → RuntimeError [跳过]")
        print("  invoke('q2') → [] (空)      [跳过]")
        print("  invoke('q3') → [doc_0]      [采纳]")
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = [
            RuntimeError("检索超时"), [], child_docs,
        ]

        parent_docs = [Document(page_content="parent0")]
        mock_splitter = MagicMock()
        mock_splitter.get_parent_context.return_value = parent_docs
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = child_docs

        pipeline = RAGPipeline(
            optimizer=mock_optimizer,
            retriever=mock_retriever,
            parent_splitter=mock_splitter,
            reranker=mock_reranker,
        )
        print("[输入] query='测试'")

        print("  [1/4] optimize → multi_query [q1, q2, q3]")
        print("  [2/4] retrieve(q1)→异常跳过, retrieve(q2)→空跳过, retrieve(q3)→1篇")
        print("  [3/4] get_parent_context → 继续")
        print("  [4/4] rerank → 最终")
        result = pipeline.run("测试")

        print("[断言] retriever.invoke 被调用 3 次", end=" ")
        assert mock_retriever.invoke.call_count == 3
        print("[OK]")

        print("[断言] child_count == 1 (仅 q3)", end=" ")
        assert result.child_count == 1
        print("[OK]")

        print("[断言] strategy == 'multi_query'", end=" ")
        assert result.strategy == "multi_query"
        print("[OK]")

        print("→ 异常查询正确跳过, 管线继续\n")
