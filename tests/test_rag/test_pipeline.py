"""RAG 管道单元测试"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.rag.pipeline import RAGPipeline, RAGPipelineResult
from app.rag.query_optimizer import QueryOptimizer, QueryOptimizeResult


class TestRAGPipelineResult:
    """RAGPipelineResult 数据模型测试"""

    def test_default_values(self):
        result = RAGPipelineResult(original_query="测试查询")
        assert result.original_query == "测试查询"
        assert result.strategy == ""
        assert result.optimized_queries == []
        assert result.child_docs == []
        assert result.parent_docs == []
        assert result.final_docs == []
        assert result.child_count == 0
        assert result.parent_count == 0

    def test_full_result(self):
        docs = [Document(page_content="测试文档")]
        result = RAGPipelineResult(
            original_query="北京旅游",
            strategy="multi_query",
            optimized_queries=["q1", "q2"],
            child_docs=docs,
            parent_docs=docs,
            final_docs=docs,
            child_count=3,
            parent_count=1,
        )
        assert result.strategy == "multi_query"
        assert len(result.optimized_queries) == 2
        assert result.child_count == 3


class TestRAGPipeline:
    """RAGPipeline 测试"""

    def test_run_empty_query(self):
        """空查询直接返回空结果"""
        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_retriever = MagicMock()
        mock_chroma = MagicMock()
        mock_reranker = MagicMock()

        pipeline = RAGPipeline(
            optimizer=mock_optimizer,
            retriever=mock_retriever,
            chroma_manager=mock_chroma,
            reranker=mock_reranker,
        )

        result = pipeline.run("")
        assert isinstance(result, RAGPipelineResult)
        assert result.original_query == ""
        mock_optimizer.optimize.assert_not_called()

    def test_run_no_results(self):
        """检索无结果时提前终止"""
        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="测试",
            strategy="none",
            optimized_queries=["测试"],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_chroma = MagicMock()
        mock_reranker = MagicMock()

        pipeline = RAGPipeline(
            optimizer=mock_optimizer,
            retriever=mock_retriever,
            chroma_manager=mock_chroma,
            reranker=mock_reranker,
        )

        result = pipeline.run("测试")
        assert result.strategy == "none"
        assert result.final_docs == []
        mock_reranker.rerank.assert_not_called()

    def test_run_full_pipeline(self):
        """正常管道流程：优化 → 检索 → 扩展 → 重排序"""
        child_docs = [
            Document(
                page_content="故宫",
                metadata={"doc_id": "doc_0", "parent_id": "beijing.md_0"},
            ),
        ]
        parent_docs = [
            Document(
                page_content="故宫是明清皇家宫殿...",
                metadata={"parent_id": "beijing.md_0"},
            ),
        ]

        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="北京故宫",
            strategy="none",
            optimized_queries=["北京故宫"],
        )

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = child_docs

        mock_chroma = MagicMock()
        mock_vectorstore = MagicMock()
        mock_chroma.get_vectorstore.return_value = mock_vectorstore

        with patch(
            "app.rag.pipeline.ParentDocumentSplitter.expand_context",
            return_value=parent_docs,
        ):
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = parent_docs

            pipeline = RAGPipeline(
                optimizer=mock_optimizer,
                retriever=mock_retriever,
                chroma_manager=mock_chroma,
                reranker=mock_reranker,
            )

            result = pipeline.run("北京故宫")

        assert result.strategy == "none"
        assert result.child_count == 1
        assert result.parent_count == 1
        assert len(result.final_docs) == 1
        mock_retriever.invoke.assert_called_once()
        mock_reranker.rerank.assert_called_once()

    def test_run_expand_fallback(self):
        """expand_context 失败时 fallback 使用 child_docs"""
        child_docs = [
            Document(
                page_content="故宫",
                metadata={"doc_id": "doc_0", "parent_id": "beijing.md_0"},
            ),
        ]

        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="北京故宫",
            strategy="none",
            optimized_queries=["北京故宫"],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = child_docs

        mock_chroma = MagicMock()
        mock_chroma.get_vectorstore.side_effect = RuntimeError("连接失败")

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = child_docs

        pipeline = RAGPipeline(
            optimizer=mock_optimizer,
            retriever=mock_retriever,
            chroma_manager=mock_chroma,
            reranker=mock_reranker,
        )

        result = pipeline.run("北京故宫")
        assert result.parent_docs == child_docs
        mock_reranker.rerank.assert_called_once()

    def test_run_hyde_extra_search(self):
        """HyDE 策略时用假设文档做额外检索"""
        child_docs = [
            Document(
                page_content="故宫",
                metadata={"doc_id": "doc_0", "parent_id": "beijing.md_0"},
            ),
        ]
        hyde_extra = [
            Document(
                page_content="故宫历史",
                metadata={"doc_id": "doc_1", "parent_id": "beijing.md_1"},
            ),
        ]

        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="北京故宫历史",
            strategy="hyde",
            optimized_queries=["北京故宫历史"],
            hypothetical_doc="故宫是中国明清两代的皇家宫殿，位于北京中轴线中心...",
        )

        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = [child_docs, hyde_extra]

        mock_chroma = MagicMock()
        mock_vectorstore = MagicMock()
        mock_chroma.get_vectorstore.return_value = mock_vectorstore

        parent_docs = [
            Document(page_content="parent0", metadata={"parent_id": "beijing.md_0"}),
            Document(page_content="parent1", metadata={"parent_id": "beijing.md_1"}),
        ]

        with patch(
            "app.rag.pipeline.ParentDocumentSplitter.expand_context",
            return_value=parent_docs,
        ):
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = parent_docs

            pipeline = RAGPipeline(
                optimizer=mock_optimizer,
                retriever=mock_retriever,
                chroma_manager=mock_chroma,
                reranker=mock_reranker,
            )

            result = pipeline.run("北京故宫历史")

        # 验证 retriever 被调用了两次（原始查询 + HyDE 文档）
        assert mock_retriever.invoke.call_count == 2
        assert result.strategy == "hyde"
        assert result.child_count == 2

    def test_run_retriever_exception_skipped(self):
        """单个查询检索异常时跳过继续"""
        child_docs = [
            Document(
                page_content="故宫",
                metadata={"doc_id": "doc_0", "parent_id": "beijing.md_0"},
            ),
        ]

        mock_optimizer = MagicMock(spec=QueryOptimizer)
        mock_optimizer.optimize.return_value = QueryOptimizeResult(
            original_query="测试",
            strategy="multi_query",
            optimized_queries=["q1", "q2", "q3"],
        )

        mock_retriever = MagicMock()
        # q1 抛异常，q2 返回空，q3 返回结果
        mock_retriever.invoke.side_effect = [
            RuntimeError("检索超时"),
            [],
            child_docs,
        ]

        mock_chroma = MagicMock()
        mock_vectorstore = MagicMock()
        mock_chroma.get_vectorstore.return_value = mock_vectorstore

        with patch(
            "app.rag.pipeline.ParentDocumentSplitter.expand_context",
            return_value=child_docs,
        ):
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = child_docs

            pipeline = RAGPipeline(
                optimizer=mock_optimizer,
                retriever=mock_retriever,
                chroma_manager=mock_chroma,
                reranker=mock_reranker,
            )

            result = pipeline.run("测试")

        assert mock_retriever.invoke.call_count == 3
        assert result.child_count == 1
        assert result.strategy == "multi_query"
