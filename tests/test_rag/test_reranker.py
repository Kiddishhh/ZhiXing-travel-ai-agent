"""重排序器单元测试"""
import pytest
from unittest.mock import patch
from langchain_core.documents import Document
from app.rag.reranker import LLMReranker


class TestLLMRerankerInit:
    """初始化参数测试"""

    def test_default_values(self):
        reranker = LLMReranker()
        assert reranker.top_k == 5
        assert reranker.score_threshold == 0.0
        assert reranker.max_chars == 2000

    def test_custom_values(self):
        reranker = LLMReranker(
            model_name="qwen-turbo",
            temperature=0.0,
            top_k=10,
            score_threshold=5.0,
            max_chars=1000,
        )
        assert reranker.top_k == 10
        assert reranker.score_threshold == 5.0
        assert reranker.max_chars == 1000


class TestLongContextReorder:
    """LongContextReorder 测试"""

    def test_reorder_basic(self):
        """验证文档被重新排列为头尾高相关、中间低相关"""
        reranker = LLMReranker()
        docs = [
            Document(page_content="最高相关", metadata={"relevance_score": 10.0}),
            Document(page_content="第二相关", metadata={"relevance_score": 8.0}),
            Document(page_content="第三相关", metadata={"relevance_score": 7.0}),
            Document(page_content="第四相关", metadata={"relevance_score": 5.0}),
            Document(page_content="第五相关", metadata={"relevance_score": 4.0}),
        ]
        result = reranker._long_context_reorder(docs)
        assert len(result) == 5
        # 最高分应在首位
        assert result[0].metadata["relevance_score"] == 10.0
        # 次高分应在末位
        assert result[-1].metadata["relevance_score"] == 8.0

    def test_reorder_two_docs_unchanged(self):
        """<=2 个文档时顺序不变"""
        reranker = LLMReranker()
        docs = [
            Document(page_content="A", metadata={"relevance_score": 10.0}),
            Document(page_content="B", metadata={"relevance_score": 8.0}),
        ]
        result = reranker._long_context_reorder(docs)
        assert result[0].metadata["relevance_score"] == 10.0
        assert result[1].metadata["relevance_score"] == 8.0

    def test_reorder_empty(self):
        reranker = LLMReranker()
        result = reranker._long_context_reorder([])
        assert result == []

    def test_reorder_single_doc(self):
        reranker = LLMReranker()
        doc = Document(page_content="Only", metadata={"relevance_score": 10.0})
        result = reranker._long_context_reorder([doc])
        assert len(result) == 1
        assert result[0].metadata["relevance_score"] == 10.0


class TestRerankEdgeCases:
    """rerank 入口边缘情况测试"""

    def test_rerank_empty_query(self):
        reranker = LLMReranker()
        result = reranker.rerank("", [Document(page_content="test")])
        assert result == []

    def test_rerank_empty_docs(self):
        reranker = LLMReranker()
        result = reranker.rerank("query", [])
        assert result == []

    @patch.object(LLMReranker, "_score_document")
    def test_rerank_applies_reorder(self, mock_score):
        """验证 rerank 输出经过了 LongContextReorder"""
        mock_score.return_value = 8.0
        reranker = LLMReranker()
        docs = [
            Document(page_content=f"Doc{i}", metadata={}) for i in range(5)
        ]
        result = reranker.rerank("test query", docs)
        assert len(result) <= reranker.top_k
        for doc in result:
            assert "relevance_score" in doc.metadata
