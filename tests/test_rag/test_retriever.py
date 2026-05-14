"""混合检索器单元测试"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from app.rag.retriever import HybridRetriever


class TestHybridRetrieverInit:
    """初始化参数测试"""

    def test_default_weights(self):
        mock_chroma = MagicMock()
        retriever = HybridRetriever(chroma_manager=mock_chroma)
        assert retriever.bm25_weight == 0.4
        assert retriever.dense_weight == 0.6

    def test_custom_weights(self):
        mock_chroma = MagicMock()
        retriever = HybridRetriever(
            chroma_manager=mock_chroma,
            bm25_weight=0.5,
            dense_weight=0.5,
        )
        assert retriever.bm25_weight == 0.5
        assert retriever.dense_weight == 0.5

    def test_default_collection_name(self):
        mock_chroma = MagicMock()
        retriever = HybridRetriever(chroma_manager=mock_chroma)
        assert retriever.collection_name == "travel_children"


class TestRRFFusion:
    """加权 RRF 融合算法测试"""

    def test_weighted_rrf_basic(self):
        mock_chroma = MagicMock()
        retriever = HybridRetriever(
            chroma_manager=mock_chroma,
            bm25_weight=0.4,
            dense_weight=0.6,
            rrf_k=60,
            final_top_k=5,
        )

        bm25 = [("doc_0", 5.0), ("doc_1", 3.0), ("doc_2", 1.0)]
        dense = [("doc_1", 0.9), ("doc_0", 0.8), ("doc_3", 0.7)]

        result = retriever._rrf_fusion(bm25, dense)

        assert len(result) <= 5
        doc_ids = [doc_id for doc_id, _ in result]
        assert doc_ids[0] in ("doc_0", "doc_1")

    def test_weighted_rrf_dense_favored(self):
        """dense_weight > bm25_weight 时语义检索结果权重更大"""
        mock_chroma = MagicMock()
        retriever_dense = HybridRetriever(
            chroma_manager=mock_chroma,
            bm25_weight=0.2, dense_weight=0.8, rrf_k=60,
        )

        bm25 = [("doc_0", 5.0), ("doc_1", 3.0)]
        dense = [("doc_1", 0.9), ("doc_0", 0.8)]

        result_dense = retriever_dense._rrf_fusion(bm25, dense)
        assert result_dense[0][0] == "doc_1"

    def test_weighted_rrf_single_source(self):
        mock_chroma = MagicMock()
        retriever = HybridRetriever(
            chroma_manager=mock_chroma,
            bm25_weight=0.4, dense_weight=0.6, rrf_k=60,
        )

        bm25 = [("doc_0", 5.0)]
        dense = []

        result = retriever._rrf_fusion(bm25, dense)
        assert len(result) == 1
        assert result[0][0] == "doc_0"

    def test_weighted_rrf_empty_both(self):
        mock_chroma = MagicMock()
        retriever = HybridRetriever(chroma_manager=mock_chroma)
        result = retriever._rrf_fusion([], [])
        assert len(result) == 0


class TestHybridRetrieverInvoke:
    """invoke 入口测试"""

    def test_invoke_empty_query(self):
        mock_chroma = MagicMock()
        retriever = HybridRetriever(chroma_manager=mock_chroma)
        result = retriever.invoke("")
        assert result == []

    def test_invoke_not_initialized(self):
        mock_chroma = MagicMock()
        retriever = HybridRetriever(chroma_manager=mock_chroma)
        result = retriever.invoke("北京旅游")
        assert result == []
