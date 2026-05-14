"""混合检索器单元测试"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from app.rag.retriever import HybridRetriever


def _print_stage(stage: str, total: int, current: int):
    print(f"\n{'─'*50}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'─'*50}")


class TestHybridRetrieverInit:
    """初始化参数测试"""

    def test_default_weights(self):
        _print_stage("HybridRetriever 初始化", 3, 1)
        mock_chroma = MagicMock()
        retriever = HybridRetriever(chroma_manager=mock_chroma)
        assert retriever.bm25_weight == 0.4
        assert retriever.dense_weight == 0.6

    def test_custom_weights(self):
        _print_stage("HybridRetriever 初始化", 3, 2)
        mock_chroma = MagicMock()
        retriever = HybridRetriever(
            chroma_manager=mock_chroma,
            bm25_weight=0.5,
            dense_weight=0.5,
        )
        assert retriever.bm25_weight == 0.5
        assert retriever.dense_weight == 0.5

    def test_default_collection_name(self):
        _print_stage("HybridRetriever 初始化", 3, 3)
        mock_chroma = MagicMock()
        retriever = HybridRetriever(chroma_manager=mock_chroma)
        assert retriever.collection_name == "travel_children"


class TestRRFFusion:
    """加权 RRF 融合算法测试"""

    def test_weighted_rrf_basic(self):
        _print_stage("加权 RRF 融合", 4, 1)
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
        print(f"[RRF] BM25权重={retriever.bm25_weight}, Dense权重={retriever.dense_weight}, k={retriever.rrf_k}")
        print(f"[RRF] 融合结果 doc_ids: {doc_ids}")

    def test_weighted_rrf_dense_favored(self):
        """dense_weight > bm25_weight 时语义检索结果权重更大"""
        _print_stage("加权 RRF 融合", 4, 2)
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
        _print_stage("加权 RRF 融合", 4, 3)
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
        _print_stage("加权 RRF 融合", 4, 4)
        mock_chroma = MagicMock()
        retriever = HybridRetriever(chroma_manager=mock_chroma)
        result = retriever._rrf_fusion([], [])
        assert len(result) == 0


class TestHybridRetrieverInvoke:
    """invoke 入口测试"""

    def test_invoke_empty_query(self):
        _print_stage("invoke 入口", 2, 1)
        mock_chroma = MagicMock()
        retriever = HybridRetriever(chroma_manager=mock_chroma)
        result = retriever.invoke("")
        assert result == []

    def test_invoke_not_initialized(self):
        _print_stage("invoke 入口", 2, 2)
        mock_chroma = MagicMock()
        retriever = HybridRetriever(chroma_manager=mock_chroma)
        result = retriever.invoke("北京旅游")
        assert result == []
