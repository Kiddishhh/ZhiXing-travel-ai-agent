"""
混合检索器 - 应用服务层

实现 BM25 关键词检索 + Dense 语义检索 + RRF 倒数排名融合。
"""
from typing import Dict, List, Tuple

import jieba
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.ChromaDB.chroma_client import ChromaManager
from app.utils.logger import app_logger


class HybridRetriever:
    """混合检索器

    组合 BM25（关键词）和 Dense（语义）两种互补的检索方式，
    通过 RRF（Reciprocal Rank Fusion）进行结果融合。
    """

    def __init__(
        self,
        chroma_manager: ChromaManager,
        bm25_top_k: int = 10,
        dense_top_k: int = 10,
        rrf_k: int = 60,
        final_top_k: int = 10,
        collection_name: str = "travel",
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ):
        self.chroma_manager = chroma_manager
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k
        self.collection_name = collection_name
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        self._bm25: BM25Okapi | None = None
        self._documents: List[Document] = []
        self._doc_id_map: Dict[str, Document] = {}
        self._is_initialized: bool = False

    def initialize(self, documents: List[Document]) -> None:
        """构建 BM25 索引和向量存储

        Args:
            documents: 已切分的子文档列表
        """
        self._documents = documents
        self._doc_id_map = {}

        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            doc.metadata["doc_id"] = doc_id
            self._doc_id_map[doc_id] = doc

        self._build_bm25_index(documents)
        self._build_vector_store(documents)
        self._is_initialized = True
        app_logger.info(
            f"混合检索器初始化完成: {len(documents)} 篇文档"
        )

    # ── 索引构建 ──────────────────────────────────────

    def _build_bm25_index(self, documents: List[Document]) -> None:
        """使用 jieba 分词构建 BM25Okapi 索引"""
        tokenized_docs = []
        empty_count = 0

        for doc in documents:
            tokens = list(jieba.cut(doc.page_content))
            if tokens:
                tokenized_docs.append(tokens)
            else:
                empty_count += 1

        self._bm25 = BM25Okapi(tokenized_docs)
        app_logger.info(
            f"BM25 索引构建完成: {len(tokenized_docs)} 篇文档"
            f"{' (' + str(empty_count) + ' 篇为空被跳过)' if empty_count else ''}"
        )

    def _build_vector_store(self, documents: List[Document]) -> None:
        """将文档添加到 ChromaDB 向量存储"""
        ids = [doc.metadata["doc_id"] for doc in documents]
        self.chroma_manager.add_documents(
            documents, ids=ids, collection_name=self.collection_name
        )
        app_logger.info(
            f"向量库构建完成: {len(documents)} 篇文档添加到 '{self.collection_name}'"
        )

    # ── 检索方法 ──────────────────────────────────────

    def _bm25_search(self, query: str) -> List[Tuple[str, float]]:
        """BM25 关键词检索"""
        tokenized_query = list(jieba.cut(query))
        scores = np.array(self._bm25.get_scores(tokenized_query))

        valid = np.where(scores > 0)[0]
        if len(valid) == 0:
            return []

        sorted_idx = valid[np.argsort(scores[valid])[::-1]][: self.bm25_top_k]
        return [(f"doc_{idx}", float(scores[idx])) for idx in sorted_idx]

    def _dense_search(self, query: str) -> List[Tuple[str, float]]:
        """Dense 语义检索"""
        results = self.chroma_manager.similarity_search_with_score(
            query, k=self.dense_top_k, collection_name=self.collection_name
        )
        dense_list = []
        for doc, distance in results:
            doc_id = doc.metadata.get("doc_id", "")
            if doc_id:
                dense_list.append((doc_id, distance))
        return dense_list

    # ── RRF 融合 ──────────────────────────────────────

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """加权 RRF（Reciprocal Rank Fusion）倒数排名融合"""
        rrf_scores: Dict[str, float] = {}
        k = self.rrf_k

        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] = self.bm25_weight * (1.0 / (k + rank + 1))

        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0.0) + self.dense_weight * (1.0 / (k + rank + 1))
            )

        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results[: self.final_top_k]

    # ── 检索入口 ──────────────────────────────────────

    def invoke(self, query: str) -> List[Document]:
        """混合检索入口

        Args:
            query: 用户查询

        Returns:
            按 RRF 分数降序排列的文档列表
        """
        if not query or not query.strip():
            app_logger.error("拒绝空查询")
            return []

        if not self._is_initialized:
            app_logger.error("检索器尚未初始化，请先调用 initialize()")
            return []

        if self._bm25 is None:
            app_logger.error("BM25 索引为空")
            return []

        app_logger.info(f"混合检索开始: query='{query[:50]}'")

        bm25_results = self._bm25_search(query)
        app_logger.debug(f"  BM25 检索到 {len(bm25_results)} 条结果")

        dense_results = self._dense_search(query)
        app_logger.debug(f"  Dense 检索到 {len(dense_results)} 条结果")

        fused = self._rrf_fusion(bm25_results, dense_results)

        result_docs = []
        for doc_id, rrf_score in fused:
            doc = self._doc_id_map.get(doc_id)
            if doc is not None:
                doc.metadata["rrf_score"] = round(rrf_score, 4)
                result_docs.append(doc)

        app_logger.info(f"混合检索完成: 返回 {len(result_docs)} 条结果")
        return result_docs
