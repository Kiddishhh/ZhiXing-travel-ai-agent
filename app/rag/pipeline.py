"""
RAG 管道 — 串联查询优化、检索、上下文扩展、重排序
"""
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from app.core.ChromaDB.chroma_client import ChromaManager
from app.rag.query_optimizer import QueryOptimizer, QueryOptimizeResult
from app.rag.retriever import HybridRetriever
from app.rag.reranker import LLMReranker
from app.rag.text_splitter import ParentDocumentSplitter
from app.utils.logger import app_logger


@dataclass
class RAGPipelineResult:
    """RAG 管道执行结果"""
    original_query: str
    strategy: str = ""
    optimized_queries: List[str] = field(default_factory=list)
    child_docs: List[Document] = field(default_factory=list)
    parent_docs: List[Document] = field(default_factory=list)
    final_docs: List[Document] = field(default_factory=list)
    child_count: int = 0
    parent_count: int = 0


class RAGPipeline:
    """RAG 检索管道

    串联查询优化 → 混合检索 → 父子文档扩展 → LLM 重排序。
    各组件通过构造器注入，管道只负责编排。
    """

    def __init__(
        self,
        optimizer: QueryOptimizer,
        retriever: HybridRetriever,
        chroma_manager: ChromaManager,
        reranker: LLMReranker,
        parent_collection_name: str = "travel_parents",
    ):
        self.optimizer = optimizer
        self.retriever = retriever
        self.chroma_manager = chroma_manager
        self.reranker = reranker
        self.parent_collection_name = parent_collection_name

    def run(self, query: str) -> RAGPipelineResult:
        """执行完整检索管道

        Args:
            query: 用户原始查询

        Returns:
            RAGPipelineResult 包含各阶段结果
        """
        if not query or not query.strip():
            app_logger.warning("管道收到空查询")
            return RAGPipelineResult(original_query=query)

        app_logger.info(f"RAG 管道开始: query='{query[:80]}'")

        # 1. 查询优化
        optimize_result = self.optimizer.optimize(query)

        # 2. 多查询检索 + 合并去重
        child_docs = self._retrieve_and_merge(optimize_result)
        if not child_docs:
            app_logger.info("检索无结果，管道终止")
            return RAGPipelineResult(
                original_query=query,
                strategy=optimize_result.strategy,
                optimized_queries=optimize_result.optimized_queries,
            )

        # 3. 父子文档扩展
        parent_docs = self._expand_context(child_docs)
        if not parent_docs:
            app_logger.info("上下文扩展为空，fallback 使用子文档")
            parent_docs = child_docs

        # 4. 重排序
        final_docs = self.reranker.rerank(query, parent_docs)

        app_logger.info(
            f"RAG 管道完成: "
            f"optimized={len(optimize_result.optimized_queries)}q, "
            f"child={len(child_docs)}, parent={len(parent_docs)}, "
            f"final={len(final_docs)}"
        )

        return RAGPipelineResult(
            original_query=query,
            strategy=optimize_result.strategy,
            optimized_queries=optimize_result.optimized_queries,
            child_docs=child_docs,
            parent_docs=parent_docs,
            final_docs=final_docs,
            child_count=len(child_docs),
            parent_count=len(parent_docs),
        )

    # ── 内部方法 ──────────────────────────────────────

    def _retrieve_and_merge(
        self, optimize_result: QueryOptimizeResult
    ) -> List[Document]:
        """对每个优化查询执行检索，合并去重"""
        seen_ids: set = set()
        merged: List[Document] = []

        for q in optimize_result.optimized_queries:
            try:
                docs = self.retriever.invoke(q)
            except Exception as e:
                app_logger.warning(f"检索失败: q='{q[:50]}', error={e}")
                continue

            for doc in docs:
                doc_id = doc.metadata.get("doc_id", "")
                if not doc_id or doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                merged.append(doc)

        # 对于 hyde 策略，额外用假设文档做一次语义检索
        if optimize_result.hypothetical_doc:
            try:
                hyde_docs = self.retriever.invoke(
                    optimize_result.hypothetical_doc[:500]
                )
                for doc in hyde_docs:
                    doc_id = doc.metadata.get("doc_id", "")
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        merged.append(doc)
            except Exception as e:
                app_logger.warning(f"HyDE 检索失败: {e}")

        app_logger.info(
            f"多查询检索: {len(optimize_result.optimized_queries)} 查询, "
            f"合并去重后 {len(merged)} 篇"
        )
        return merged

    def _expand_context(self, child_docs: List[Document]) -> List[Document]:
        """将子文档扩展为父文档"""
        try:
            parent_collection = self.chroma_manager.get_vectorstore(
                self.parent_collection_name
            )
            return ParentDocumentSplitter.expand_context(
                child_docs, parent_collection
            )
        except Exception as e:
            app_logger.warning(f"上下文扩展失败: {e}")
            return []
