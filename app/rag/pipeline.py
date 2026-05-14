"""
RAG 管道 — 串联查询优化、混合检索、上下文扩展、重排序
"""
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from app.rag.query_optimizer import QueryOptimizer, QueryOptimizeResult
from app.rag.retriever import HybridRetriever
from app.rag.reranker import LLMReranker
from app.rag.text_splitter import ParentDocumentSplitter
from app.utils.logger import app_logger


@dataclass
class RAGPipelineResult:
    """管道执行结果"""
    original_query: str
    strategy: str = ""
    optimized_queries: List[str] = field(default_factory=list)
    child_docs: List[Document] = field(default_factory=list)
    parent_docs: List[Document] = field(default_factory=list)
    final_docs: List[Document] = field(default_factory=list)
    child_count: int = 0
    parent_count: int = 0


class RAGPipeline:
    """RAG 检索管道 — 各组件通过构造器注入，管道只负责编排"""

    def __init__(
        self,
        optimizer: QueryOptimizer,
        retriever: HybridRetriever,
        parent_splitter: ParentDocumentSplitter,
        reranker: LLMReranker,
    ):
        self.optimizer = optimizer
        self.retriever = retriever
        self.parent_splitter = parent_splitter
        self.reranker = reranker

    def run(self, query: str) -> RAGPipelineResult:
        if not query or not query.strip():
            app_logger.warning("管道收到空查询")
            return RAGPipelineResult(original_query=query)

        app_logger.info(f"RAG 管道开始: query='{query[:80]}'")

        # 阶段 1: 查询优化
        opt_result = self.optimizer.optimize(query)
        app_logger.info(f"[1/4] 查询优化: strategy={opt_result.strategy}, "
                        f"生成 {len(opt_result.optimized_queries)} 个查询")

        # 阶段 2: 混合检索（多查询 + 合并去重）
        child_docs = self._retrieve_and_merge(opt_result)
        app_logger.info(f"[2/4] 混合检索: {len(child_docs)} 个子文档")
        if not child_docs:
            return RAGPipelineResult(
                original_query=query,
                strategy=opt_result.strategy,
                optimized_queries=opt_result.optimized_queries,
            )

        # 阶段 3: 上下文扩展（子文档 → 父文档）
        parent_docs = self.parent_splitter.get_parent_context(child_docs)
        app_logger.info(f"[3/4] 上下文扩展: {len(parent_docs)} 个父文档")
        if not parent_docs:
            app_logger.info("上下文扩展为空，fallback 使用子文档")
            parent_docs = child_docs

        # 阶段 4: 重排序
        final_docs = self.reranker.rerank(query, parent_docs)
        app_logger.info(f"[4/4] 重排序: 返回 {len(final_docs)} 个文档")

        return RAGPipelineResult(
            original_query=query,
            strategy=opt_result.strategy,
            optimized_queries=opt_result.optimized_queries,
            child_docs=child_docs,
            parent_docs=parent_docs,
            final_docs=final_docs,
            child_count=len(child_docs),
            parent_count=len(parent_docs),
        )

    def _retrieve_and_merge(self, opt_result: QueryOptimizeResult) -> List[Document]:
        """多查询检索 + 合并去重"""
        seen_ids: set = set()
        merged: List[Document] = []

        for q in opt_result.optimized_queries:
            try:
                docs = self.retriever.invoke(q)
            except Exception as e:
                app_logger.warning(f"检索失败: q='{q[:50]}', error={e}")
                continue
            for doc in docs:
                doc_id = doc.metadata.get("doc_id", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(doc)

        if opt_result.hypothetical_doc:
            try:
                hyde_docs = self.retriever.invoke(opt_result.hypothetical_doc[:500])
                for doc in hyde_docs:
                    doc_id = doc.metadata.get("doc_id", "")
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        merged.append(doc)
            except Exception as e:
                app_logger.warning(f"HyDE 检索失败: {e}")

        return merged
