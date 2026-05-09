"""
RAG 系统功能测试脚本

测试混合检索与 LLM 重排序功能。
"""
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.ChromaDB.chroma_client import ChromaManager
from app.rag.document_loader import DocumentManager
from app.rag.retriever import HybridRetriever
from app.rag.text_splitter import ParentDocumentSplitter
from app.utils.logger import app_logger


def _build_retriever():
    """构建并初始化检索器"""
    doc_manager = DocumentManager()
    documents = []
    for load_fn in [
        doc_manager.load_destination_documents,
        doc_manager.load_food_documents,
        doc_manager.load_accommodation_documents,
    ]:
        documents.extend(load_fn())

    if not documents:
        app_logger.error("未加载到任何文档")
        return None

    splitter = ParentDocumentSplitter(
        parent_chunk_size=1000, parent_chunk_overlap=200,
        child_chunk_size=200, child_chunk_overlap=50,
    )
    _, child_docs = splitter.split_documents(documents)

    chroma_manager = ChromaManager()
    retriever = HybridRetriever(chroma_manager=chroma_manager)
    retriever.initialize(child_docs)
    return retriever


def test_retriever(retriever):
    """运行检索测试"""
    test_queries = [
        "北京有什么好玩的景点",
        "推荐一些四川美食",
        "西安适合带孩子去吗",
        "杭州的住宿推荐",
    ]
    passed = 0
    for query in test_queries:
        t0 = time.time()
        results = retriever.invoke(query)
        elapsed = time.time() - t0
        if results:
            app_logger.info(
                f"  [PASS] query='{query}' "
                f"→ {len(results)} 条结果 (耗时 {elapsed:.2f}s)"
            )
            for i, doc in enumerate(results[:3]):
                score = doc.metadata.get("rrf_score", "N/A")
                source = doc.metadata.get("source", "N/A")
                snippet = doc.page_content[:60].replace("\n", " ")
                app_logger.info(f"    #{i + 1} [score={score}] [{source}] {snippet}...")
            passed += 1
        else:
            app_logger.warning(f"  [FAIL] query='{query}' → 无结果")
    return passed, len(test_queries)


def test_reranker(retriever):
    """（可选）LLM 重排序测试"""
    query = "北京旅游景点推荐"
    sample_docs = retriever.invoke(query)
    if not sample_docs:
        app_logger.warning("重排序测试跳过: 无检索结果")
        return

    from app.rag.reranker import LLMReranker

    reranker = LLMReranker(top_k=3)
    t0 = time.time()
    reranked = reranker.rerank(query, sample_docs)
    elapsed = time.time() - t0
    app_logger.info(
        f"  [OK] 重排序完成: {len(sample_docs)} → {len(reranked)} 条 (耗时 {elapsed:.2f}s)"
    )
    for i, doc in enumerate(reranked):
        rs = doc.metadata.get("relevance_score", "N/A")
        snippet = doc.page_content[:60].replace("\n", " ")
        app_logger.info(f"    #{i + 1} [relevance={rs}] {snippet}...")


def main():
    app_logger.info("=" * 60)
    app_logger.info("RAG 系统功能测试")
    app_logger.info("=" * 60)

    retriever = _build_retriever()
    if retriever is None:
        sys.exit(1)

    app_logger.info("[1/2] 混合检索测试...")
    passed, total = test_retriever(retriever)
    app_logger.info(f"检索测试: {passed}/{total} 通过")

    app_logger.info("[2/2] LLM 重排序测试...")
    try:
        test_reranker(retriever)
    except Exception as e:
        app_logger.warning(f"重排序测试跳过: {e}")

    app_logger.info("=" * 60)
    app_logger.info("测试完成")
    app_logger.info("=" * 60)


if __name__ == "__main__":
    main()
