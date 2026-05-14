"""
测试 RAG 完整管道（集成测试脚本）

初始化 RAG 系统 → 执行测试查询 → 打印结果。

运行方式: python scripts/test_rag_pipeline.py
"""
import asyncio
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.ChromaDB.chroma_client import ChromaManager
from app.rag.document_loader import DocumentManager
from app.rag.text_splitter import ParentDocumentSplitter
from app.rag.retriever import HybridRetriever
from app.rag.query_optimizer import QueryOptimizer
from app.rag.reranker import LLMReranker
from app.rag.pipeline import RAGPipeline


async def main():
    print("\n=== 初始化 RAG 系统 ===")

    # 1. 加载文档
    doc_manager = DocumentManager()
    documents = doc_manager.load_all_documents()
    print(f"加载了 {len(documents)} 个文档")

    # 2. 切分文档
    splitter = ParentDocumentSplitter(
        parent_chunk_size=1000, parent_chunk_overlap=200,
        child_chunk_size=200, child_chunk_overlap=50,
    )
    parent_docs, child_docs = splitter.split_documents(documents)
    print(f"父文档: {len(parent_docs)}, 子文档: {len(child_docs)}")

    # 3. 构建向量库 + 检索器
    chroma_manager = ChromaManager()
    chroma_manager.delete_collection("travel_children")
    retriever = HybridRetriever(
        chroma_manager=chroma_manager,
        collection_name="travel_children",
    )
    retriever.initialize(child_docs)
    print("向量库 + BM25 索引构建完成")

    # 4. 创建 RAG 管道
    pipeline = RAGPipeline(
        optimizer=QueryOptimizer(),
        retriever=retriever,
        parent_splitter=splitter,
        reranker=LLMReranker(top_k=5),
    )
    print("RAG 管道就绪")

    print("\n=== 测试检索 ===")

    test_queries = [
        "推荐北京三天亲子游行程",
        "有什么适合情侣的浪漫旅行目的地",
        "重庆火锅哪家好吃",
        "西安有哪些必去的历史景点",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试 {i}: {query} ---")

        start_time = time.time()
        result = pipeline.run(query)
        elapsed = time.time() - start_time

        print(f"策略: {result.strategy}")
        print(f"优化查询: {result.optimized_queries}")
        print(f"子文档: {len(result.child_docs)}, 父文档: {len(result.parent_docs)}, "
              f"最终: {len(result.final_docs)}")
        print(f"耗时: {elapsed:.2f}秒")

        for j, doc in enumerate(result.final_docs, 1):
            score = doc.metadata.get("relevance_score", "N/A")
            source = doc.metadata.get("source", "unknown")
            parent_id = doc.metadata.get("parent_id", "?")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{j}] {score}分 source={source} parent={parent_id}")
            print(f"      {preview}...")

        if not result.final_docs:
            print("  (无结果)")

    print(f"\n=== 测试完成 ===")


if __name__ == "__main__":
    asyncio.run(main())
