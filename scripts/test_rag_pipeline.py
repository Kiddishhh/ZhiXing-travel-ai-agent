"""
RAG 管道集成测试脚本

初始化 RAG 系统 → 执行测试查询 → 打印结果摘要。
"""
import asyncio
import sys
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
from app.utils.logger import app_logger


def _load_all_documents(doc_manager):
    """加载所有分类文档"""
    documents = []
    for load_fn, label in [
        (doc_manager.load_destination_documents, "目的地"),
        (doc_manager.load_food_documents, "美食"),
        (doc_manager.load_accommodation_documents, "住宿"),
    ]:
        docs = load_fn()
        documents.extend(docs)
        print(f"  {label}: {len(docs)} 篇")
    return documents


async def init_pipeline():
    """初始化 RAG 管道"""
    print("=" * 60)
    print("RAG 管道集成测试")
    print("=" * 60)

    # 1. 加载文档
    print("\n[1/5] 加载文档...")
    doc_manager = DocumentManager()
    documents = _load_all_documents(doc_manager)
    print(f"  共加载 {len(documents)} 篇文档")

    # 2. 切分文档
    print("\n[2/5] 切分文档...")
    splitter = ParentDocumentSplitter(
        parent_chunk_size=1000, parent_chunk_overlap=200,
        child_chunk_size=200, child_chunk_overlap=50,
    )
    parent_docs, child_docs = splitter.split_documents(documents)
    print(f"  父文档: {len(parent_docs)}, 子文档: {len(child_docs)}")

    # 3. 初始化 ChromaDB
    print("\n[3/5] 初始化 ChromaDB...")
    chroma_manager = ChromaManager()
    chroma_manager.delete_collection("travel_children")
    chroma_manager.delete_collection("travel_parents")

    # 4. 构建检索器 + 索引
    print("\n[4/5] 构建检索器 + 索引...")
    retriever = HybridRetriever(
        chroma_manager=chroma_manager,
        collection_name="travel_children",
    )
    retriever.initialize(child_docs)

    # 索引父文档
    parent_ids = [p.metadata["parent_id"] for p in parent_docs]
    chroma_manager.add_documents(
        parent_docs, ids=parent_ids, collection_name="travel_parents",
    )
    print(f"  子文档索引: {len(child_docs)} 篇 → travel_children")
    print(f"  父文档索引: {len(parent_docs)} 篇 → travel_parents")

    # 5. 创建管道
    print("\n[5/5] 创建 RAG 管道...")
    optimizer = QueryOptimizer()
    reranker = LLMReranker(top_k=5)
    pipeline = RAGPipeline(
        optimizer=optimizer,
        retriever=retriever,
        chroma_manager=chroma_manager,
        reranker=reranker,
    )
    print("  管道就绪")
    return pipeline


def print_result(result, index: int):
    """打印单个查询结果"""
    print(f"\n{'─' * 60}")
    print(f"查询 {index}: {result.original_query}")
    print(f"{'─' * 60}")
    print(f"  策略: {result.strategy}")
    print(f"  优化查询: {result.optimized_queries}")
    print(f"  子文档数: {result.child_count}")
    print(f"  父文档数: {result.parent_count}")
    print(f"  最终文档数: {len(result.final_docs)}")
    if result.final_docs:
        print(f"\n  重排序结果:")
        for i, doc in enumerate(result.final_docs):
            score = doc.metadata.get("relevance_score", "N/A")
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"    {i+1}. [{score}分] ({source}) {preview}...")
    else:
        print(f"  (无结果)")


async def main():
    print("\n正在初始化 RAG 管道...\n")
    pipeline = await init_pipeline()

    # 测试查询
    test_queries = [
        "推荐北京三天亲子游行程",
        "有什么适合情侣的浪漫旅行目的地",
        "重庆火锅哪家好吃",
        "西安有哪些必去的历史景点",
    ]

    print(f"\n{'=' * 60}")
    print(f"开始测试 {len(test_queries)} 个查询")
    print(f"{'=' * 60}")

    for i, query in enumerate(test_queries, 1):
        try:
            result = pipeline.run(query)
            print_result(result, i)
        except Exception as e:
            print(f"\n  查询失败: {e}")

    print(f"\n{'=' * 60}")
    print("集成测试完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
