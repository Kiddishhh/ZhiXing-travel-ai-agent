"""
RAG 管道集成测试脚本

初始化 RAG 系统 → 模拟真实查询输入 → 打印完整检索流程和结果对比。

运行方式: python scripts/test_rag_pipeline.py
需要: 先确保 .env 文件配置正确, data/documents/ 中有文档
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


SEP = "=" * 70
SUB = "─" * 70


def _load_all_documents(doc_manager):
    """加载所有分类文档"""
    documents = []
    for load_fn, label in [
        (doc_manager.load_destination_documents, "目的地"),
        (doc_manager.load_food_documents, "美食"),
        (doc_manager.load_accommodation_documents, "住宿"),
    ]:
        t0 = time.time()
        docs = load_fn()
        elapsed = time.time() - t0
        print(f"  [{label}] {len(docs)} 篇 (耗时 {elapsed:.2f}s)")
        documents.extend(docs)
    return documents


async def init_pipeline():
    """初始化 RAG 管道"""
    print(SEP)
    print("  RAG 管道集成测试 — 初始化阶段")
    print(SEP)
    total_start = time.time()

    # ── 1. 加载文档 ──
    t0 = time.time()
    print("\n[1/5] 加载文档 (DocumentManager)...")
    doc_manager = DocumentManager()
    documents = _load_all_documents(doc_manager)
    print(f"  → 共加载 {len(documents)} 篇文档 (耗时 {time.time()-t0:.2f}s)")

    if not documents:
        print("  ⚠ 未加载到任何文档! 请检查 data/documents/ 目录")
        return None

    # ── 2. 切分文档 ──
    t0 = time.time()
    print("\n[2/5] 切分文档 (ParentDocumentSplitter)...")
    print(f"  参数: parent_chunk=1000/200, child_chunk=200/50")
    splitter = ParentDocumentSplitter(
        parent_chunk_size=1000, parent_chunk_overlap=200,
        child_chunk_size=200, child_chunk_overlap=50,
    )
    parent_docs, child_docs = splitter.split_documents(documents)
    print(f"  → 父文档: {len(parent_docs)} 个, 子文档: {len(child_docs)} 个 "
          f"(耗时 {time.time()-t0:.2f}s)")

    # ── 3. 初始化 ChromaDB ──
    t0 = time.time()
    print("\n[3/5] 初始化 ChromaDB...")
    chroma_manager = ChromaManager()
    chroma_manager.delete_collection("travel_children")
    chroma_manager.delete_collection("travel_parents")
    print(f"  → ChromaDB 就绪, 旧 collection 已清理 (耗时 {time.time()-t0:.2f}s)")

    # ── 4. 构建检索器 + 双索引 ──
    t0 = time.time()
    print("\n[4/5] 构建混合检索器 + 双索引...")
    print(f"  检索器参数: bm25_weight=0.4, dense_weight=0.6, rrf_k=60")
    retriever = HybridRetriever(
        chroma_manager=chroma_manager,
        collection_name="travel_children",
    )
    retriever.initialize(child_docs)

    parent_ids = [p.metadata["parent_id"] for p in parent_docs]
    chroma_manager.add_documents(
        parent_docs, ids=parent_ids, collection_name="travel_parents",
    )
    print(f"  → travel_children: {len(child_docs)} 篇子文档 (BM25 + Dense)")
    print(f"  → travel_parents:  {len(parent_docs)} 篇父文档 (上下文扩展)")
    print(f"  (耗时 {time.time()-t0:.2f}s)")

    # ── 5. 创建管道 ──
    t0 = time.time()
    print("\n[5/5] 创建 RAG 管道...")
    optimizer = QueryOptimizer()
    reranker = LLMReranker(top_k=5)
    pipeline = RAGPipeline(
        optimizer=optimizer,
        retriever=retriever,
        chroma_manager=chroma_manager,
        reranker=reranker,
    )
    print(f"  → QueryOptimizer(model=qwen-turbo)")
    print(f"  → LLMReranker(top_k=5, model=qwen-turbo)")
    print(f"  → 管道就绪 (耗时 {time.time()-t0:.2f}s)")

    print(f"\n  初始化总耗时: {time.time()-total_start:.2f}s")
    return pipeline


def print_result(result, index: int, elapsed: float):
    """打印单个查询的完整结果"""
    print(f"\n{SUB}")
    print(f"查询 #{index}: \"{result.original_query}\"")
    print(SUB)
    print(f"  耗时:       {elapsed:.2f}s")
    print(f"  优化策略:   {result.strategy}")
    print(f"  优化查询:   {result.optimized_queries}")
    print(f"  子文档数:   {result.child_count}")
    print(f"  父文档数:   {result.parent_count}")
    print(f"  最终文档数: {len(result.final_docs)}")

    if result.final_docs:
        print(f"\n  ┌─ 重排序结果 (头尾高相关, 中间低相关) ─────────────────────")
        for i, doc in enumerate(result.final_docs):
            score = doc.metadata.get("relevance_score", "N/A")
            source = doc.metadata.get("source", "unknown")
            parent_id = doc.metadata.get("parent_id", "?")
            preview = doc.page_content[:100].replace("\n", " ")
            position = "★头" if i == 0 else ("★尾" if i == len(result.final_docs)-1 else "  ")
            print(f"  │ {position} [{score}分] source={source} id={parent_id}")
            print(f"  │    {preview}...")
        print(f"  └──────────────────────────────────────────────────────────")
    else:
        print(f"  ⚠ 无检索结果")


async def main():
    print("\n正在初始化 RAG 管道...")
    pipeline = await init_pipeline()

    if pipeline is None:
        print("\n初始化失败, 退出。")
        return

    # ── 测试查询 (模拟真实用户输入) ──
    test_queries = [
        "推荐北京三天亲子游行程",
        "有什么适合情侣的浪漫旅行目的地",
        "重庆火锅哪家好吃",
        "西安有哪些必去的历史景点",
    ]

    print(f"\n\n{SEP}")
    print(f"  检索测试阶段 — {len(test_queries)} 个查询")
    print(SEP)

    print(f"\n[模拟输入] 注入 {len(test_queries)} 个真实旅游查询:")
    for i, q in enumerate(test_queries, 1):
        print(f"  {i}. \"{q}\"")

    # ── 逐个执行查询 ──
    total_start = time.time()
    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n  ⏳ 执行查询 #{i}...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = pipeline.run(query)
            elapsed = time.time() - t0
            results.append((result, elapsed))
            print(f"完成 ({result.child_count} 子文档 → {len(result.final_docs)} 最终)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"失败: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start

    # ── 详细结果 ──
    for i, (result, elapsed) in enumerate(results, 1):
        print_result(result, i, elapsed)

    # ── 汇总对比 ──
    print(f"\n\n{SEP}")
    print(f"  结果汇总对比")
    print(SEP)
    print(f"{'#':<3} {'查询':<30} {'策略':<14} {'子文档':<8} {'最终':<6} {'耗时':<8}")
    print(f"{'─'*3} {'─'*30} {'─'*14} {'─'*8} {'─'*6} {'─'*8}")
    for i, (result, elapsed) in enumerate(results, 1):
        query_short = result.original_query[:28] + (".." if len(result.original_query) > 30 else "")
        print(f"{i:<3} {query_short:<30} {result.strategy:<14} "
              f"{result.child_count:<8} {len(result.final_docs):<6} {elapsed:.2f}s")

    print(f"\n  总耗时: {total_elapsed:.2f}s")
    print(f"  平均: {total_elapsed/len(test_queries):.2f}s/查询")
    print(f"\n{SEP}")
    print("  集成测试完成")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())
